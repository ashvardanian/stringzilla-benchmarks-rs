# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "stringzillas-cpus>=5.0.0",
#   "datasketch",
#   "numpy",
#   "tqdm",
# ]
# ///
"""
Fingerprinting benchmarks in Python: docs/s for MinHash operations.

- MinHash: datasketch.MinHash, stringzillas.Fingerprints, cudf.minhash_ngrams,
  and cudf.minhash64_ngrams (if RAPIDS cuDF is installed).

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')
- STRINGWARS_BATCH_PER_CORE: Items processed per core (default: 128)

The only batch knob is STRINGWARS_BATCH_PER_CORE (items per core); the per-device batch is
auto-derived from the hardware core count — one CPU core is a core, one GPU streaming
multiprocessor (SM) is a core — so each device is fed enough work without manual scaling.
The --batch-size flag overrides STRINGWARS_BATCH_PER_CORE as the per-core base.

Examples:
  uv run --with stringzillas-cpus fingerprints/bench.py --dataset README.md --tokens lines
  uv run --with stringzillas-cpus fingerprints/bench.py --dataset data/xlsum/xlsum.csv --tokens words -k "datasketch"
  STRINGWARS_DATASET=README.md STRINGWARS_TOKENS=lines uv run --with stringzillas-cpus fingerprints/bench.py
"""

import argparse
import os
import re
import sys

import numpy as np
import stringzilla as sz
import stringzillas as szs
from datasketch import MinHash
from tqdm import tqdm

from utils import (
    add_common_args,
    auto_batch_size,
    clamped_subranges,
    gpu_multiprocessor_count,
    load_dataset,
    now_nanoseconds,
    report_stats,
    should_run,
    tokenize_dataset,
)

# For RAPIDS cuDF GPU-accelerated MinHash
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Fixed n-gram widths for multi-scale fingerprinting (matching Rust benchmark)
NGRAM_WIDTHS = [5, 9, 17, 33]
NGRAM_WIDTHS_ARRAY = np.array(NGRAM_WIDTHS, dtype=np.uint64)

# Default per-core batch base for fingerprinting (items processed per core).
DEFAULT_BATCH_PER_CORE = 128


def log_system_info():
    """Log Python version and fingerprinting library versions."""

    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- DataSketch: {MinHash.__module__.split('.')[0]} (available)")
    if CUDF_AVAILABLE:
        print(f"- CuDF: {cudf.__version__}")
    print()  # Add blank line


def bench_fingerprint(name, documents, kernel, doc_bytes, dimensions, time_limit_seconds, batch_size):
    """Time kernel over documents in batches until the deadline; report hashes/s and bytes/s."""
    count = len(documents)
    deadline_nanoseconds = now_nanoseconds() + int(time_limit_seconds * 1e9)
    start_time = now_nanoseconds()
    processed = 0
    bar = tqdm(total=count, desc=name, unit="docs", leave=False)
    try:
        for low, high in clamped_subranges(count, batch_size):
            if now_nanoseconds() >= deadline_nanoseconds:
                break
            kernel(documents[low:high])
            processed = high
            bar.update(high - low)
    finally:
        bar.close()

    elapsed_seconds = (now_nanoseconds() - start_time) / 1e9
    processed_bytes = int(doc_bytes[:processed].sum())
    # Hash operations mirror the Rust harness: dimensions hash updates per byte of each
    # processed document, so total_hash_ops = dimensions * bytes spanned by processed docs.
    total_hash_ops = dimensions * processed_bytes
    report_stats(name, "hashes", elapsed_seconds, total_hash_ops, processed_bytes)


def document_byte_lengths(documents):
    return np.fromiter((len(document.encode("utf-8")) for document in documents), dtype=np.int64, count=len(documents))


def benchmark_stringzillas(documents, dimensions, batch_size, time_limit_seconds, filter_pattern):
    """StringZilla Fingerprints on 1 core, all cores, and the GPU (if present)."""
    cpu_cores = os.cpu_count()

    moved = sz.Strs(documents)
    doc_bytes = document_byte_lengths(documents)

    all_cpu_batch_size = auto_batch_size(cpu_cores, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE
    )

    def run_variant(suffix, scope_arguments, variant_batch_size):
        """Opens this variant's scope, measures it, and drops the scope on return: an idle ForkUnion
        pool spin-waits, so a multi-core scope held across the single-core variant would consume the
        very cores that variant is being measured on."""
        scope = szs.DeviceScope(**scope_arguments)
        engine = szs.Fingerprints(ndim=dimensions, window_widths=NGRAM_WIDTHS_ARRAY, capabilities=scope)

        def kernel(strs_slice):
            engine(strs_slice, device=scope)  # returns (hashes, counts); discarded for throughput

        bench_fingerprint(
            f"stringzillas.Fingerprints{suffix}",
            moved,
            kernel,
            doc_bytes,
            dimensions,
            time_limit_seconds,
            variant_batch_size,
        )
        # Scope and engine die with this call, so the next variant starts on a quiet machine.

    if should_run("minhash/stringzillas.Fingerprints<1cpu>", filter_pattern):
        run_variant("<1cpu>", {}, 1)
    if should_run(f"minhash/stringzillas.Fingerprints<{cpu_cores}cpu,batch={all_cpu_batch_size}>", filter_pattern):
        run_variant(f"<{cpu_cores}cpu,batch={all_cpu_batch_size}>", {"cpu_cores": cpu_cores}, all_cpu_batch_size)
    if "cuda" in szs.__capabilities__ and should_run(
        f"minhash/stringzillas.Fingerprints<1gpu,batch={gpu_batch_size}>", filter_pattern
    ):
        run_variant(f"<1gpu,batch={gpu_batch_size}>", {"gpu_device": 0}, gpu_batch_size)


def benchmark_datasketch(documents, dimensions, batch_size, time_limit_seconds, filter_pattern):
    """datasketch MinHash on CPU: the common data-science baseline, n-grams built in Python."""
    if not should_run("minhash/datasketch.MinHash", filter_pattern):
        return
    cpu_batch_size = auto_batch_size(1, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    per_width = max(1, dimensions // len(NGRAM_WIDTHS))
    doc_bytes = document_byte_lengths(documents)

    def kernel(slice_of_documents):
        for document in slice_of_documents:
            data = document.encode("utf-8")
            for width in NGRAM_WIDTHS:
                signature = MinHash(num_perm=per_width)
                for offset in range(len(data) - width + 1):
                    signature.update(data[offset : offset + width])
                _ = signature.hashvalues  # force materialization

    bench_fingerprint(
        "datasketch.MinHash", documents, kernel, doc_bytes, dimensions, time_limit_seconds, cpu_batch_size
    )


def benchmark_cudf(documents, dimensions, batch_size, time_limit_seconds, filter_pattern):
    """cuDF MinHash on the GPU: the CUDA first-party comparison (optional, best-effort)."""
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE
    )
    if not should_run(f"minhash/cudf.minhash<1gpu,batch={gpu_batch_size}>", filter_pattern):
        return
    try:
        import cupy as cp
    except ImportError:
        print("cudf.minhash<1gpu>: SKIPPED (cupy not available)")
        return

    per_width = max(1, dimensions // len(NGRAM_WIDTHS))
    parameters_a = cp.arange(1, per_width + 1, dtype=cp.uint32)
    parameters_b = cp.arange(1, per_width + 1, dtype=cp.uint32)
    doc_bytes = document_byte_lengths(documents)
    series = cudf.Series(documents)

    def kernel(series_slice):
        for width in NGRAM_WIDTHS:
            series_slice.str.minhash(seed=0, a=parameters_a, b=parameters_b, width=width)

    try:
        bench_fingerprint(
            f"cudf.minhash<1gpu,batch={gpu_batch_size}>",
            series,
            kernel,
            doc_bytes,
            dimensions,
            time_limit_seconds,
            gpu_batch_size,
        )
    except Exception as error:
        print(f"cudf.minhash<1gpu>: SKIPPED ({type(error).__name__}: {error})")


_main_epilog = """
Examples:

  # Benchmark all algorithms with default settings
  %(prog)s --dataset data/leipzig-1m/leipzig1M_en.txt

  # Benchmark with limited docs and specific dimensions
  %(prog)s --dataset data/leipzig-1m/leipzig1M_en.txt --max-docs 1000 --dimensions 128

  # Test only specific algorithms
  %(prog)s --dataset data/leipzig-1m/leipzig1M_en.txt -k "(datasketch|szs.Fingerprints)"

  # GPU-only benchmarks
  %(prog)s --dataset data/leipzig-1m/leipzig1M_en.txt -k "(cudf|GPU)"

  # High-throughput batch processing
  %(prog)s --dataset data/leipzig-1m/leipzig1M_en.txt --batch-size 1024
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla fingerprinting operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser, default_dataset="data/xlsum/xlsum.csv", default_dataset_limit="64mb")
    parser.add_argument("-n", "--max-docs", type=int, help="Maximum number of docs to process")
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        default=256,
        help="Number of hash functions for MinHash (default: 256)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="Items processed per core (overrides STRINGWARS_BATCH_PER_CORE, default: 128)",
    )

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load and tokenize dataset
    dataset = load_dataset(args.dataset, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(dataset, args.tokens)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Limit number of documents if specified
    if args.max_docs is not None:
        tokens = tokens[: args.max_docs]

    docs_sizes = [len(doc.encode("utf-8")) for doc in tokens]
    total_bytes = sum(docs_sizes)
    avg_doc_length = total_bytes / len(tokens) if tokens else 0

    print(f"Dataset: {len(tokens):,} docs, {total_bytes:,} bytes, {avg_doc_length:.1f} avg doc length")
    log_system_info()

    print("\nMinHash Throughput")
    benchmark_stringzillas(tokens, args.dimensions, args.batch_size, args.time_limit, filter_pattern)
    benchmark_datasketch(tokens, args.dimensions, args.batch_size, args.time_limit, filter_pattern)
    if CUDF_AVAILABLE:
        benchmark_cudf(tokens, args.dimensions, args.batch_size, args.time_limit, filter_pattern)
    return 0


if __name__ == "__main__":
    exit(main())
