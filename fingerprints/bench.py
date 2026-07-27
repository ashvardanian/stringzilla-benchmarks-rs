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
"""MinHash fingerprinting benchmarks in Python across CPU and GPU. Mirrors `fingerprints/bench.rs`."""

import argparse
import sys

import numpy as np
import stringzilla as sz
import stringzillas as szs
from datasketch import MinHash

from utils import (
    MeasureSpec,
    add_common_args,
    auto_batch_size,
    finish,
    gpu_multiprocessor_count,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    resolve_core_count,
    resolve_dataset,
    set_filter,
    should_run,
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


def bench_fingerprint(name, documents, kernel, doc_bytes, dimensions, batch_size):
    """One pass sketches every document, in batches; reports hashes/s and bytes/s."""
    count = len(documents)
    total_bytes = int(doc_bytes.sum())
    # Hash operations mirror the Rust harness: `dimensions` hash updates per byte.
    work = MeasureSpec(report="hashes", elements=dimensions * total_bytes, total_bytes=total_bytes)

    def one_pass() -> None:
        for low in range(0, count, batch_size):
            kernel(documents[low : min(low + batch_size, count)])

    measure(name, work, one_pass)


def document_byte_lengths(documents):
    return np.fromiter((len(document.encode("utf-8")) for document in documents), dtype=np.int64, count=len(documents))


def benchmark_stringzillas(documents, dimensions, batch_size):
    """StringZilla Fingerprints on 1 core, all cores, and the GPU (if present)."""
    cpu_cores = resolve_core_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    moved = sz.Strs(documents)
    doc_bytes = document_byte_lengths(documents)

    all_cpu_batch_size = auto_batch_size(cpu_cores, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64,
        base=batch_size,
        default_base=DEFAULT_BATCH_PER_CORE,
    )

    def run_variant(suffix, scope, variant_batch_size):
        engine = szs.Fingerprints(ndim=dimensions, window_widths=NGRAM_WIDTHS_ARRAY, capabilities=scope)

        def kernel(strs_slice):
            engine(strs_slice, device=scope)  # returns (hashes, counts); discarded for throughput

        bench_fingerprint(
            f"stringzillas.Fingerprints{suffix}",
            moved,
            kernel,
            doc_bytes,
            dimensions,
            variant_batch_size,
        )

    run_variant("<1cpu>", default_scope, 1)
    if should_run(f"minhash/stringzillas.Fingerprints<{cpu_cores}cpu,batch={all_cpu_batch_size}>"):
        run_variant(f"<{cpu_cores}cpu,batch={all_cpu_batch_size}>", cpu_scope, all_cpu_batch_size)
    if gpu_scope is not None and should_run(
        f"minhash/stringzillas.Fingerprints<1gpu,batch={gpu_batch_size}>",
    ):
        run_variant(f"<1gpu,batch={gpu_batch_size}>", gpu_scope, gpu_batch_size)


def benchmark_datasketch(documents, dimensions, batch_size):
    """datasketch MinHash on CPU: the common data-science baseline, n-grams built in Python."""
    if not should_run("minhash/datasketch.MinHash"):
        return
    cpu_batch_size = auto_batch_size(1, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    per_width = max(1, dimensions // len(NGRAM_WIDTHS))
    doc_bytes = document_byte_lengths(documents)
    # Encoded once: doing it inside the kernel charged datasketch a full UTF-8
    # encode of the working set on every pass that the StringZilla rows never pay.
    encoded = [document.encode("utf-8") for document in documents]

    def kernel(slice_of_documents):
        for data in slice_of_documents:
            for width in NGRAM_WIDTHS:
                signature = MinHash(num_perm=per_width)
                for offset in range(len(data) - width + 1):
                    signature.update(data[offset : offset + width])
                _ = signature.hashvalues  # force materialization

    bench_fingerprint(
        "datasketch.MinHash",
        encoded,
        kernel,
        doc_bytes,
        dimensions,
        cpu_batch_size,
    )


def benchmark_cudf(documents, dimensions, batch_size):
    """cuDF MinHash on the GPU: the CUDA first-party comparison (optional, best-effort)."""
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64,
        base=batch_size,
        default_base=DEFAULT_BATCH_PER_CORE,
    )
    if not should_run(f"minhash/cudf.minhash<1gpu,batch={gpu_batch_size}>"):
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
            gpu_batch_size,
        )
    except Exception as error:
        print(f"cudf.minhash<1gpu>: SKIPPED ({type(error).__name__}: {error})")


_main_epilog = """
Examples:

  %(prog)s --dataset leipzig1M.txt

  %(prog)s --dataset leipzig1M.txt --max-docs 1000 --dimensions 128

  # Test only specific algorithms
  %(prog)s --dataset leipzig1M.txt -k "(datasketch|szs.Fingerprints)"

  # GPU-only benchmarks
  %(prog)s --dataset leipzig1M.txt -k "(cudf|GPU)"

  # High-throughput batch processing
  %(prog)s --dataset leipzig1M.txt --batch-size 1024
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla fingerprinting operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)
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
    set_filter(args.filter)

    # Load and tokenize dataset
    dataset = resolve_dataset("fingerprints", as_bytes=False, dataset_path=args.dataset)
    tokens = dataset.tokens
    log_dataset(dataset)
    log_timing_overhead()

    # Limit number of documents if specified
    if args.max_docs is not None:
        tokens = tokens[: args.max_docs]

    log_system_info()

    print("\nMinHash Throughput")
    benchmark_stringzillas(tokens, args.dimensions, args.batch_size)
    benchmark_datasketch(tokens, args.dimensions, args.batch_size)
    if not CUDF_AVAILABLE:
        note_unavailable("minhash/cudf.minhash<1gpu>", "cudf not installed")
    else:
        benchmark_cudf(tokens, args.dimensions, args.batch_size)
    finish()
    return 0


if __name__ == "__main__":
    exit(main())
