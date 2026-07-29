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
    get_env,
    get_env_or_default,
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
            kernel(documents[low : low + batch_size])

    measure(name, work, one_pass)


def document_byte_lengths(documents):
    return np.fromiter((len(document.encode("utf-8")) for document in documents), dtype=np.int64, count=len(documents))


def benchmark_stringzillas(documents, doc_bytes, dimensions, batch_size):
    """StringZilla Fingerprints on 1 core, all cores, and the GPU (if present)."""
    cpu_cores = resolve_core_count()
    default_scope = szs.DeviceScope()
    cpu_scope = szs.DeviceScope(cpu_cores=cpu_cores)
    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None

    moved = sz.Strs(documents)

    single_cpu_batch_size = auto_batch_size(1, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    all_cpu_batch_size = auto_batch_size(cpu_cores, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64,
        base=batch_size,
        default_base=DEFAULT_BATCH_PER_CORE,
    )

    def run_variant(name, scope, variant_batch_size):
        engine = szs.Fingerprints(ndim=dimensions, window_widths=NGRAM_WIDTHS_ARRAY, capabilities=scope)

        def kernel(strs_slice):
            engine(strs_slice, device=scope)  # returns (hashes, counts); discarded for throughput

        bench_fingerprint(name, moved, kernel, doc_bytes, dimensions, variant_batch_size)

    # Row names carry no batch size, matching what bench.rs prints, so `-k` selects the
    # same rows in both harnesses.
    single_cpu_name = "minhash/stringzillas.Fingerprints<1cpu>"
    all_cpu_name = f"minhash/stringzillas.Fingerprints<{cpu_cores}cpu>"
    gpu_name = "minhash/stringzillas.Fingerprints<1gpu>"

    if should_run(single_cpu_name):
        run_variant(single_cpu_name, default_scope, single_cpu_batch_size)
    if should_run(all_cpu_name):
        run_variant(all_cpu_name, cpu_scope, all_cpu_batch_size)
    if gpu_scope is not None and should_run(gpu_name):
        run_variant(gpu_name, gpu_scope, gpu_batch_size)


def benchmark_datasketch(documents, doc_bytes, dimensions, batch_size):
    """datasketch MinHash on CPU: the common data-science baseline, n-grams built in Python."""
    if not should_run("minhash/datasketch.MinHash"):
        return
    cpu_batch_size = auto_batch_size(1, base=batch_size, default_base=DEFAULT_BATCH_PER_CORE)
    per_width = max(1, dimensions // len(NGRAM_WIDTHS))
    # Encoded once: doing it inside the kernel charged datasketch a full UTF-8
    # encode of the working set on every pass that the StringZilla rows never pay.
    encoded = [document.encode("utf-8") for document in documents]
    # A fresh sketch per document is what the algorithm requires; a fresh permutation
    # table is not, and regenerating one per document per width is pure setup cost.
    prototype = MinHash(num_perm=per_width)
    permutations, scheme = prototype.permutations, prototype.scheme

    def kernel(slice_of_documents):
        for data in slice_of_documents:
            for width in NGRAM_WIDTHS:
                signature = MinHash(num_perm=per_width, permutations=permutations, scheme=scheme)
                signature.update_batch(data[offset : offset + width] for offset in range(len(data) - width + 1))

    bench_fingerprint(
        "minhash/datasketch.MinHash",
        encoded,
        kernel,
        doc_bytes,
        dimensions,
        cpu_batch_size,
    )


def benchmark_cudf(documents, doc_bytes, dimensions, batch_size):
    """cuDF MinHash on the GPU: the CUDA first-party comparison (optional, best-effort)."""
    gpu_batch_size = auto_batch_size(
        gpu_multiprocessor_count(0) or 64,
        base=batch_size,
        default_base=DEFAULT_BATCH_PER_CORE,
    )
    name = "minhash/cudf.minhash<1gpu>"
    if not should_run(name):
        return
    try:
        import cupy as cp
    except ImportError:
        print(f"{name}: SKIPPED (cupy not available)")
        return

    per_width = max(1, dimensions // len(NGRAM_WIDTHS))
    parameters_a = cp.arange(1, per_width + 1, dtype=cp.uint32)
    parameters_b = cp.arange(1, per_width + 1, dtype=cp.uint32)
    series = cudf.Series(documents)

    def kernel(series_slice):
        for width in NGRAM_WIDTHS:
            series_slice.str.minhash(seed=0, a=parameters_a, b=parameters_b, width=width)

    try:
        bench_fingerprint(
            name,
            series,
            kernel,
            doc_bytes,
            dimensions,
            gpu_batch_size,
        )
    except Exception as error:
        print(f"{name}: SKIPPED ({type(error).__name__}: {error})")


_main_epilog = """
Examples:

  %(prog)s --dataset leipzig1M.txt

  %(prog)s --dataset leipzig1M.txt --max-docs 1000 --dimensions 128

  # Test only specific algorithms
  %(prog)s --dataset leipzig1M.txt -k "(datasketch|stringzillas.Fingerprints)"

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
        default=None,
        help="Pin one MinHash width; otherwise STRINGWARS_NDIM_SCALES sweeps 64,128,256,512",
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

    # Encoding the corpus is not free, and every row prices its work off the same lengths.
    doc_bytes = document_byte_lengths(tokens)

    # Sweep the same widths as `bench.rs`: `--dimensions` pins one, otherwise
    # STRINGWARS_NDIM / STRINGWARS_NDIM_SCALES decide, so both languages emit rows at the
    # same scales. Python used to be fixed at 256 while Rust swept 64/128/256/512, which
    # left every scale but one with no Python counterpart to compare against.
    if args.dimensions is not None:
        scales = [args.dimensions]
    elif pinned := get_env("STRINGWARS_NDIM"):
        scales = [int(pinned)]
    else:
        scales = [int(s) for s in get_env_or_default("STRINGWARS_NDIM_SCALES", "64,128,256,512").split(",")]

    for dimensions in scales:
        print(f"\n# minhash/ndim_{dimensions}")
        benchmark_stringzillas(tokens, doc_bytes, dimensions, args.batch_size)
        benchmark_datasketch(tokens, doc_bytes, dimensions, args.batch_size)
        if not CUDF_AVAILABLE:
            note_unavailable("minhash/cudf.minhash<1gpu>", "cudf not installed")
        else:
            benchmark_cudf(tokens, doc_bytes, dimensions, args.batch_size)
    finish()
    return 0


if __name__ == "__main__":
    exit(main())
