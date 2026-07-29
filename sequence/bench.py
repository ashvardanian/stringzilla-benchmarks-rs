#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "polars",
# ]
# ///
"""Sorting benchmarks in Python, reported in comparisons/s. Mirrors `sequence/bench.rs`."""

import argparse
import functools
import math
import os
import sys
from collections.abc import Callable

# Must precede `import polars`: Polars sizes its thread pool once, at import. Every other
# engine in this suite sorts on one core, so leaving Polars on all 18 made its rows 5-10x
# faster than the contenders they sit beside.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

# Assume core deps are present; only cuDF is optional
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa
import pyarrow.compute as pc
import stringzilla as sz

try:
    import cudf

    CUDF_AVAILABLE = True
except Exception:
    cudf = None  # type: ignore
    CUDF_AVAILABLE = False
else:
    # cuDF sorts run on GPU; nothing to set for CPU threads here
    pass

from utils import (
    MeasureSpec,
    add_common_args,
    finish,
    log_dataset,
    log_timing_overhead,
    measure_with_setup,
    resolve_dataset,
    set_filter,
    should_run,
)


def log_system_info():
    """Log Python version and sequence library versions."""

    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- Pandas: {pd.__version__}")
    print(f"- PyArrow: {pa.__version__}")
    print(f"- Polars: {pl.__version__} ({pl.thread_pool_size()} threads)")
    if CUDF_AVAILABLE:
        print(f"- CuDF: {cudf.__version__}")
    print()  # Add blank line


def bench_sort_operation(
    name: str,
    build_input: Callable[[], object],
    sort_input: Callable[[object], object],
    n_items: int,
    token_bytes: int,
):
    """Repeatedly sort the same unsorted data under a wall-clock time budget.

    Each pass rebuilds the input from the original unsorted tokens via `build_input`
    (so every pass sorts identical data and only the sort itself is timed), runs
    `sort_input`, and accumulates the elapsed time and the comparison-count estimate.
    """
    # A pass is one full sort; the input is rebuilt inside the pass because sorting
    # in place would leave the next pass with already-sorted input.
    comparisons_per_pass = n_items * math.log2(max(n_items, 2))
    work = MeasureSpec(report="comparisons", elements=int(comparisons_per_pass), total_bytes=token_bytes)
    result = None

    def run(unsorted_input):
        nonlocal result
        result = sort_input(unsorted_input)

    # Sorting consumes its input, so the rebuild happens outside the timed span.
    measure_with_setup(name, work, build_input, run)
    return result


_main_epilog = """
Examples:

  %(prog)s --dataset README.md --tokens lines

  # Test only Python list.sort
  %(prog)s --dataset data.txt --tokens lines -k "list.sort"

  # Compare StringZilla vs other libraries
  %(prog)s --dataset large.txt --tokens words -k "stringzilla.Strs|pandas|polars"

  # GPU-accelerated sorting (if cuDF available)
  %(prog)s --dataset text.txt --tokens lines -k "cudf"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark string sorting operations with various libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    set_filter(args.filter)

    # Load and tokenize dataset
    dataset = resolve_dataset("sequence", as_bytes=False, dataset_path=args.dataset)
    log_dataset(dataset)
    log_timing_overhead()
    tokens = dataset.tokens

    if not tokens:
        print("No tokens found in dataset")
        return 1

    total_chars = sum(len(token) for token in tokens)
    avg_token_length = total_chars / len(tokens)
    print(f"Dataset: {len(tokens):,} tokens, {total_chars:,} chars, {avg_token_length:.1f} avg token length")
    log_system_info()

    print("\nSort Benchmarks")

    n_items = len(tokens)
    token_bytes = dataset.token_bytes

    # Python list.sort — mutates in place, so rebuild a fresh copy of the unsorted
    # tokens before every timed pass.

    def std_sort(py_list):
        py_list.sort()
        return py_list

    bench_sort_operation("argsort/list.sort", lambda: list(tokens), std_sort, n_items, token_bytes)
    # Case-insensitive list.sort driven by StringZilla's own pairwise Unicode case-folding
    # comparator, `sz.utf8_uncased_order`, adapted to a sort key via `functools.cmp_to_key`.
    # CPython's sort takes only a key (no `cmp=`), so the comparator is wrapped per element;
    # holding the folding identical to `stringzilla.Strs.sorted(uncased=True)` isolates the sort
    # algorithm itself (CPython's Timsort vs StringZilla's radix sort).
    uncased_key = functools.cmp_to_key(sz.utf8_uncased_order)

    def std_sort_uncased(py_list):
        py_list.sort(key=uncased_key)
        return py_list

    bench_sort_operation("argsort/list.sort<uncased>", lambda: list(tokens), std_sort_uncased, n_items, token_bytes)
    # StringZilla — rebuild the Strs view each pass so every pass sorts identical data.
    bench_sort_operation(
        "argsort/stringzilla.Strs.sorted",
        lambda: sz.Strs(tokens),
        lambda strs: strs.sorted(),
        n_items,
        token_bytes,
    )

    # StringZilla case-insensitive sort: orders by Unicode case-folding natively.
    bench_sort_operation(
        "argsort/stringzilla.Strs.sorted<uncased>",
        lambda: sz.Strs(tokens),
        lambda strs: strs.sorted(uncased=True),
        n_items,
        token_bytes,
    )

    # StringZilla argsort: writes the index permutation into a caller-owned NumPy buffer
    # (`out=`), so no per-pass allocation — the same zero-copy reuse the other argsort engines
    # get. The buffer holds `sz_sorted_idx_t` indices (pointer-sized unsigned), i.e. `np.uintp`.
    argsort_out = np.empty(n_items, dtype=np.uintp)
    bench_sort_operation(
        "argsort/stringzilla.Strs.argsort",
        lambda: sz.Strs(tokens),
        lambda strs: strs.argsort(out=argsort_out),
        n_items,
        token_bytes,
    )
    # StringZilla case-insensitive argsort: index permutation under Unicode case-folding, same
    # caller-owned `out=` buffer so the only difference from the cased row is the comparator.
    argsort_out_uncased = np.empty(n_items, dtype=np.uintp)
    bench_sort_operation(
        "argsort/stringzilla.Strs.argsort<uncased>",
        lambda: sz.Strs(tokens),
        lambda strs: strs.argsort(uncased=True, out=argsort_out_uncased),
        n_items,
        token_bytes,
    )
    # NumPy (object-dtype array; the most familiar Python baseline). argsort is
    # non-mutating, so the prebuilt array is the same unsorted input every pass.
    np_array = np.array(tokens, dtype=object)
    bench_sort_operation(
        "argsort/numpy.argsort",
        lambda: np_array,
        lambda array: np.argsort(array, kind="stable"),
        n_items,
        token_bytes,
    )
    # Pandas (sort_values returns a new Series; the source stays unsorted). Force a stable sort
    # (`kind="stable"`); the default `quicksort` is unstable, and StringZilla's sort is always
    # stable, so a stable comparator keeps the head-to-head honest.
    s = pd.Series(tokens)
    bench_sort_operation(
        "argsort/pandas.Series.sort_values",
        lambda: s,
        lambda series: series.sort_values(ignore_index=True, kind="stable"),
        n_items,
        token_bytes,
    )
    # PyArrow. `string` carries 32-bit offsets, so a tape past that needs `large_string`.
    INT32_MAX = 2_147_483_647
    use_large = dataset.token_bytes > INT32_MAX
    arr = pa.array(tokens, type=pa.large_string() if use_large else pa.string())

    bench_sort_operation(
        "argsort/pyarrow.compute.sort_indices",
        lambda: arr,
        lambda array: pc.sort_indices(array),
        n_items,
        token_bytes,
    )
    # Polars argsort: returns an index Series (no materialization).
    ps = pl.Series(tokens)
    bench_sort_operation(
        "argsort/polars.Series.arg_sort",
        lambda: ps,
        lambda series: series.arg_sort(),
        n_items,
        token_bytes,
    )
    # Polars full sort (returns a new materialized Series; the source stays unsorted).
    ps = pl.Series(tokens)
    bench_sort_operation("argsort/polars.Series.sort", lambda: ps, lambda series: series.sort(), n_items, token_bytes)
    # cuDF GPU (if available; sort_values returns a new Series).
    if CUDF_AVAILABLE and should_run("argsort/cudf.Series.sort_values<1gpu>"):
        cs = cudf.Series(tokens)
        bench_sort_operation(
            "cudf.Series.sort_values<1gpu>",
            lambda: cs,
            lambda series: series.sort_values(ignore_index=True),
            n_items,
            token_bytes,
        )

    finish()
    return 0


if __name__ == "__main__":
    exit(main())
