# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "stringzillas-cpus>=5.0.0",
#   "rapidfuzz",
#   "python-Levenshtein",
#   "levenshtein",
#   "jellyfish",
#   "editdistance",
#   "polyleven",
#   "edlib",
#   "nltk",
#   "biopython",
#   "numpy",
# ]
# ///
"""
Similarity benchmarks in Python: MCUPS for string similarity operations.

The input file is tokenized into lines or words. The StringZilla engines evaluate a square
``side x side`` cross-product: the first ``side`` tokens (queries) against the next ``side``
disjoint tokens (candidates), producing a dense ``side x side`` similarity matrix in one native
call. As most algorithms have quadratic complexity and use Dynamic Programming techniques, their
throughput is reported in CUPS (Cell Updates Per Second). This mirrors the Rust harness `bench.rs`.

- Edit Distance baselines: rapidfuzz, python-Levenshtein, jellyfish, editdistance, nltk, edlib, polyleven
- StringZilla cross-product: szs.LevenshteinDistances, szs.LevenshteinDistancesUTF8,
  szs.NeedlemanWunschScores, szs.SmithWatermanScores
- BioPython: PairwiseAligner baseline (unary match/mismatch scoring)
- cuDF: GPU-accelerated edit distance (optional)

Environment variables (identical to bench.rs / the C++ harness):
- STRINGWARS_DATASET: Path to the input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')
- STRINGWARS_MAX_TOKENS: Limit on the number of tokens loaded
- STRINGWARS_BATCH_PER_CORE: Pairs processed per core (default: 256)
- STRINGWARS_TIME: Wall-time budget per benchmark variant (seconds)
- STRINGWARS_WARMUP: Uncounted warm-up budget per variant (seconds)
- STRINGWARS_SEED: Seed for the token shuffle
- STRINGWARS_FILTER: Regex selecting which benchmark variants run

A CPU core counts as one core; a GPU streaming multiprocessor (SM) counts as one core. The
per-device pair budget is ``STRINGWARS_BATCH_PER_CORE * cores``, and the square cross-product side
is ``round(sqrt(budget))`` clamped so queries ``[0, side)`` and candidates ``[side, 2*side)`` stay
disjoint, i.e. ``2 * side <= num_tokens``.

Examples:
  uv run --with stringzillas-cpus similarities/bench.py --dataset README.md
  STRINGWARS_DATASET=README.md STRINGWARS_TOKENS=lines uv run --with stringzillas-cpus similarities/bench.py
"""

import argparse
import os
import random
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# String similarity libraries
import stringzilla as sz
import stringzillas as szs

from utils import (
    add_common_args,
    auto_batch_size,
    get_env_parsed,
    gpu_multiprocessor_count,
    load_dataset,
    now_nanoseconds,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
)

# Edit-distance baselines (each optional so a missing wheel skips just its row).
try:
    from rapidfuzz.distance import Levenshtein as rapidfuzz_levenshtein

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import Levenshtein as python_levenshtein

    PYTHON_LEVENSHTEIN_AVAILABLE = True
except ImportError:
    PYTHON_LEVENSHTEIN_AVAILABLE = False

try:
    import jellyfish

    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False

try:
    import editdistance

    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False

try:
    from nltk.metrics.distance import edit_distance as nltk_edit_distance

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import edlib

    EDLIB_AVAILABLE = True
except ImportError:
    EDLIB_AVAILABLE = False

try:
    import polyleven

    POLYLEVEN_AVAILABLE = True
except ImportError:
    POLYLEVEN_AVAILABLE = False

# For Needleman-Wunsch / Smith-Waterman alignment baseline.
try:
    from Bio import Align

    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# For RAPIDS cuDF GPU-accelerated edit distance.
try:
    import cudf

    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False

# Default per-core pair budget for similarity benchmarks (pairs processed per core).
# 256 is the measured GPU saturation knee for short-word edit distance; auto_batch_size scales it
# by each variant's core count, and the cross-product side is round(sqrt(budget)).
DEFAULT_BATCH_PER_CORE = 256


def crossproduct_side(budget: int, num_tokens: int) -> int:
    """Square cross-product side for a per-device pair `budget` and `num_tokens` available tokens.

    A ``side x side`` cross-product holds about `budget` pairs, clamped so the query slice
    ``[0, side)`` and candidate slice ``[side, 2*side)`` are disjoint, i.e. ``2 * side <= num_tokens``.
    """
    target = max(1, round(budget**0.5))
    max_side = num_tokens // 2
    return max(1, min(target, max_side))


def crossproduct_metrics(
    query_lengths: np.ndarray,
    candidate_lengths: np.ndarray,
    query_byte_lengths: np.ndarray,
    candidate_byte_lengths: np.ndarray,
    side: int,
) -> tuple[int, int]:
    """True aggregate cells and bytes spanned by a ``side x side`` cross-product.

    Cells = ``sum(query_lengths[:side]) * sum(candidate_lengths[:side])`` — the real number of
    matrix cells the dense cross-product fills. Bytes = the UTF-8 bytes fed to the kernel,
    ``sum(query_byte_lengths[:side]) + sum(candidate_byte_lengths[:side])``. The length arrays are
    either codepoint counts (UTF-8 metric) or byte counts (binary metric).
    """
    sum_query = int(query_lengths[:side].sum())
    sum_candidate = int(candidate_lengths[:side].sum())
    total_cells = sum_query * sum_candidate
    total_bytes = int(query_byte_lengths[:side].sum()) + int(candidate_byte_lengths[:side].sum())
    return total_cells, total_bytes


def _crossproduct_supported(engine: Any, queries: Any, candidates: Any) -> bool:
    """Probe whether the installed binding exposes the queries x candidates cross-product call.

    A supporting binding accepts two disjoint, differently-sized collections and returns a 2-D
    matrix; a binding without the cross-product call raises on unequal lengths. We probe once with a
    tiny mismatched pair so an unsupported binding degrades to a clear SKIP instead of a crash.
    """
    try:
        probe = engine(queries, candidates)
    except Exception:
        return False
    return getattr(np.asarray(probe), "ndim", 0) == 2


def measure_crossproduct(
    name: str,
    compute: Callable[[], None],
    total_cells: int,
    total_bytes: int,
    warmup_seconds: float,
    time_limit_seconds: float,
) -> None:
    """Run `compute` (one full cross-product per call) for a wall-time budget and report CUPS.

    The matrix/output buffer lives inside `compute`'s closure and is reused across iterations, so no
    allocation happens in the measured loop. After an uncounted warm-up the kernel is cycled until
    the deadline; throughput is the TRUE aggregate cell count divided by elapsed time. Mirrors the
    Rust `measure_throughput` cross-product blocks.
    """
    # Uncounted warm-up so caches and CPU frequency settle.
    if warmup_seconds > 0:
        warmup_deadline = now_nanoseconds() + int(warmup_seconds * 1e9)
        while now_nanoseconds() < warmup_deadline:
            compute()

    deadline_nanoseconds = now_nanoseconds() + int(time_limit_seconds * 1e9)
    start_nanoseconds = now_nanoseconds()
    iterations = 0
    try:
        # At least one measured iteration, even with a zero time budget (smoke tests).
        while True:
            compute()
            iterations += 1
            if now_nanoseconds() >= deadline_nanoseconds:
                break
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return

    elapsed_seconds = (now_nanoseconds() - start_nanoseconds) / 1e9
    processed_cells = total_cells * iterations
    processed_bytes = total_bytes * iterations
    report_stats(name, "cups", elapsed_seconds, processed_cells, processed_bytes)


def measure_pairwise_baseline(
    name: str,
    scalar_function: Callable[[Any, Any], int],
    queries: Sequence,
    candidates: Sequence,
    side: int,
    query_lengths: np.ndarray,
    candidate_lengths: np.ndarray,
    query_byte_lengths: np.ndarray,
    candidate_byte_lengths: np.ndarray,
    warmup_seconds: float,
    time_limit_seconds: float,
) -> None:
    """Benchmark a one-pair-at-a-time baseline along the cross-product diagonal.

    Baselines (rapidfuzz, jellyfish, edlib, biopython, ...) have no cross-product API, so they are
    measured exactly as bench.rs measures them: one ``(query[index], candidate[index])`` pair per
    call, cycling the diagonal ``index in [0, side)``. Each call's true cells (length product) are
    accumulated, so the reported CUPS are directly comparable to the StringZilla matrix engines.
    """
    if side <= 0:
        print(f"{name}: No pairs to process")
        return

    if warmup_seconds > 0:
        warmup_deadline = now_nanoseconds() + int(warmup_seconds * 1e9)
        pair_index = 0
        while now_nanoseconds() < warmup_deadline:
            scalar_function(queries[pair_index], candidates[pair_index])
            pair_index = (pair_index + 1) % side

    deadline_nanoseconds = now_nanoseconds() + int(time_limit_seconds * 1e9)
    start_nanoseconds = now_nanoseconds()
    checksum = 0
    processed_cells = 0
    processed_bytes = 0
    pair_index = 0
    # Adaptive cadence: check the clock every `stride` pairs, doubling toward the cap so cheap
    # pairs amortize the timer syscall while a slow pair cannot overshoot the deadline much.
    stride = 1
    countdown = 1
    pacing_cap = 1024
    pacing_target_nanoseconds = 1_000_000
    last_check_nanoseconds = start_nanoseconds
    try:
        while True:
            checksum += int(scalar_function(queries[pair_index], candidates[pair_index]))
            processed_cells += int(query_lengths[pair_index]) * int(candidate_lengths[pair_index])
            processed_bytes += int(query_byte_lengths[pair_index]) + int(candidate_byte_lengths[pair_index])
            pair_index = (pair_index + 1) % side
            countdown -= 1
            if countdown:
                continue
            current_nanoseconds = now_nanoseconds()
            if current_nanoseconds >= deadline_nanoseconds:
                break
            if current_nanoseconds - last_check_nanoseconds < pacing_target_nanoseconds and stride < pacing_cap:
                stride = min(stride * 2, pacing_cap)
            last_check_nanoseconds = current_nanoseconds
            countdown = stride
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")
        return

    elapsed_seconds = (now_nanoseconds() - start_nanoseconds) / 1e9
    report_stats(name, "cups", elapsed_seconds, processed_cells, processed_bytes)
    print(f"  {name} checksum={checksum}", file=sys.stderr)


def unary_class_costs(match_cost: int, mismatch_cost: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the 32-class substitution table for classic unary scoring (mirrors bench.rs).

    Each byte folds into one of 32 classes via ``byte % 32``, keeping the table compact; the cost is
    `match_cost` on the diagonal and `mismatch_cost` off it. Throughput (CUPS) is invariant to the
    actual cost values, so this stays apples-to-apples with the Rust harness.
    """
    byte_to_class = np.array([byte % 32 for byte in range(256)], dtype=np.uint8)
    class_substitution_costs = np.full((32, 32), mismatch_cost, dtype=np.int8)
    np.fill_diagonal(class_substitution_costs, match_cost)
    return byte_to_class, class_substitution_costs


class DeviceVariant:
    """A named ``(label, scope, side)`` benchmark variant for one device configuration."""

    def __init__(self, label: str, scope: Any, side: int):
        self.label = label
        self.scope = scope
        self.side = side


def build_device_variants(num_tokens: int, batch_size_override: int | None) -> list[DeviceVariant]:
    """Single-core, all-cores, and (when present) single-GPU variants with per-variant sides.

    Each variant scales ``DEFAULT_BATCH_PER_CORE`` by its own core count: a CPU core is one core, a
    GPU streaming multiprocessor is one core. The cross-product side is ``round(sqrt(budget))``
    clamped to the available tokens. The GPU variant is included only when a GPU DeviceScope can be
    created. Mirrors the Rust side derivation.
    """
    cpu_cores = os.cpu_count() or 1
    variants: list[DeviceVariant] = []

    single_cpu_budget = auto_batch_size(1, base=batch_size_override, default_base=DEFAULT_BATCH_PER_CORE)
    variants.append(
        DeviceVariant(
            "<1cpu>",
            szs.DeviceScope(cpu_cores=1),
            crossproduct_side(single_cpu_budget, num_tokens),
        )
    )

    all_cpu_budget = auto_batch_size(cpu_cores, base=batch_size_override, default_base=DEFAULT_BATCH_PER_CORE)
    variants.append(
        DeviceVariant(
            f"<{cpu_cores}cpu>",
            szs.DeviceScope(cpu_cores=cpu_cores),
            crossproduct_side(all_cpu_budget, num_tokens),
        )
    )

    try:
        gpu_scope = szs.DeviceScope(gpu_device=0)
    except Exception:
        gpu_scope = None
    if gpu_scope is not None:
        gpu_cores = gpu_multiprocessor_count(0) or 64
        gpu_budget = auto_batch_size(gpu_cores, base=batch_size_override, default_base=DEFAULT_BATCH_PER_CORE)
        variants.append(DeviceVariant("<1gpu>", gpu_scope, crossproduct_side(gpu_budget, num_tokens)))

    return variants


def benchmark_stringzillas_distances(
    tokens: Sequence,
    device_variants: list[DeviceVariant],
    category: str,
    engine_name: str,
    engine_class: Any,
    result_dtype: Any,
    byte_lengths: np.ndarray,
    metric_lengths: np.ndarray,
    is_utf8: bool,
    warmup_seconds: float,
    time_limit_seconds: float,
    filter_pattern: re.Pattern | None,
) -> None:
    """Cross-product benchmark for a StringZilla edit-distance engine (Levenshtein / UTF-8).

    For each device variant the disjoint query slice ``[0, side)`` and candidate slice
    ``[side, 2*side)`` are wrapped as ``sz.Strs`` once, a 2-D output matrix is preallocated, and the
    engine is invoked as ``engine(queries, candidates, device, out=matrix)`` each iteration so the
    matrix is reused. The GPU variant is skipped for the UTF-8 engine (no GPU UTF-8 kernel), as in
    bench.rs.
    """
    for variant in device_variants:
        full_name = f"{engine_name}{variant.label}"
        if not should_run(f"{category}/{full_name}", filter_pattern):
            continue

        side = variant.side
        queries = sz.Strs(list(tokens[0:side]))
        candidates = sz.Strs(list(tokens[side : 2 * side]))

        try:
            engine = engine_class(capabilities=variant.scope)
        except Exception as creation_error:
            print(f"{full_name}: SKIPPED ({creation_error})")
            continue

        if not _crossproduct_supported(engine, queries, candidates):
            print(f"{full_name}: SKIPPED (installed stringzillas lacks the queries x candidates cross-product API)")
            continue

        total_cells, total_bytes = crossproduct_metrics(
            metric_lengths,
            metric_lengths[side:],
            byte_lengths,
            byte_lengths[side:],
            side,
        )
        matrix = np.zeros((side, side), dtype=result_dtype)

        def compute(engine=engine, queries=queries, candidates=candidates, scope=variant.scope, matrix=matrix):
            engine(queries, candidates, scope, out=matrix)

        # Mirror bench.rs: attempt the kernel once; a backend that declines (e.g. no working GPU
        # path in the installed wheel for these inputs) surfaces as an exception we SKIP on rather
        # than aborting the whole suite.
        try:
            compute()
        except Exception as compute_error:
            print(f"{full_name}: SKIPPED ({compute_error})")
            continue

        measure_crossproduct(full_name, compute, total_cells, total_bytes, warmup_seconds, time_limit_seconds)


def benchmark_stringzillas_scores(
    tokens: Sequence,
    device_variants: list[DeviceVariant],
    category: str,
    engine_name: str,
    engine_class: Any,
    byte_to_class: np.ndarray,
    class_substitution_costs: np.ndarray,
    gap_open: int,
    gap_extend: int,
    byte_lengths: np.ndarray,
    warmup_seconds: float,
    time_limit_seconds: float,
    filter_pattern: re.Pattern | None,
) -> None:
    """Cross-product benchmark for a StringZilla scoring engine (Needleman-Wunsch / Smith-Waterman).

    Same structure as `benchmark_stringzillas_distances`, but the engine is built from the 32-class
    unary cost model plus affine gap penalties, and the output matrix is int64 (signed scores). The
    throughput denominator uses byte lengths (the binary cells), matching bench.rs.
    """
    for variant in device_variants:
        full_name = f"{engine_name}{variant.label}"
        if not should_run(f"{category}/{full_name}", filter_pattern):
            continue

        side = variant.side
        queries = sz.Strs(list(tokens[0:side]))
        candidates = sz.Strs(list(tokens[side : 2 * side]))

        try:
            engine = engine_class(
                byte_to_class,
                class_substitution_costs,
                open=gap_open,
                extend=gap_extend,
                capabilities=variant.scope,
            )
        except Exception as creation_error:
            print(f"{full_name}: SKIPPED ({creation_error})")
            continue

        if not _crossproduct_supported(engine, queries, candidates):
            print(f"{full_name}: SKIPPED (installed stringzillas lacks the queries x candidates cross-product API)")
            continue

        total_cells, total_bytes = crossproduct_metrics(
            byte_lengths,
            byte_lengths[side:],
            byte_lengths,
            byte_lengths[side:],
            side,
        )
        matrix = np.zeros((side, side), dtype=np.int64)

        def compute(engine=engine, queries=queries, candidates=candidates, scope=variant.scope, matrix=matrix):
            engine(queries, candidates, scope, out=matrix)

        # Mirror bench.rs: attempt the kernel once; a backend that declines (e.g. no working GPU
        # path in the installed wheel for these inputs) surfaces as an exception we SKIP on rather
        # than aborting the whole suite.
        try:
            compute()
        except Exception as compute_error:
            print(f"{full_name}: SKIPPED ({compute_error})")
            continue

        measure_crossproduct(full_name, compute, total_cells, total_bytes, warmup_seconds, time_limit_seconds)


def benchmark_edit_distance_baselines(
    tokens: Sequence,
    baseline_side: int,
    codepoint_lengths: np.ndarray,
    byte_lengths: np.ndarray,
    warmup_seconds: float,
    time_limit_seconds: float,
    filter_pattern: re.Pattern | None,
    batch_size_override: int | None,
) -> None:
    """Third-party edit-distance baselines along the single-CPU cross-product diagonal."""

    queries = list(tokens[0:baseline_side])
    candidates = list(tokens[baseline_side : 2 * baseline_side])
    query_codepoints = codepoint_lengths[:baseline_side]
    candidate_codepoints = codepoint_lengths[baseline_side : 2 * baseline_side]
    query_bytes = byte_lengths[:baseline_side]
    candidate_bytes = byte_lengths[baseline_side : 2 * baseline_side]

    def run(name: str, scalar_function: Callable[[Any, Any], int], length_metric: tuple[np.ndarray, np.ndarray]):
        if not should_run(f"levenshtein/{name}", filter_pattern):
            return
        measure_pairwise_baseline(
            name,
            scalar_function,
            queries,
            candidates,
            baseline_side,
            length_metric[0],
            length_metric[1],
            query_bytes,
            candidate_bytes,
            warmup_seconds,
            time_limit_seconds,
        )

    codepoint_metric = (query_codepoints, candidate_codepoints)
    byte_metric = (query_bytes, candidate_bytes)

    if RAPIDFUZZ_AVAILABLE:
        run("rapidfuzz.Levenshtein.distance", rapidfuzz_levenshtein.distance, codepoint_metric)
    if PYTHON_LEVENSHTEIN_AVAILABLE:
        run("Levenshtein.distance", python_levenshtein.distance, codepoint_metric)
    if JELLYFISH_AVAILABLE:
        run("jellyfish.levenshtein_distance", jellyfish.levenshtein_distance, codepoint_metric)
    if EDITDISTANCE_AVAILABLE:
        run("editdistance.eval", editdistance.eval, codepoint_metric)
    if NLTK_AVAILABLE:
        run("nltk.edit_distance", nltk_edit_distance, codepoint_metric)
    if EDLIB_AVAILABLE:

        def edlib_distance(first_string: str, second_string: str) -> int:
            return edlib.align(first_string, second_string, mode="NW", task="distance")["editDistance"]

        run("edlib.align", edlib_distance, byte_metric)
    if POLYLEVEN_AVAILABLE:
        run("polyleven.levenshtein", polyleven.levenshtein, byte_metric)

    # cuDF batched GPU edit distance: it scores a whole batch per call, but exposes no cross-product,
    # so it is benchmarked over the diagonal pairs as a batched array kernel.
    if CUDF_AVAILABLE:
        gpu_cores = gpu_multiprocessor_count(0) or 64
        gpu_batch_size = auto_batch_size(gpu_cores, base=batch_size_override, default_base=DEFAULT_BATCH_PER_CORE)
        name = f"cudf.edit_distance<1gpu,batch={gpu_batch_size}>"
        if should_run(f"levenshtein/{name}", filter_pattern):
            _benchmark_cudf_edit_distance(
                name,
                queries,
                candidates,
                query_codepoints,
                candidate_codepoints,
                query_bytes,
                candidate_bytes,
                warmup_seconds,
                time_limit_seconds,
            )


def _benchmark_cudf_edit_distance(
    name: str,
    queries: Sequence,
    candidates: Sequence,
    query_codepoints: np.ndarray,
    candidate_codepoints: np.ndarray,
    query_bytes: np.ndarray,
    candidate_bytes: np.ndarray,
    warmup_seconds: float,
    time_limit_seconds: float,
) -> None:
    """cuDF GPU edit-distance baseline over the diagonal pairs (one batched call per iteration)."""
    query_series = cudf.Series(queries)
    candidate_series = cudf.Series(candidates)
    # Diagonal-pair cells = sum over pairs of (q_i * c_i); cudf scores element-wise pairs, not a matrix.
    diagonal_cells = int((query_codepoints * candidate_codepoints).sum())
    diagonal_bytes = int(query_bytes.sum() + candidate_bytes.sum())

    def compute():
        results = query_series.str.edit_distance(candidate_series)
        return int(results.to_arrow().to_numpy().sum())

    if warmup_seconds > 0:
        warmup_deadline = now_nanoseconds() + int(warmup_seconds * 1e9)
        while now_nanoseconds() < warmup_deadline:
            compute()

    deadline_nanoseconds = now_nanoseconds() + int(time_limit_seconds * 1e9)
    start_nanoseconds = now_nanoseconds()
    iterations = 0
    checksum = 0
    while True:
        checksum += compute()
        iterations += 1
        if now_nanoseconds() >= deadline_nanoseconds:
            break
    elapsed_seconds = (now_nanoseconds() - start_nanoseconds) / 1e9
    report_stats(name, "cups", elapsed_seconds, diagonal_cells * iterations, diagonal_bytes * iterations)
    print(f"  {name} checksum={checksum}", file=sys.stderr)


def benchmark_biopython_baseline(
    tokens: Sequence,
    baseline_side: int,
    byte_lengths: np.ndarray,
    gap_open: int,
    gap_extend: int,
    category: str,
    mode: str,
    warmup_seconds: float,
    time_limit_seconds: float,
    filter_pattern: re.Pattern | None,
) -> None:
    """BioPython PairwiseAligner baseline (global or local) over the cross-product diagonal.

    Uses the same unary match=+2 / mismatch=-1 scoring as the StringZilla score engines so the CUPS
    are comparable. `mode` selects global (Needleman-Wunsch) or local (Smith-Waterman) alignment.
    """
    if not BIOPYTHON_AVAILABLE:
        return
    name = f"biopython.PairwiseAligner.{mode}"
    if not should_run(f"{category}/{name}", filter_pattern):
        return

    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    queries = list(tokens[0:baseline_side])
    candidates = list(tokens[baseline_side : 2 * baseline_side])
    query_bytes = byte_lengths[:baseline_side]
    candidate_bytes = byte_lengths[baseline_side : 2 * baseline_side]

    measure_pairwise_baseline(
        name,
        aligner.score,
        queries,
        candidates,
        baseline_side,
        query_bytes,
        candidate_bytes,
        query_bytes,
        candidate_bytes,
        warmup_seconds,
        time_limit_seconds,
    )


def perform_uniform_benchmarks(
    tokens: Sequence,
    device_variants: list[DeviceVariant],
    codepoint_lengths: np.ndarray,
    byte_lengths: np.ndarray,
    warmup_seconds: float,
    time_limit_seconds: float,
    filter_pattern: re.Pattern | None,
    batch_size_override: int | None,
) -> None:
    """Uniform-cost group: classic Levenshtein (match=0, mismatch=1, open=1, extend=1)."""
    baseline_side = device_variants[0].side

    benchmark_edit_distance_baselines(
        tokens,
        baseline_side,
        codepoint_lengths,
        byte_lengths,
        warmup_seconds,
        time_limit_seconds,
        filter_pattern,
        batch_size_override,
    )

    benchmark_stringzillas_distances(
        tokens,
        device_variants,
        "uniform",
        "stringzillas.LevenshteinDistances",
        szs.LevenshteinDistances,
        np.uint64,
        byte_lengths,
        byte_lengths,  # binary metric: cells = byte_length product
        is_utf8=False,
        warmup_seconds=warmup_seconds,
        time_limit_seconds=time_limit_seconds,
        filter_pattern=filter_pattern,
    )

    benchmark_stringzillas_distances(
        tokens,
        device_variants,
        "uniform",
        "stringzillas.LevenshteinDistancesUTF8",
        szs.LevenshteinDistancesUTF8,
        np.uint64,
        byte_lengths,
        codepoint_lengths,  # UTF-8 metric: cells = codepoint_length product
        is_utf8=True,
        warmup_seconds=warmup_seconds,
        time_limit_seconds=time_limit_seconds,
        filter_pattern=filter_pattern,
    )


def perform_score_benchmarks(
    tokens: Sequence,
    device_variants: list[DeviceVariant],
    byte_lengths: np.ndarray,
    group_name: str,
    gap_open: int,
    gap_extend: int,
    warmup_seconds: float,
    time_limit_seconds: float,
    filter_pattern: re.Pattern | None,
) -> None:
    """NW/SW score group (linear or affine) with unary match=+2 / mismatch=-1 scoring."""
    byte_to_class, class_substitution_costs = unary_class_costs(2, -1)

    benchmark_biopython_baseline(
        tokens,
        device_variants[0].side,
        byte_lengths,
        gap_open,
        gap_extend,
        "needleman-wunsch",
        "global",
        warmup_seconds,
        time_limit_seconds,
        filter_pattern,
    )
    benchmark_stringzillas_scores(
        tokens,
        device_variants,
        "needleman-wunsch",
        "stringzillas.NeedlemanWunschScores",
        szs.NeedlemanWunschScores,
        byte_to_class,
        class_substitution_costs,
        gap_open,
        gap_extend,
        byte_lengths,
        warmup_seconds,
        time_limit_seconds,
        filter_pattern,
    )

    benchmark_biopython_baseline(
        tokens,
        device_variants[0].side,
        byte_lengths,
        gap_open,
        gap_extend,
        "smith-waterman",
        "local",
        warmup_seconds,
        time_limit_seconds,
        filter_pattern,
    )
    benchmark_stringzillas_scores(
        tokens,
        device_variants,
        "smith-waterman",
        "stringzillas.SmithWatermanScores",
        szs.SmithWatermanScores,
        byte_to_class,
        class_substitution_costs,
        gap_open,
        gap_extend,
        byte_lengths,
        warmup_seconds,
        time_limit_seconds,
        filter_pattern,
    )


_main_epilog = """
Examples:

  # Benchmark with a file
  %(prog)s --dataset data/leipzig-1m/leipzig1M_en.txt

  # Benchmark protein sequences with BioPython scoring baselines
  %(prog)s --bio --dataset data/acgt/acgt_1k.txt

  # Custom time limit
  %(prog)s --dataset data/leipzig-1m/leipzig1M_en.txt --time-limit 30
"""


def main() -> int:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla similarity operations (all-pairs cross-product)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)
    parser.add_argument(
        "--bio",
        action="store_true",
        help="Include BioPython + NW/SW alignment score benchmarks (linear + affine gap costs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Pairs processed per core (overrides STRINGWARS_BATCH_PER_CORE, default: 256)",
    )

    args = parser.parse_args()

    if not args.dataset and not os.environ.get("STRINGWARS_DATASET"):
        parser.error("Dataset is required (use --dataset or STRINGWARS_DATASET env var)")

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as compile_error:
            parser.error(f"Invalid regex for --filter: {compile_error}")

    seed = get_env_parsed("STRINGWARS_SEED", 42)
    random.seed(seed)

    # Wall-time budget per variant: --time-limit / STRINGWARS_TIME for measurement, STRINGWARS_WARMUP
    # for the uncounted warm-up. The CLI flag wins, then the env var, then the default.
    time_limit_seconds = get_env_parsed("STRINGWARS_TIME", args.time_limit, parser=float)
    warmup_seconds = get_env_parsed("STRINGWARS_WARMUP", 0.0, parser=float)

    # Load and tokenize the dataset; STRINGWARS_MAX_TOKENS caps the token count.
    dataset = load_dataset(args.dataset, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(dataset, tokens_mode=resolve_tokens(args.tokens, "words"))
    max_tokens = get_env_parsed("STRINGWARS_MAX_TOKENS", None, parser=int)
    if max_tokens is not None and max_tokens > 0:
        tokens = tokens[:max_tokens]

    if len(tokens) < 2:
        parser.error("Dataset must contain at least two tokens for the cross-product")

    # Shuffle so the disjoint query/candidate halves are not biased by file order.
    random.shuffle(tokens)
    num_tokens = len(tokens)

    # Per-token length metrics, computed once and sliced per variant.
    codepoint_lengths = np.fromiter((len(token) for token in tokens), dtype=np.int64, count=num_tokens)
    byte_lengths = np.fromiter((len(token.encode("utf-8")) for token in tokens), dtype=np.int64, count=num_tokens)

    device_variants = build_device_variants(num_tokens, args.batch_size)

    print("Benchmark configuration (all-pairs cross-product):")
    for variant in device_variants:
        side = variant.side
        print(f"- {variant.label}: {side}x{side} cross-product ({side * side:,} pairs)")
    print(f"- Tokens available: {num_tokens:,}")
    print(f"- Time budget per variant: {time_limit_seconds}s (warmup {warmup_seconds}s), seed {seed}")
    print()

    print("# uniform")
    perform_uniform_benchmarks(
        tokens,
        device_variants,
        codepoint_lengths,
        byte_lengths,
        warmup_seconds,
        time_limit_seconds,
        filter_pattern,
        args.batch_size,
    )

    if args.bio:
        print("\n# linear")
        perform_score_benchmarks(
            tokens,
            device_variants,
            byte_lengths,
            "linear",
            gap_open=-2,
            gap_extend=-2,
            warmup_seconds=warmup_seconds,
            time_limit_seconds=time_limit_seconds,
            filter_pattern=filter_pattern,
        )

        print("\n# affine")
        perform_score_benchmarks(
            tokens,
            device_variants,
            byte_lengths,
            "affine",
            gap_open=-5,
            gap_extend=-1,
            warmup_seconds=warmup_seconds,
            time_limit_seconds=time_limit_seconds,
            filter_pattern=filter_pattern,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
