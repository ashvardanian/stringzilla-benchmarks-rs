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
"""String-similarity benchmarks in Python, reported in CUPS. Mirrors `similarities/bench.rs`."""

import argparse
import random
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# String similarity libraries
import stringzilla as sz
import stringzillas as szs

from utils import (
    MeasureSpec,
    add_common_args,
    auto_batch_size,
    finish,
    get_env_parsed,
    gpu_multiprocessor_count,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    pass_over,
    resolve_core_count,
    resolve_dataset,
    set_filter,
    should_run,
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


# Deliberately unequal lengths: that is the shape a binding without the cross-product call rejects.
_PROBE_QUERIES = sz.Strs(["ab", "cde", "f"])
_PROBE_CANDIDATES = sz.Strs(["gh", "ij"])


def _crossproduct_supported(engine: Any) -> bool:
    """Probe whether the installed binding exposes the queries x candidates cross-product call.

    A supporting binding accepts two disjoint, differently-sized collections and returns a 2-D
    matrix; a binding without the cross-product call raises on unequal lengths. The probe uses a
    tiny mismatched pair so an unsupported binding degrades to a clear SKIP instead of a crash,
    and so the check costs nothing next to the row it guards.
    """
    try:
        probe = engine(_PROBE_QUERIES, _PROBE_CANDIDATES)
    except Exception:
        return False
    return getattr(np.asarray(probe), "ndim", 0) == 2


def measure_crossproduct(
    name: str,
    compute: Callable[[], None],
    total_cells: int,
    total_bytes: int,
) -> None:
    """One pass is one full cross-product; the output buffer lives in `compute`'s closure."""
    work = MeasureSpec(report="cups", elements=total_cells, total_bytes=total_bytes)
    try:
        measure(name, work, compute)
    except KeyboardInterrupt:
        print(f"\n{name}: SKIPPED (interrupted by user)")


def measure_pairwise_baseline(
    name: str,
    scalar_function: Callable[[Any, Any], Any],
    queries: list,
    candidates: list,
    total_cells: int,
    total_bytes: int,
) -> None:
    """One pass scores every query/candidate pair, so the pair mixture cancels exactly."""
    work = MeasureSpec(report="cups", elements=total_cells, total_bytes=total_bytes)
    measure(name, work, pass_over(scalar_function, queries, candidates))


def unary_class_costs(match_cost: int, mismatch_cost: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the 32-class substitution table for classic unary scoring (mirrors bench.rs).

    Each byte folds into one of 32 classes via ``byte % 32``, keeping the table compact; the cost is
    `match_cost` on the diagonal and `mismatch_cost` off it. Throughput (CUPS) is invariant to the
    actual cost values, so this stays apples-to-apples with the Rust harness.
    """
    byte_to_class = np.arange(256, dtype=np.uint8) % 32
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
    cpu_cores = resolve_core_count()
    variants: list[DeviceVariant] = []

    single_cpu_budget = auto_batch_size(1, base=batch_size_override, default_base=DEFAULT_BATCH_PER_CORE)
    variants.append(
        DeviceVariant(
            "<1cpu>",
            szs.DeviceScope(cpu_cores=1),
            crossproduct_side(single_cpu_budget, num_tokens),
        ),
    )

    all_cpu_budget = auto_batch_size(cpu_cores, base=batch_size_override, default_base=DEFAULT_BATCH_PER_CORE)
    variants.append(
        DeviceVariant(
            f"<{cpu_cores}cpu>",
            szs.DeviceScope(cpu_cores=cpu_cores),
            crossproduct_side(all_cpu_budget, num_tokens),
        ),
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
) -> None:
    """Cross-product benchmark for a StringZilla edit-distance engine (Levenshtein / UTF-8).

    For each device variant the disjoint query slice ``[0, side)`` and candidate slice
    ``[side, 2*side)`` are wrapped as ``sz.Strs`` once, a 2-D output matrix is preallocated, and the
    engine is invoked as ``engine(queries, candidates, device, out=matrix)`` each iteration so the
    matrix is reused. The GPU variant is skipped for the UTF-8 engine (no GPU UTF-8 kernel), as in
    bench.rs.
    """
    for variant in device_variants:
        full_name = f"{category}/{engine_name}{variant.label}"
        if not should_run(full_name):
            continue

        side = variant.side
        queries = sz.Strs(tokens[0:side])
        candidates = sz.Strs(tokens[side : 2 * side])

        try:
            engine = engine_class(capabilities=variant.scope)
        except Exception as creation_error:
            print(f"{full_name}: SKIPPED ({creation_error})")
            continue

        if not _crossproduct_supported(engine):
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

        measure_crossproduct(full_name, compute, total_cells, total_bytes)


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
) -> None:
    """Cross-product benchmark for a StringZilla scoring engine (Needleman-Wunsch / Smith-Waterman).

    Same structure as `benchmark_stringzillas_distances`, but the engine is built from the 32-class
    unary cost model plus affine gap penalties, and the output matrix is int64 (signed scores). The
    throughput denominator uses byte lengths (the binary cells), matching bench.rs.
    """
    for variant in device_variants:
        full_name = f"{category}/{engine_name}{variant.label}"
        if not should_run(full_name):
            continue

        side = variant.side
        queries = sz.Strs(tokens[0:side])
        candidates = sz.Strs(tokens[side : 2 * side])

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

        if not _crossproduct_supported(engine):
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

        measure_crossproduct(full_name, compute, total_cells, total_bytes)


def benchmark_edit_distance_baselines(
    tokens: Sequence,
    baseline_side: int,
    codepoint_lengths: np.ndarray,
    byte_lengths: np.ndarray,
    batch_size_override: int | None,
) -> None:
    """Third-party edit-distance baselines along the single-CPU cross-product diagonal."""

    queries = tokens[0:baseline_side]
    candidates = tokens[baseline_side : 2 * baseline_side]
    query_codepoints = codepoint_lengths[:baseline_side]
    candidate_codepoints = codepoint_lengths[baseline_side : 2 * baseline_side]
    query_bytes = byte_lengths[:baseline_side]
    candidate_bytes = byte_lengths[baseline_side : 2 * baseline_side]

    def run(name: str, scalar_function: Callable[[Any, Any], int], length_metric: tuple[np.ndarray, np.ndarray]):
        name = f"levenshtein/{name}"
        if not should_run(name):
            return
        # These baselines score the diagonal pairs, not a cross-product, so the work is
        # summed pairwise: cells are len(query_i) * len(candidate_i) under whichever
        # length metric the library uses, bytes are what is actually fed to the kernel.
        query_lengths, candidate_lengths = length_metric
        total_cells = int((query_lengths * candidate_lengths).sum())
        total_bytes = int(query_bytes.sum()) + int(candidate_bytes.sum())
        measure_pairwise_baseline(
            name,
            scalar_function,
            queries,
            candidates,
            total_cells,
            total_bytes,
        )

    codepoint_metric = (query_codepoints, candidate_codepoints)
    byte_metric = (query_bytes, candidate_bytes)

    if not RAPIDFUZZ_AVAILABLE:
        note_unavailable("levenshtein/rapidfuzz.Levenshtein.distance", "rapidfuzz not installed")
    else:
        run("rapidfuzz.Levenshtein.distance", rapidfuzz_levenshtein.distance, codepoint_metric)
    if not PYTHON_LEVENSHTEIN_AVAILABLE:
        note_unavailable("levenshtein/Levenshtein.distance", "python-Levenshtein not installed")
    else:
        run("Levenshtein.distance", python_levenshtein.distance, codepoint_metric)
    if not JELLYFISH_AVAILABLE:
        note_unavailable("levenshtein/jellyfish.levenshtein_distance", "jellyfish not installed")
    else:
        run("jellyfish.levenshtein_distance", jellyfish.levenshtein_distance, codepoint_metric)
    if not EDITDISTANCE_AVAILABLE:
        note_unavailable("levenshtein/editdistance.eval", "editdistance not installed")
    else:
        run("editdistance.eval", editdistance.eval, codepoint_metric)
    if not NLTK_AVAILABLE:
        note_unavailable("levenshtein/nltk.edit_distance", "nltk not installed")
    else:
        run("nltk.edit_distance", nltk_edit_distance, codepoint_metric)
    if not EDLIB_AVAILABLE:
        note_unavailable("levenshtein/edlib.align", "edlib not installed")
    else:

        def edlib_distance(first_string: str, second_string: str) -> int:
            return edlib.align(first_string, second_string, mode="NW", task="distance")["editDistance"]

        run("edlib.align", edlib_distance, byte_metric)
    if not POLYLEVEN_AVAILABLE:
        note_unavailable("levenshtein/polyleven.levenshtein", "polyleven not installed")
    else:
        run("polyleven.levenshtein", polyleven.levenshtein, byte_metric)

    # cuDF batched GPU edit distance: it scores a whole batch per call, but exposes no cross-product,
    # so it is benchmarked over the diagonal pairs as a batched array kernel.
    if not CUDF_AVAILABLE:
        note_unavailable("levenshtein/cudf.edit_distance<1gpu>", "cudf not installed")
    else:
        gpu_cores = gpu_multiprocessor_count(0) or 64
        gpu_batch_size = auto_batch_size(gpu_cores, base=batch_size_override, default_base=DEFAULT_BATCH_PER_CORE)
        name = f"levenshtein/cudf.edit_distance<1gpu,batch={gpu_batch_size}>"
        if should_run(name):
            _benchmark_cudf_edit_distance(
                name,
                queries,
                candidates,
                query_codepoints,
                candidate_codepoints,
                query_bytes,
                candidate_bytes,
            )


def _benchmark_cudf_edit_distance(
    name: str,
    queries: Sequence,
    candidates: Sequence,
    query_codepoints: np.ndarray,
    candidate_codepoints: np.ndarray,
    query_bytes: np.ndarray,
    candidate_bytes: np.ndarray,
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

    work = MeasureSpec(report="cups", elements=diagonal_cells, total_bytes=diagonal_bytes)
    measure(name, work, compute)


def benchmark_biopython_baseline(
    tokens: Sequence,
    baseline_side: int,
    byte_lengths: np.ndarray,
    gap_open: int,
    gap_extend: int,
    category: str,
    mode: str,
) -> None:
    """BioPython PairwiseAligner baseline (global or local) over the cross-product diagonal.

    Uses the same unary match=+2 / mismatch=-1 scoring as the StringZilla score engines so the CUPS
    are comparable. `mode` selects global (Needleman-Wunsch) or local (Smith-Waterman) alignment.
    """
    if not BIOPYTHON_AVAILABLE:
        note_unavailable(f"{category}/biopython.PairwiseAligner.{mode}", "biopython not installed")
        return
    name = f"{category}/biopython.PairwiseAligner.{mode}"
    if not should_run(name):
        return

    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    queries = tokens[0:baseline_side]
    candidates = tokens[baseline_side : 2 * baseline_side]
    query_bytes = byte_lengths[:baseline_side]
    candidate_bytes = byte_lengths[baseline_side : 2 * baseline_side]

    total_cells = int((query_bytes * candidate_bytes).sum())
    total_bytes = int(query_bytes.sum()) + int(candidate_bytes.sum())
    measure_pairwise_baseline(name, aligner.score, queries, candidates, total_cells, total_bytes)


def perform_uniform_benchmarks(
    tokens: Sequence,
    device_variants: list[DeviceVariant],
    codepoint_lengths: np.ndarray,
    byte_lengths: np.ndarray,
    batch_size_override: int | None,
) -> None:
    """Uniform-cost group: classic Levenshtein (match=0, mismatch=1, open=1, extend=1)."""
    baseline_side = device_variants[0].side

    benchmark_edit_distance_baselines(
        tokens,
        baseline_side,
        codepoint_lengths,
        byte_lengths,
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
    )


def perform_score_benchmarks(
    tokens: Sequence,
    device_variants: list[DeviceVariant],
    byte_lengths: np.ndarray,
    group_name: str,
    gap_open: int,
    gap_extend: int,
) -> None:
    """NW/SW score group (linear or affine) with unary match=+2 / mismatch=-1 scoring."""
    byte_to_class, class_substitution_costs = unary_class_costs(2, -1)

    benchmark_biopython_baseline(
        tokens,
        device_variants[0].side,
        byte_lengths,
        gap_open,
        gap_extend,
        group_name,
        "global",
    )
    benchmark_stringzillas_scores(
        tokens,
        device_variants,
        group_name,
        "stringzillas.NeedlemanWunschScores",
        szs.NeedlemanWunschScores,
        byte_to_class,
        class_substitution_costs,
        gap_open,
        gap_extend,
        byte_lengths,
    )

    benchmark_biopython_baseline(
        tokens,
        device_variants[0].side,
        byte_lengths,
        gap_open,
        gap_extend,
        group_name,
        "local",
    )
    benchmark_stringzillas_scores(
        tokens,
        device_variants,
        group_name,
        "stringzillas.SmithWatermanScores",
        szs.SmithWatermanScores,
        byte_to_class,
        class_substitution_costs,
        gap_open,
        gap_extend,
        byte_lengths,
    )


_main_epilog = """
Examples:

  %(prog)s --dataset leipzig1M.txt

  %(prog)s --bio --dataset acgt_1k.txt

  # Custom time limit
  %(prog)s --dataset leipzig1M.txt --time-limit 30
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
        "--no-bio",
        dest="bio",
        action="store_false",
        help="Skip the BioPython + NW/SW alignment score groups (linear and affine gap costs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Pairs processed per core (overrides STRINGWARS_BATCH_PER_CORE, default: 256)",
    )

    args = parser.parse_args()

    set_filter(args.filter)

    seed = get_env_parsed("STRINGWARS_SEED", 42)
    random.seed(seed)

    dataset = resolve_dataset("similarities", as_bytes=False, dataset_path=args.dataset)
    tokens = list(dataset.tokens)
    log_dataset(dataset)
    log_timing_overhead()

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
    print()

    print("# uniform")
    perform_uniform_benchmarks(
        tokens,
        device_variants,
        codepoint_lengths,
        byte_lengths,
        args.batch_size,
    )

    # On by default, because `bench.rs` runs these groups unconditionally: gating them
    # behind an opt-in meant a plain `bench.py` produced no Python column for any of the
    # four alignment tables, while Rust filled all four.
    if args.bio:
        print("\n# linear")
        perform_score_benchmarks(
            tokens,
            device_variants,
            byte_lengths,
            "linear",
            gap_open=-2,
            gap_extend=-2,
        )

        print("\n# affine")
        perform_score_benchmarks(
            tokens,
            device_variants,
            byte_lengths,
            "affine",
            gap_open=-5,
            gap_extend=-1,
        )

    finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
