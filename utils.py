"""
Shared utilities for StringWars Python benchmarking scripts.

Common functions for dataset loading, tokenization, timing, and argument parsing
used across bench_find.py, bench_hash.py, and other benchmarking scripts.
"""

import os
import re
import time
from collections.abc import Callable

# region: Environment Variable Helpers
# Standardized functions for fetching environment variables consistently.
# Use these instead of raw os.environ.get() calls throughout the codebase.


def get_env(name: str) -> str | None:
    """Get an optional environment variable, returning None if not set."""
    return os.environ.get(name)


def get_env_or_default(name: str, default: str) -> str:
    """Get an environment variable with a default value."""
    return os.environ.get(name, default)


def get_env_parsed[T](name: str, default: T, parser: Callable[[str], T] = int) -> T:
    """
    Get an environment variable parsed to a type, with a default value.
    Returns the default if the variable is not set or cannot be parsed.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return parser(value)
    except (ValueError, TypeError):
        return default


def get_env_parsed_opt[T](name: str, parser: Callable[[str], T] = int) -> T | None:
    """
    Get an optional environment variable parsed to a type.
    Returns None if the variable is not set or cannot be parsed.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return parser(value)
    except (ValueError, TypeError):
        return None


def get_env_bool(name: str) -> bool:
    """
    Get a boolean environment variable.
    Accepts "1", "true", or "yes" (case-insensitive) as true values.
    Returns False if not set or set to any other value.
    """
    value = os.environ.get(name, "").lower()
    return value in ("1", "true", "yes")


# endregion: Environment Variable Helpers


def now_nanoseconds() -> int:
    """Get current time in nanoseconds for benchmarking."""
    return time.monotonic_ns()


# region: Benchmark loop profile

# Reporting stride: the smallest number of operations to run between progress logging
# and clock reads, so the cost of terminal rendering and timer syscalls is amortized
# away. This is independent of batch size. Benchmarks come in two shapes:
#
# - One item at a time: a single element per call, logged every LOGGING_STEP calls.
# - Batched: many items per call, where one batch already spans many items.
#
# Batched benchmarks use clamped_subranges to slice the input and hand each slice to a
# kernel that consumes a whole batch. One-at-a-time benchmarks use paced_items, which
# walks a single iterator and checks the clock from inside the loop, so it never builds
# a slice and works on any iterable.
LOGGING_STEP = 1024


def clamped_subranges(count: int, stride: int = LOGGING_STEP):
    """Yield (low, high) index windows covering [0, count) in stride-sized steps.

    Used by the batched benchmarks: slice the input with the returned indices, hand
    each slice to a kernel, then read the clock and repaint progress once per window.
    Pass stride explicitly to use the batch size as the window.
    """
    for low in range(0, count, stride):
        yield low, min(low + stride, count)


# Target wall-time between deadline clock reads in paced_items. The clock read costs tens of
# nanoseconds, so reading it once per ~1 ms of work keeps its overhead negligible (~0.01%).
PACING_TARGET_BETWEEN_CHECKS_NS = 1_000_000


def paced_items(items, deadline_nanoseconds: int, step: int = LOGGING_STEP, progress=None):
    """Yield items from a single iterator, checking the deadline from inside the loop.

    The companion to clamped_subranges for one-at-a-time benchmarks. It walks one
    iterator, repainting progress every step items, and stops once the deadline passes.
    Works on any iterable, such as a list, a zip of two lists, or an itertools.cycle,
    and never builds a slice.

    The cadence is *adaptive* rather than a fixed step. A fixed step would either read the
    clock too often on fine-grained items (e.g. hashing words) or overshoot the time limit
    on few-but-huge items (e.g. one whole file in `file` tokenization mode, where there is a
    single item). So `stride` — the number of items between checkpoints — starts at 1 and
    doubles toward `step` whenever a stride ran faster than `PACING_TARGET_BETWEEN_CHECKS_NS`:
    cheap items climb to the fully-amortized `step`, while a slow item leaves `stride` at 1
    and is checked every iteration, so the deadline cannot overshoot by more than one item.
    Progress repaint and the deadline check share the checkpoint, so the hot path is a single
    countdown — the same per-item cost as a non-adaptive loop.
    """
    stride = 1  # items between checkpoints; doubles toward `step` for cheap items
    countdown = 1
    last_check_nanoseconds = now_nanoseconds()
    for item in items:
        yield item
        countdown -= 1
        if countdown:
            continue
        current_nanoseconds = now_nanoseconds()
        if progress is not None:
            progress.update(stride)
        if current_nanoseconds >= deadline_nanoseconds:
            return
        if current_nanoseconds - last_check_nanoseconds < PACING_TARGET_BETWEEN_CHECKS_NS and stride < step:
            stride = min(stride * 2, step)
        last_check_nanoseconds = current_nanoseconds
        countdown = stride


def reduce_in_windows(
    function,
    *columns,
    deadline_nanoseconds: int,
    step: int = LOGGING_STEP,
    combine=sum,
    progress=None,
):
    """Apply function across the zipped columns one window at a time and reduce each window.

    The shared home for the trick of pushing a per-item loop into C: every window is
    handled by combine(map(...)), so the function calls and the reduction run in C with
    no Python bytecode per item, and the deadline and optional progress are checked once
    per window. This runs about 1.5 to 1.9 times faster than an explicit Python loop on
    real kernels, but only when the body is "call a function on each item and reduce the
    results" with no per-item side effect.

    The window size is *adaptive*, exactly like paced_items: it starts at 1 and doubles toward
    `step` whenever a window ran faster than PACING_TARGET_BETWEEN_CHECKS_NS. A window of
    few-but-huge items (e.g. one full-haystack find per call) stays small, so the deadline is
    re-checked promptly and cannot overshoot by more than one window's worth of a single item;
    cheap items climb to the fully-amortized `step` and keep the C-map speedup.

    Pass several equal-length sequences to vary more than one argument, for example two
    string columns. Pin a fixed argument with a leading functools.partial or a constant
    column such as [fixed] * n. Returns (reduced_total, processed_count).
    """
    count = min((len(column) for column in columns), default=0)
    total = 0
    low = 0
    window = 1  # items per checkpoint; doubles toward `step` for cheap items
    last_check_nanoseconds = now_nanoseconds()
    while low < count:
        if now_nanoseconds() >= deadline_nanoseconds:
            break
        high = min(low + window, count)
        total += combine(map(function, *(column[low:high] for column in columns)))
        if progress is not None:
            progress.update(high - low)
        current_nanoseconds = now_nanoseconds()
        if current_nanoseconds - last_check_nanoseconds < PACING_TARGET_BETWEEN_CHECKS_NS and window < step:
            window = min(window * 2, step)
        last_check_nanoseconds = current_nanoseconds
        low = high
    return total, low


def items_per_core(base: int | None = None, default_base: int = 128) -> int:
    """Items processed per core — one CPU core, or on the GPU one streaming multiprocessor (SM).
    "Core" here means an SM, not an individual warp or CUDA core. Precedence: an explicit `base`
    (e.g. a --batch-size CLI flag) wins, then `STRINGWARS_BATCH_PER_CORE`, then the bench's own
    `default_base` (the saturating batch differs by kernel — short-string similarity peaks at a
    different per-core batch than document fingerprinting).
    """
    if base is not None:
        return max(1, base)
    return max(1, get_env_parsed("STRINGWARS_BATCH_PER_CORE", default_base))


def auto_batch_size(cores: int, base: int | None = None, default_base: int = 128) -> int:
    """Batch size for a backend with `cores` parallel cores, scaling the bench's `default_base` by
    the hardware. A CPU core counts as one core and a GPU streaming multiprocessor counts as one
    core, so the batch scales automatically: a 1-core scope gets `items_per_core`, an N-core scope
    `N * items_per_core`, and a GPU `streaming_multiprocessors * items_per_core`. Mirrors the Rust
    `auto_batch_size`.
    """
    return max(1, items_per_core(base, default_base) * max(1, cores))


def gpu_multiprocessor_count(device_index: int = 0) -> int | None:
    """Number of streaming multiprocessors on the given CUDA device, queried straight from the
    CUDA runtime via ctypes (no cupy/torch needed). Each SM is counted as one core for batch
    sizing (an SM, not a warp or an individual CUDA core). Returns None when CUDA is unavailable
    or the query fails, so callers fall back to a default core count. Attribute id 16 is
    `cudaDevAttrMultiProcessorCount`.
    """
    import ctypes
    import ctypes.util

    candidates = ["libcudart.so", "libcudart.so.12", ctypes.util.find_library("cudart")]
    multiprocessor_count_attribute = 16
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        count = ctypes.c_int(0)
        status = library.cudaDeviceGetAttribute(
            ctypes.byref(count), ctypes.c_int(multiprocessor_count_attribute), ctypes.c_int(device_index)
        )
        if status == 0 and count.value > 0:
            return count.value
    return None


# endregion: Benchmark loop profile


# region: Reporting
# One canonical, column-aligned result line per variant, identical in layout to the Rust harness
# (utils.rs). The Rust-only cycles-per-byte and IPC columns are absent here (Python cannot read
# hardware perf counters), but every other column — primary rate, bytes/s, latency percentiles —
# uses the same units, the same SI thresholds, and the same 2-decimal precision, so the two suites
# stay diff-compatible.

# Width of the left-aligned variant-name column, matching the Rust reporter.
REPORT_NAME_WIDTH = 42


def scale_si(value: float) -> tuple[float, str]:
    """Scale a value to a metric prefix (G/M/k), returning (scaled_value, prefix)."""
    if value >= 1e9:
        return value / 1e9, "G"
    if value >= 1e6:
        return value / 1e6, "M"
    if value >= 1e3:
        return value / 1e3, "k"
    return value, ""


def format_byte_rate(bytes_per_second: float) -> str:
    """Render a bytes-per-second rate as `<value> <prefix>B/s` (decimal SI, 2 decimals)."""
    value, prefix = scale_si(bytes_per_second)
    return f"{value:.2f} {prefix}B/s"


def format_si_rate(rate: float, unit: str, space_before_unit: bool) -> str:
    """Render an SI rate as `<value> <prefix><unit>` (e.g. `1.24 GCUPS`), with a space between the
    prefix and a word unit when `space_before_unit` is set (e.g. `1.24 G hashes/s`)."""
    value, prefix = scale_si(rate)
    if not prefix:
        return f"{value:.2f} {unit}"
    return f"{value:.2f} {prefix} {unit}" if space_before_unit else f"{value:.2f} {prefix}{unit}"


def format_seconds(value_seconds: float) -> str:
    """Render a duration with an appropriate sub-second unit, matching the Rust reporter."""
    if value_seconds < 1e-6:
        return f"{value_seconds * 1e9:.2f} ns"
    if value_seconds < 1e-3:
        return f"{value_seconds * 1e6:.2f} µs"
    if value_seconds < 1.0:
        return f"{value_seconds * 1e3:.2f} ms"
    return f"{value_seconds:.2f} s"


def report_stats(
    name: str,
    report: str,
    elapsed_seconds: float,
    elements: int,
    total_bytes: int,
    latencies_seconds: list[float] | None = None,
) -> None:
    """Print the single canonical result line for one variant.

    `report` selects the primary unit: "bytes", "cups", "hashes", "bits", or "comparisons". bytes/s is
    always shown as the secondary metric (unless the primary already is bytes/s). Columns are
    joined by " | " in a fixed order, and columns that cannot be computed are omitted, never
    reformatted, so the layout matches the Rust harness line-for-line.
    """
    seconds = max(elapsed_seconds, 1e-12)
    columns: list[str] = []

    elements_per_second = elements / seconds
    bytes_per_second = total_bytes / seconds
    if report == "bytes":
        columns.append(format_byte_rate(bytes_per_second))
    elif report == "cups":
        columns.append(format_si_rate(elements_per_second, "CUPS", False))
    elif report == "hashes":
        columns.append(format_si_rate(elements_per_second, "hashes/s", True))
    elif report == "bits":
        columns.append(format_si_rate(elements_per_second, "bits/s", True))
    elif report == "comparisons":
        columns.append(format_si_rate(elements_per_second, "cmp/s", True))
    else:
        raise ValueError(f"Unknown report unit: {report!r}")

    if report != "bytes" and total_bytes > 0:
        columns.append(format_byte_rate(bytes_per_second))

    if latencies_seconds:
        ordered = sorted(latencies_seconds)

        def quantile(fraction: float) -> float:
            rank = round(fraction * (len(ordered) - 1))
            return ordered[min(rank, len(ordered) - 1)]

        columns.append(f"p50 {format_seconds(quantile(0.5))} p99 {format_seconds(quantile(0.99))}")

    print(f"{name:<{REPORT_NAME_WIDTH}} {' | '.join(columns)}")


# endregion: Reporting


def parse_size(size_str: str) -> int:
    """
    Parse a size string like '128mb', '1gb', '500kb' into bytes.

    Supports: b, kb, mb, gb (case insensitive)
    Returns size in bytes.
    """
    if not size_str:
        raise ValueError("Size string cannot be empty")

    # Match number followed by optional unit
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb)?$", size_str.lower().strip())
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Use formats like '128mb', '1gb', '500kb'")

    number, unit = match.groups()
    number = float(number)

    # Convert to bytes
    multipliers = {
        None: 1,
        "b": 1,
        "kb": 1024,
        "mb": 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
    }

    return int(number * multipliers[unit])


def load_dataset(
    dataset_path: str | None = None,
    as_bytes: bool = False,
    size_limit: str | None = None,
) -> str | bytes:
    """
    Load dataset from file path or environment variable.

    Args:
        dataset_path: Path to dataset file (uses STRINGWARS_DATASET env var if None)
        as_bytes: If True, return bytes; if False, return str
        size_limit: Maximum size to read (e.g., "128mb", "1gb"). If None, read entire file.

    Returns:
        Dataset contents as str or bytes based on as_bytes parameter
    """
    if dataset_path is None:
        dataset_path = get_env("STRINGWARS_DATASET")
        if dataset_path is None:
            raise ValueError("No dataset path provided and STRINGWARS_DATASET not set")

    # Parse size limit if provided
    max_bytes = None
    if size_limit:
        max_bytes = parse_size(size_limit)

    if as_bytes:
        with open(dataset_path, "rb") as f:
            if max_bytes is not None:
                return f.read(max_bytes)
            else:
                return f.read()
    else:
        with open(dataset_path, encoding="utf-8", errors="ignore") as f:
            if max_bytes is not None:
                return f.read(max_bytes)
            else:
                return f.read()


def tokenize_dataset(
    haystack: str | bytes,
    tokens_mode: str | None = None,
    unique: bool | None = None,
) -> list[str] | list[bytes]:
    """
    Tokenize haystack based on mode from argument or environment variable.

    Args:
        haystack: Input data to tokenize (str or bytes)
        tokens_mode: Tokenization mode ('lines', 'words', 'file') or None to use env var
        unique: If True, deduplicate tokens (preserving order). If None, uses STRINGWARS_UNIQUE.

    Returns:
        List of tokens in the same type as input (List[str] or List[bytes])
    """
    if tokens_mode is None:
        tokens_mode = get_env_or_default("STRINGWARS_TOKENS", "lines")

    is_bytes = isinstance(haystack, bytes)

    if tokens_mode == "lines":
        # Split on LF only
        tokens = haystack.split(b"\n" if is_bytes else "\n")
    elif tokens_mode == "words":
        # Use default split() which handles all whitespace
        tokens = haystack.split()
    elif tokens_mode == "file":
        tokens = [haystack]
    else:
        raise ValueError(f"Unknown tokens mode: {tokens_mode}. Use 'lines', 'words', or 'file'.")

    if unique is None:
        unique = get_env_bool("STRINGWARS_UNIQUE")

    # Deduplicate, preserving the order of first appearance.
    if tokens_mode != "file" and unique:
        tokens = list(dict.fromkeys(tokens))

    return tokens


def resolve_tokens(cli_value: str | None, default: str) -> str:
    """Resolve the token granularity with precedence: an explicit --tokens flag wins, then the
    STRINGWARS_TOKENS environment variable, then the bench's own default. Each bench passes the
    granularity its kernel measures (e.g. "words" for similarity, "lines" for hashing,
    normalization and fingerprinting), matching the Rust `load_dataset_with_default_mode`.
    """
    if cli_value is not None:
        return cli_value
    return get_env_or_default("STRINGWARS_TOKENS", default)


def add_common_args(parser):
    """Add common dataset and tokenization arguments to an ArgumentParser."""
    parser.add_argument(
        "--dataset",
        help="Path to input dataset file (overrides STRINGWARS_DATASET env var)",
    )
    parser.add_argument(
        "--tokens",
        choices=["lines", "words", "file"],
        help="Tokenization mode (overrides STRINGWARS_TOKENS env var)",
    )
    parser.add_argument(
        "-k",
        "--filter",
        metavar="REGEX",
        default=get_env("STRINGWARS_FILTER"),
        help="Regex to select which benchmarks to run (or set STRINGWARS_FILTER env var)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Time limit per benchmark function in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--dataset-limit",
        type=str,
        default="128mb",
        help="Maximum dataset size (default: 128mb). Supports formats like '1gb', '500mb', '10kb'",
    )


def should_run(name: str, pattern: re.Pattern | None) -> bool:
    """Check if benchmark should run based on filter pattern."""
    if pattern is None:
        return True
    assert hasattr(pattern, "search"), "Pattern must be a compiled regex"
    return bool(pattern.search(name))
