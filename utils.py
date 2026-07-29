"""Shared harness for the StringWars Python suites. Mirrors `utils.rs`."""

import functools
import math
import os
import re
import time
import tomllib
import zlib
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

# Closed vocabularies, mirroring the Rust enums. `Literal` rather than `Enum`: it
# costs nothing at runtime and keeps call sites spelling `report="bytes"`.
ReportUnit = Literal["bytes", "cups", "hashes", "bits", "comparisons"]
TokensMode = Literal["lines", "words", "file"]
RowStatus = Literal["ok", "refused", "too_slow", "skipped", "filtered"]

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


def get_env_bool(name: str) -> bool:
    """
    Get a boolean environment variable.
    Accepts "1", "true", or "yes" (case-insensitive) as true values.
    Returns False if not set or set to any other value.
    """
    value = os.environ.get(name, "").lower()
    return value in ("1", "true", "yes")


# endregion: Environment Variable Helpers


_FILTER: re.Pattern[str] | None = None


def set_filter(expression: str | None) -> None:
    """Compile `STRINGWARS_FILTER` once, at startup, for the whole process."""
    global _FILTER
    _FILTER = re.compile(expression) if expression else None


# region: Benchmark loop profile

# Default window for suites whose kernel consumes a batch rather than one item.
# This is a batching knob, not a pacing one: the clock is read twice per sample and
# never inside a window, so the window size cannot affect timing overhead.


def resolve_core_count(available: int | None = None) -> int:
    """Logical cores for a multi-core scope, overridable with `STRINGWARS_CPU_CORES`."""
    available = available or os.cpu_count() or 1
    override = get_env_parsed("STRINGWARS_CPU_CORES", 0)
    return max(1, override if override > 0 else available)


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
            ctypes.byref(count),
            ctypes.c_int(multiprocessor_count_attribute),
            ctypes.c_int(device_index),
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


# endregion: Reporting


def parse_size(size_str: str) -> int:
    """
    Parse a size string like '128MB', '1GB', '500kB' into bytes.

    Decimal SI, matching how throughput is reported: 1 kB = 1000 B. The previous
    implementation was 1024-based while `scale_si` renders 1000-based, so a
    '128mb' budget read 134,217,728 bytes and printed as "134.22 MB".
    """
    if not size_str:
        raise ValueError("Size string cannot be empty")

    match = re.match(r"^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb)?$", size_str.lower().strip())
    if not match:
        raise ValueError(f"Invalid size format: {size_str}. Use formats like '128MB', '1GB', '500kB'")

    number, unit = match.groups()
    multipliers = {None: 1, "b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3}
    return int(float(number) * multipliers[unit])


# region: Manifest


@functools.cache
def manifest() -> dict:
    """`stringwars.toml` — the single source of defaults shared with `utils.rs`.

    Parsed once: `limits()` runs per row, and re-reading the file there put an open and a
    TOML parse between every pair of measurements.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stringwars.toml")
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@dataclass(frozen=True)
class Limits:
    """Global measurement limits, mirroring the Rust `Limits` struct field for field."""

    bytes: int
    min_sample_ms: float
    min_samples: int
    min_seconds: float
    max_seconds: float
    target_spread: float
    warmup_max_seconds: float


@dataclass(frozen=True)
class SuiteSettings:
    """One suite's dataset, token mode and working-set budget."""

    dataset: str | None
    tokens: TokensMode
    bytes: int


def limits() -> Limits:
    """Global measurement limits, with `STRINGWARS_*` overrides applied."""
    values = dict(manifest()["limits"])
    # Every limit is overridable; the previous loop silently omitted min_sample_ms
    # and min_seconds, so those two knobs did nothing.
    for key, parser in (
        ("bytes", str),
        ("min_sample_ms", float),
        ("min_samples", int),
        ("min_seconds", float),
        ("max_seconds", float),
        ("target_spread", float),
        ("warmup_max_seconds", float),
    ):
        raw = get_env("STRINGWARS_" + key.upper())
        if raw is not None:
            values[key] = parser(raw)
    budget = values["bytes"]
    return Limits(
        # Parsed here, once, rather than by each caller.
        bytes=parse_size(budget) if isinstance(budget, str) else int(budget),
        min_sample_ms=float(values["min_sample_ms"]),
        min_samples=int(values["min_samples"]),
        min_seconds=float(values["min_seconds"]),
        max_seconds=float(values["max_seconds"]),
        target_spread=float(values["target_spread"]),
        warmup_max_seconds=float(values["warmup_max_seconds"]),
    )


def suite_settings(suite: str) -> SuiteSettings:
    """One suite's dataset, token mode, and working-set budget in bytes."""
    entry = manifest().get("suite", {}).get(suite, {})
    budget = get_env("STRINGWARS_BYTES") or entry.get("bytes")
    return SuiteSettings(
        dataset=get_env("STRINGWARS_DATASET") or entry.get("dataset"),
        tokens=get_env("STRINGWARS_TOKENS") or entry.get("tokens", "lines"),
        bytes=(parse_size(budget) if isinstance(budget, str) else int(budget)) if budget else limits().bytes,
    )


# endregion: Manifest


_ASCII_WHITESPACE = b" \n\t\r\v\f"


def tokenize_dataset(
    haystack: str | bytes,
    tokens_mode: str | None = None,
    unique: bool | None = None,
) -> list[str] | list[bytes]:
    """
    Split a buffer into tokens. Normative definition, shared with `utils.rs`:

      lines  split on \\n, drop empty
      words  split on ASCII whitespace {space \\n \\t \\r \\v \\f}, drop empty
      file   one token, the whole buffer

    Both harnesses must produce identical token counts and bytes; the pre-commit
    conformance test asserts it.
    """
    if tokens_mode is None:
        tokens_mode = get_env_or_default("STRINGWARS_TOKENS", "lines")

    is_bytes = isinstance(haystack, bytes)

    if tokens_mode == "lines":
        tokens = haystack.split(b"\n" if is_bytes else "\n")
        tokens = [token for token in tokens if token]
    elif tokens_mode == "words":
        separators = _ASCII_WHITESPACE if is_bytes else _ASCII_WHITESPACE.decode("ascii")
        pattern = b"[" + re.escape(_ASCII_WHITESPACE) + b"]+" if is_bytes else "[" + re.escape(separators) + "]+"
        tokens = [token for token in re.split(pattern, haystack) if token]
    elif tokens_mode == "file":
        tokens = [haystack]
    else:
        raise ValueError(f"Unknown tokens mode: {tokens_mode}. Use 'lines', 'words', or 'file'.")

    if unique is None:
        unique = get_env_bool("STRINGWARS_UNIQUE")

    # Deduplicate first, then cap — the reverse order yields "up to N tokens, then
    # deduplicated", which is fewer than N unique tokens.
    if tokens_mode != "file" and unique:
        tokens = list(dict.fromkeys(tokens))

    return tokens


@dataclass(frozen=True)
class Dataset:
    """A pinned working set. `token_bytes` is the denominator of every bytes/s figure."""

    tokens: list
    token_bytes: int
    token_count: int
    fingerprint: int
    mode: str
    path: str


def _char_boundary_floor(raw: bytes, limit: int) -> int:
    """Largest `end <= limit` where `raw[:end]` is still valid UTF-8.

    Mirrors `utils.rs::char_boundary_floor`. The old form walked back over continuation
    bytes but guarded on `end < len(raw)`, which is never true in `file` mode where the
    read is cut at exactly the budget — so a buffer ending mid-sequence went through whole,
    and only `errors="ignore"` downstream hid it here while Rust panicked.
    """
    end = min(limit, len(raw))
    try:
        raw[:end].decode("utf-8")
    except UnicodeDecodeError as error:
        return error.start
    return end


def _token_bytes(token) -> int:
    return len(token) if isinstance(token, (bytes, bytearray)) else len(token.encode("utf-8"))


def _read_within_budget(handle, budget: int, mode: str) -> bytes:
    """Read only what the budget can consume. Mirrors utils.rs::read_within_budget."""
    if mode == "file":
        return handle.read(budget)
    if get_env_bool("STRINGWARS_UNIQUE"):
        # Dedup precedes the cap, so unique-token counts genuinely need the whole corpus.
        return handle.read()

    separators = b"\n" if mode == "lines" else _ASCII_WHITESPACE

    # The budget counts token bytes but a read returns raw bytes, and separators are
    # not free: `budget` bytes of xlsum.csv yields only ~1.80 MB of words per 2 MB.
    # `translate(None, ...)` drops the separators in one C pass; counting only the
    # newly-read segment keeps the loop linear.
    buffer = bytearray(handle.read(budget))
    token_bytes = len(bytes(buffer).translate(None, separators))
    while len(buffer) >= budget and token_bytes < budget:
        more = handle.read(budget - token_bytes + (1 << 16))
        if not more:
            break
        token_bytes += len(more.translate(None, separators))
        buffer += more

    # Complete the straddling token so the cap can judge it on its true length.
    tail = handle.read(1 << 20)
    cut = min(
        (position for position in (tail.find(bytes([sep])) for sep in separators) if position >= 0),
        default=len(tail),
    )
    buffer += tail[:cut]
    return bytes(buffer)


def resolve_dataset(suite: str, as_bytes: bool = True, dataset_path: str | None = None) -> Dataset:
    """
    Resolve the working set for one suite.

    Normative algorithm, implemented identically in `utils.rs`: read the file,
    tokenize, then accumulate whole tokens and stop *before* the first token that
    would push the running total of token bytes past the budget. Never a torn
    token. The budget counts token bytes rather than file bytes, so the two
    languages share an exact denominator regardless of separator overhead.
    """
    settings = suite_settings(suite)
    dataset_path = dataset_path or settings.dataset
    if dataset_path is None:
        raise ValueError(f"No dataset for suite {suite!r}: set STRINGWARS_DATASET or add one to stringwars.toml")

    budget = settings.bytes
    # Always read binary. The old text path used `f.read(n)` on a TextIOWrapper,
    # which counts codepoints, so the byte budget silently varied with the script.
    with open(dataset_path, "rb") as handle:
        raw = _read_within_budget(handle, budget, settings.tokens)

    if settings.tokens == "file":
        # The one mode where the budget must truncate: a single token cannot be dropped
        # without emptying the working set, so the "never tear a token" rule would leave
        # `file` mode unbounded — Rust capped and Python did not, silently comparing a
        # 128 MB working set against a 1 MB one. Back off to a UTF-8 boundary so both
        # harnesses cap at the same byte.
        raw = raw[: _char_boundary_floor(raw, budget)]

    haystack = raw if as_bytes else raw.decode("utf-8", errors="ignore")

    kept, used = [], 0
    for token in tokenize_dataset(haystack, settings.tokens):
        size = _token_bytes(token)
        if used + size > budget and kept:
            break
        kept.append(token)
        used += size

    if not kept:
        raise ValueError(f"No tokens from {dataset_path} in mode {settings.tokens}")

    dataset = Dataset(
        tokens=kept,
        token_bytes=used,
        token_count=len(kept),
        fingerprint=fingerprint_tokens(kept),
        mode=settings.tokens,
        path=dataset_path,
    )

    global _RUN
    _RUN = {
        "lang": "python",
        "suite": suite,
        "dataset": dataset_path,
        "mode": dataset.mode,
        "tokens": dataset.token_count,
        "token_bytes": dataset.token_bytes,
        "crc": f"0x{dataset.fingerprint:08x}",
    }
    return dataset


# What every record is stamped with, captured once when the working set resolves.
# Module-level because `measure` is handed only the row it is timing, and threading
# run context through every call site would be noise.
_RUN: dict | None = None

# The fields that must agree across languages for a suite to be conformant.
_IDENTITY_FIELDS = ("dataset", "mode", "tokens", "token_bytes", "crc")

# The closed set of report units, mirroring Rust's `ReportAs` variants. A `Literal`
# rather than an `Enum`: it costs nothing at runtime and keeps the 20 call sites
# spelling `report="bytes"` instead of `ReportAs.BYTES`.

# name for the record, display unit, whether a space precedes a word unit
_REPORT_UNITS: dict[str, tuple[str, str, bool]] = {
    "bytes": ("bytes/s", "B/s", False),
    "cups": ("CUPS", "CUPS", False),
    "hashes": ("hashes/s", "hashes/s", True),
    "bits": ("bits/s", "bits/s", True),
    "comparisons": ("cmp/s", "cmp/s", True),
}


# Every row the run reached, in order, with what became of it. A row that vanishes
# silently is indistinguishable from a row that was never written: `similarities`
# lost four whole tables behind a `--bio` gate and the output looked complete.
_ROSTER: list[tuple[str, str]] = []


def note_unavailable(name: str, reason: str) -> None:
    """
    Record a contender that could not run at all — a missing optional dependency, a
    gated backend. Prints a line so the absence is in the output rather than implied
    by a gap in the table.
    """
    print(f"{name:<{REPORT_NAME_WIDTH}} SKIPPED: {reason}")
    _ROSTER.append((name, "skipped"))


def finish() -> None:
    """Print the roster tally; exit non-zero if a row that was asked to run produced nothing."""
    import sys

    tally = Counter(status for _, status in _ROSTER)
    print(
        f"\nRoster: {tally['ok']} measured, {tally['refused']} refused, "
        f"{tally['too_slow']} too slow, {tally['skipped']} skipped, {tally['filtered']} filtered",
    )
    refused = [name for name, status in _ROSTER if status == "refused"]
    if refused:
        for name in refused:
            print(f"  refused: {name}")
        sys.exit(1)


def _record_outcome(outcome: "Outcome", spec: "MeasureSpec", bytes_per_second: float) -> None:
    """
    Append one NDJSON record per row when `STRINGWARS_RESULTS_DIR` is set.

    Records exist so a published number can be traced back to the working set that
    produced it. They never touch a README, whose tables stay hand-written.
    """
    directory = os.environ.get("STRINGWARS_RESULTS_DIR")
    if not directory or _RUN is None or outcome.status == "filtered":
        return
    import json

    os.makedirs(directory, exist_ok=True)
    record = dict(_RUN)
    record.update(
        row=outcome.name,
        unit=_REPORT_UNITS[spec.report][0],
        rate=outcome.median_rate,
        bytes_per_second=bytes_per_second,
        spread=outcome.spread,
        samples=outcome.samples,
        passes_per_sample=outcome.passes_per_sample,
        concurrency=spec.concurrency,
        status=outcome.status,
    )
    with open(os.path.join(directory, f"{_RUN['suite']}.ndjson"), "a") as handle:
        handle.write(json.dumps(record) + "\n")


@dataclass(frozen=True)
class MeasureSpec:
    """
    What one pass over the pinned working set costs, declared up front.

    Declared rather than accumulated: counting work per call costs a Python-level
    increment per item, which at ~50-80 ns swamps a short kernel.
    """

    report: ReportUnit
    elements: int
    total_bytes: int
    concurrency: int = 1


@dataclass
class Outcome:
    name: str
    median_rate: float
    spread: float
    samples: int
    passes_per_sample: int
    status: str


def clock_overhead_nanoseconds() -> float:
    """
    Cost of one `time.monotonic_ns()`. The harness reads the clock exactly twice
    per sample, so this over `min_sample_ms` is the whole timing overhead.
    """
    rounds = 10_000
    start = time.monotonic_ns()
    for _ in range(rounds):
        time.monotonic_ns()
    return (time.monotonic_ns() - start) / rounds


def log_timing_overhead() -> None:
    """Proof obligation behind "the harness does not perturb the measurement"."""
    per_clock = clock_overhead_nanoseconds()
    floor_nanoseconds = limits().min_sample_ms * 1e6
    print(
        f"Timing: {per_clock:.0f} ns/clock, {limits().min_sample_ms:.1f} ms sample floor "
        f"-> harness overhead <= {200.0 * per_clock / floor_nanoseconds:.4f}%",
    )


def _quantile(ordered: list[float], fraction: float) -> float:
    """
    Select a sample by rank, using a rule written out rather than delegated.

    Python's `round` breaks ties to even and Rust's `f64::round` breaks them away
    from zero, so the two harnesses picked different samples: at n = 10 — the modal
    terminating count, since `min_samples` is 10 and the convergence gate fires as
    soon as it is met — Python took the 5th-smallest and Rust the 6th. Every median
    rate and spread is derived from this, so the disagreement was systematic.
    """
    if not ordered:
        return 0.0
    rank = math.floor(fraction * (len(ordered) - 1) + 0.5)
    return ordered[min(rank, len(ordered) - 1)]


def _round_to_earned_digits(value: float, relative_halfwidth: float) -> float:
    """Round to the significant figures the measured dispersion justifies."""
    if value == 0.0 or not math.isfinite(value):
        return value
    digits = 1
    for candidate in range(1, 5):
        if 10.0 ** (-(candidate - 1)) >= relative_halfwidth:
            digits = candidate
    magnitude = math.floor(math.log10(abs(value)))
    scale = 10.0 ** (digits - 1 - magnitude)
    # Half-up, spelled out, for the same reason as `_quantile`: rates are positive,
    # so this is the tie rule both harnesses can state rather than inherit.
    return math.floor(value * scale + 0.5) / scale


def measure(
    name: str,
    spec: MeasureSpec,
    pass_fn: Callable[[], None],
    setup: Callable[[], None] | None = None,
) -> Outcome:
    """
    Times `pass_fn` over the pinned working set and reports one row.

    A sample is `passes_per_sample` complete traversals bracketed by exactly two
    clock reads; the count is chosen during warm-up so a sample lasts at least
    `min_sample_ms`. There is no per-call clock and therefore no adaptive stride,
    and the loop between the two reads contains no harness bookkeeping.

    `setup` runs before each sample and outside the clock, for kernels that consume
    their input. It forces one pass per sample, since a rebuild between passes would
    land inside the timed span.
    """

    def run_sample(passes: int) -> float:
        if setup is not None:
            setup()
        started = time.monotonic_ns()
        for _ in range(passes):
            pass_fn()
        return time.monotonic_ns() - started

    def too_slow(seconds: float) -> Outcome:
        """One pass rules out the three samples a dispersion estimate needs."""
        rate = spec.total_bytes / max(seconds, 1e-9)
        print(
            f"{name:<{REPORT_NAME_WIDTH}} TOO SLOW: ~{format_byte_rate(_round_to_earned_digits(rate, 1.0))}"
            f", one pass ~{seconds:.0f}s against a {resolved.max_seconds:.0f}s cap"
        )
        # An expected outcome, not a failure: its own bucket, so `finish` does not exit 1.
        _ROSTER.append((name, "too_slow"))
        return Outcome(name, 0.0, float("nan"), 1, 0, "too_slow")

    def refuse(status: str, samples: int) -> Outcome:
        # A refusal must be as visible as a result; printing nothing is how a row
        # goes missing for weeks without anyone noticing.
        if status != "filtered":
            print(f"{name:<{REPORT_NAME_WIDTH}} REFUSED: {status}")
        # `should_run` already noted a filtered row; only refusals are ours to record.
        if status != "filtered":
            _ROSTER.append((name, "refused"))
        return Outcome(name, 0.0, float("nan"), samples, 0, status)

    if not should_run(name):
        return refuse("filtered", 0)

    resolved = limits()
    floor_nanoseconds = resolved.min_sample_ms * 1e6

    passes = 1
    if setup is None:
        while True:
            took = run_sample(passes)
            if took >= floor_nanoseconds or passes >= 1 << 40:
                break
            passes = min(max(2, math.ceil(floor_nanoseconds / max(took, 1))) * passes, 1 << 40)
    else:
        # A consuming kernel cannot batch passes, but one sample still prices one. Skipping
        # it left `took` at zero, so the refusal below could never fire for `sequence` - the
        # one suite whose work is superlinear and therefore most able to outrun the cap.
        took = run_sample(passes)

    # Calibration is the first moment the cost of a pass is known. If one pass already
    # rules out the three samples a dispersion estimate needs, stop here rather than
    # after warm-up and a measured sample have each paid it again.
    pass_seconds = took / 1e9 / max(passes, 1)
    if pass_seconds * 3.0 > resolved.max_seconds:
        return too_slow(pass_seconds)

    warmup_deadline = time.monotonic_ns() + int(resolved.warmup_max_seconds * 1e9)
    recent: deque[float] = deque(maxlen=3)
    # A single pass longer than the deadline used to yield zero warm-up samples, so the
    # row entered measurement cold - what warm-up exists to prevent.
    while len(recent) < 3 or time.monotonic_ns() < warmup_deadline:
        recent.append(run_sample(passes) / 1e9)
        if len(recent) == 3 and (max(recent) - min(recent)) / max(recent) <= resolved.target_spread:
            break

    measure_start = time.monotonic_ns()
    cap_nanoseconds = resolved.max_seconds * 1e9
    seconds_per_sample: list[float] = []
    while True:
        seconds_per_sample.append(run_sample(passes) / 1e9)

        elapsed = time.monotonic_ns() - measure_start
        if len(seconds_per_sample) >= resolved.min_samples and elapsed / 1e9 >= resolved.min_seconds:
            ordered = sorted(seconds_per_sample)
            median = _quantile(ordered, 0.5)
            halfwidth = (_quantile(ordered, 0.9) - _quantile(ordered, 0.1)) / (2.0 * max(median, 1e-12))
            if halfwidth <= resolved.target_spread:
                break
        if elapsed >= cap_nanoseconds:
            break

    if len(seconds_per_sample) < 3:
        return refuse(f"{len(seconds_per_sample)} samples in {resolved.max_seconds:.0f}s cap (need 3)", 0)

    ordered = sorted(seconds_per_sample)
    median_seconds = _quantile(ordered, 0.5)
    spread = (_quantile(ordered, 0.9) - _quantile(ordered, 0.1)) / max(median_seconds, 1e-12)
    mean_seconds = sum(seconds_per_sample) / len(seconds_per_sample)
    gap = abs(mean_seconds - median_seconds) / max(median_seconds, 1e-12)

    if gap > 2.0 * resolved.target_spread:
        status = f"NON-STATIONARY {100.0 * gap:.0f}%"
    elif spread / 2.0 > resolved.target_spread:
        status = f"UNCONVERGED {100.0 * spread:.0f}%"
    else:
        status = "converged"

    elements_per_second = spec.elements * passes / median_seconds
    bytes_per_second = spec.total_bytes * passes / median_seconds
    primary = bytes_per_second if spec.report == "bytes" else elements_per_second

    halfwidth = spread / 2.0
    columns = [_format_primary(spec.report, _round_to_earned_digits(primary, halfwidth))]
    if spec.report != "bytes" and spec.total_bytes > 0:
        columns.append(format_byte_rate(_round_to_earned_digits(bytes_per_second, halfwidth)))
    columns.append(f"+-{100.0 * halfwidth:.1f}% n={len(seconds_per_sample)}")
    if status != "converged":
        columns.append(status)
    print(f"{name:<{REPORT_NAME_WIDTH}} {' | '.join(columns)}")

    outcome = Outcome(name, primary, spread, len(seconds_per_sample), passes, status)
    _record_outcome(outcome, spec, bytes_per_second)
    _ROSTER.append((name, "ok"))
    return outcome


def measure_with_setup(
    name: str,
    spec: MeasureSpec,
    setup: Callable[[], object],
    body: Callable[[object], object],
) -> Outcome:
    """
    Like `measure`, for kernels that consume their input — sorting, in-place edits.

    A thin wrapper over `measure` rather than a second implementation: the copy this
    replaced had no warm-up, no pass calibration, no `min_sample_ms` floor and no
    non-stationary test, printed a different refusal wording, and recorded
    `status="converged"` into the NDJSON even when the line it had just printed said
    `UNCONVERGED` — so the record contradicted the console.
    """
    state: dict = {}

    def prepare() -> None:
        state["input"] = setup()

    return measure(name, spec, lambda: body(state["input"]), setup=prepare)


def pass_over(function: Callable, *columns) -> Callable[[], None]:
    """
    Build a pass that drives `function` across whole columns inside C.

    The interpreter must not appear in the measured loop: a Python-level `for` body
    costs ~50-80 ns per item, which swamps a short kernel and gets attributed to it.
    `deque(..., maxlen=0)` drains the `map` at C speed and keeps no results.
    """

    def run() -> None:
        deque(map(function, *columns), maxlen=0)

    return run


def _format_primary(report: str, value: float) -> str:
    """Render the primary column. Unknown units raise rather than silently print `cmp/s`."""
    if report == "bytes":
        return format_byte_rate(value)
    if report not in _REPORT_UNITS:
        raise ValueError(f"Unknown report unit {report!r}; expected one of {sorted(_REPORT_UNITS)}")
    _, unit, spaced = _REPORT_UNITS[report]
    return format_si_rate(value, unit, spaced)


def log_dataset(dataset: "Dataset") -> None:
    """
    Print the working set's identity, in the same format as `utils.rs`.

    This line is what makes the two harnesses checkable against each other: same
    mode, count, bytes and CRC means they resolved the same working set. They
    silently diverged for months (Rust split words on ASCII bytes, Python's
    `str.split()` also split U+3000 in the CJK corpora) and nothing in the output
    would have shown it.
    """
    gigabytes = dataset.token_bytes / 1e9
    print(f"Dataset: {dataset.token_count:,} tokens, {dataset.token_bytes:,} bytes ({gigabytes:.2f} GB)")
    print(f"  Identity: mode {dataset.mode} crc 0x{dataset.fingerprint:08x}")


def fingerprint_tokens(tokens) -> int:
    """
    Identity of a resolved working set: CRC32 over the tokens joined by a NUL.

    Deterministic across processes (unlike `hash()`, which is seed-randomized) and
    trivially mirrored in Rust with `crc32fast`, which the tree already depends on.
    Digesting the tokens rather than the file means capping the working set does
    not require re-hashing gigabytes.
    """
    blob = b"\x00".join(token if isinstance(token, (bytes, bytearray)) else token.encode("utf-8") for token in tokens)
    return zlib.crc32(blob) & 0xFFFFFFFF


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


def should_run(name: str) -> bool:
    """Whether a row passes `STRINGWARS_FILTER`, recording the skip as it decides."""
    if _FILTER is None:
        return True
    matches = bool(_FILTER.search(name))
    if not matches:
        # Recorded here rather than in `measure` because several suites consult the
        # filter themselves and return early; those rows used to vanish from the
        # tally entirely instead of being counted as filtered.
        _ROSTER.append((name, "filtered"))
    return matches


def check_conformance(directory: str) -> int:
    """
    Verify the two harnesses resolved the same working set, from the records they wrote.

    Run both languages of a suite with `STRINGWARS_RESULTS_DIR` pointed here, then
    `python utils.py <dir>`. A mismatch means the numbers in that suite's table are
    not comparable, however plausible they look side by side — Rust and Python
    silently disagreed on tokenization for months and no output showed it.

    Records append, so a directory accumulates history. Only each language's most
    recent identity is compared: otherwise a since-fixed disagreement keeps failing.
    """
    import glob
    import json

    failures = 0
    for path in sorted(glob.glob(os.path.join(directory, "*.ndjson"))):
        suite = os.path.basename(path).removesuffix(".ndjson")
        identities = {}
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                identities[record["lang"]] = tuple(record[field] for field in _IDENTITY_FIELDS)
        distinct = set(identities.values())
        if len(identities) < 2:
            print(f"{suite:<16} SKIP  only {'/'.join(identities) or 'no'} records")
        elif len(distinct) == 1:
            dataset, mode, tokens, token_bytes, crc = distinct.pop()
            print(f"{suite:<16} OK    {mode} {tokens:,} tokens, {token_bytes:,} bytes, crc {crc}")
        else:
            failures += 1
            print(f"{suite:<16} FAIL  harnesses resolved different working sets")
            for lang, values in sorted(identities.items()):
                # Pair each value with its field name. Sorting the tuple instead raised
                # TypeError on int-against-str, crashing the one branch that reports a
                # mismatch.
                fields = " ".join(f"{f}={v}" for f, v in zip(_IDENTITY_FIELDS, values, strict=True))
                print(f"                  {lang:<8} {fields}")
    return failures


if __name__ == "__main__":
    import sys

    raise SystemExit(check_conformance(sys.argv[1] if len(sys.argv) > 1 else "results"))
