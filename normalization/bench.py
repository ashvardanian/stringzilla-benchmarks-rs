# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "regex",
#   "PyICU",
# ]
# ///
"""
Python case folding and normalization benchmarks.

Benchmarks case-insensitive operations and Unicode normalization:
- Case-insensitive comparison of adjacent token pairs
- Case-insensitive substring search
- Case folding transformation
- Unicode normalization (NFC / NFD / NFKC / NFKD)

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run normalization/bench.py --dataset README.md --tokens lines
  uv run normalization/bench.py --dataset data/xlsum/xlsum.csv --tokens words -k "casefold"

Timing via time.monotonic_ns; throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import itertools
import random
import re
import sys
import unicodedata
from collections.abc import Callable
from functools import partial
from importlib.metadata import version as pkg_version

import icu
import regex
import stringzilla as sz

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    paced_items,
    reduce_in_windows,
    report_stats,
    should_run,
    tokenize_dataset,
)


def log_system_info():
    """Log Python version and library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- regex: {pkg_version('regex')}")
    print(f"- PyICU: {pkg_version('PyICU')} (ICU {icu.ICU_VERSION})")
    print()


def make_pairs(tokens: list[str]) -> list[tuple[str, str]]:
    """Create adjacent token pairs for comparison benchmarks."""
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def bench_case_compare(
    name: str,
    pairs: list[tuple[str, str]],
    compare_function: Callable[[str, str], bool],
    time_limit_seconds: float = 10.0,
):
    """Benchmark case-insensitive comparison of string pairs."""
    if not pairs:
        print(f"{name}: no pairs to compare", file=sys.stderr)
        return

    # Encode once: byte length of every pair plus cumulative prefix sums, so the
    # hot loop never re-encodes a string just to count throughput bytes.
    cumulative_bytes = [0]
    for first_string, second_string in pairs:
        pair_bytes = len(first_string.encode("utf-8")) + len(second_string.encode("utf-8"))
        cumulative_bytes.append(cumulative_bytes[-1] + pair_bytes)
    bytes_per_pass = cumulative_bytes[-1]

    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    first_column = [first_string for first_string, _ in pairs]
    second_column = [second_string for _, second_string in pairs]

    compared_pairs = 0
    matches_found = 0

    # Cycle whole C-windowed passes over the dataset until the time limit; a pass that
    # returns short means the deadline was hit mid-pass.
    while now_nanoseconds() < deadline_nanoseconds:
        pass_matches, pass_count = reduce_in_windows(
            compare_function,
            first_column,
            second_column,
            deadline_nanoseconds=deadline_nanoseconds,
        )
        matches_found += pass_matches
        compared_pairs += pass_count
        if pass_count < len(pairs):
            break

    seconds = (now_nanoseconds() - start_time) / 1e9
    full_passes, remainder = divmod(compared_pairs, len(pairs))
    compared_bytes = full_passes * bytes_per_pass + cumulative_bytes[remainder]

    print(f"{name}: {matches_found:,} matches over {compared_pairs:,} pairs", file=sys.stderr)
    report_stats(name, "bytes", seconds, compared_pairs, compared_bytes)


def compare_casefold(first_string: str, second_string: str) -> bool:
    """Compare using Python's str.casefold()."""
    return first_string.casefold() == second_string.casefold()


def compare_regex_fullcase(first_string: str, second_string: str) -> bool:
    """Compare using regex with IGNORECASE | FULLCASE."""
    # Escape special regex characters and do a full match
    pattern = regex.compile(regex.escape(first_string), regex.IGNORECASE | regex.FULLCASE)
    return pattern.fullmatch(second_string) is not None


def compare_icu(first_string: str, second_string: str) -> bool:
    """Compare using ICU case folding."""
    first_folded = icu.UnicodeString(first_string).foldCase()
    second_folded = icu.UnicodeString(second_string).foldCase()
    return first_folded == second_folded


def compare_stringzilla(first_string: str, second_string: str) -> bool:
    """Compare using StringZilla's utf8_uncased_order."""
    return sz.utf8_uncased_order(first_string, second_string) == 0


def bench_case_find(
    name: str,
    haystack: str,
    needles: list[str],
    find_function: Callable[[str, str], int],
    time_limit_seconds: float = 10.0,
):
    """Benchmark case-insensitive substring search."""
    if not needles:
        print(f"{name}: no needles to search", file=sys.stderr)
        return

    haystack_bytes = len(haystack.encode("utf-8"))
    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    queries_done = 0
    total_matches = 0

    # Haystack is fixed, so bind it and cycle whole C-windowed passes over the needles.
    search = partial(find_function, haystack)
    while now_nanoseconds() < deadline_nanoseconds:
        pass_matches, pass_count = reduce_in_windows(
            search,
            needles,
            deadline_nanoseconds=deadline_nanoseconds,
        )
        total_matches += pass_matches
        queries_done += pass_count
        if pass_count < len(needles):
            break

    seconds = (now_nanoseconds() - start_time) / 1e9

    # Throughput = haystack bytes searched per query
    print(f"{name}: {total_matches:,} matches over {queries_done:,} queries", file=sys.stderr)
    report_stats(name, "bytes", seconds, queries_done, haystack_bytes * queries_done)


def find_casefold(haystack: str, needle: str) -> int:
    """Count occurrences using casefold on both strings."""
    haystack_folded = haystack.casefold()
    needle_folded = needle.casefold()
    if not needle_folded:
        return 0
    count = 0
    start = 0
    while True:
        pos = haystack_folded.find(needle_folded, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def find_regex_fullcase(haystack: str, needle: str) -> int:
    """Count occurrences using regex with IGNORECASE | FULLCASE."""
    if not needle:
        return 0
    pattern = regex.compile(regex.escape(needle), regex.IGNORECASE | regex.FULLCASE)
    # Use finditer for fair comparison (same Python loop overhead as StringZilla)
    return sum(1 for _ in pattern.finditer(haystack))


def find_icu(haystack: str, needle: str) -> int:
    """Count occurrences using ICU StringSearch."""
    if not needle:
        return 0
    collator = icu.Collator.createInstance(icu.Locale.getRoot())
    collator.setStrength(icu.Collator.SECONDARY)  # Case-insensitive
    searcher = icu.StringSearch(needle, haystack, collator)
    count = 0
    pos = searcher.nextMatch()
    while pos != -1:
        count += 1
        pos = searcher.nextMatch()
    return count


def find_stringzilla(haystack: str, needle: str) -> int:
    """Count occurrences using StringZilla's utf8_uncased_matches."""
    if not needle:
        return 0
    return sum(1 for _ in sz.utf8_uncased_matches(haystack, needle))


def bench_case_fold(
    name: str,
    strings: list[str],
    fold_function: Callable[[str], str | bytes],
    time_limit_seconds: float = 10.0,
):
    """Benchmark case folding transformation."""
    if not strings:
        print(f"{name}: no strings to fold", file=sys.stderr)
        return

    # Encode once: cumulative prefix sums of byte lengths for exact throughput
    # accounting without re-encoding inside the hot loop.
    cumulative_bytes = [0]
    for string in strings:
        cumulative_bytes.append(cumulative_bytes[-1] + len(string.encode("utf-8")))
    bytes_per_pass = cumulative_bytes[-1]

    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    processed_strings = 0

    # Cycle the dataset until the time limit, one pair at a time.
    for string in paced_items(itertools.cycle(strings), deadline_nanoseconds):
        _ = fold_function(string)
        processed_strings += 1

    seconds = (now_nanoseconds() - start_time) / 1e9
    full_passes, remainder = divmod(processed_strings, len(strings))
    processed_bytes = full_passes * bytes_per_pass + cumulative_bytes[remainder]

    report_stats(name, "bytes", seconds, processed_strings, processed_bytes)


def fold_casefold(s: str) -> str:
    """Fold using Python's str.casefold() - full Unicode."""
    return s.casefold()


def fold_stringzilla(s: str) -> bytes:
    """Fold using StringZilla's utf8_uncased_fold() - full Unicode."""
    return sz.utf8_uncased_fold(s)


def fold_icu(s: str) -> str:
    """Fold using ICU case folding."""
    return str(icu.UnicodeString(s).foldCase())


NORMALIZATION_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


def bench_normalize(
    name: str,
    strings: list[str],
    normalize_function: Callable[[str], str | bytes],
    time_limit_seconds: float = 10.0,
):
    """Benchmark a per-string Unicode normalization transform and report input-byte throughput."""
    if not strings:
        print(f"{name}: no strings to normalize", file=sys.stderr)
        return

    # Encode once: cumulative prefix sums of byte lengths for exact throughput
    # accounting without re-encoding inside the hot loop.
    cumulative_bytes = [0]
    for string in strings:
        cumulative_bytes.append(cumulative_bytes[-1] + len(string.encode("utf-8")))
    bytes_per_pass = cumulative_bytes[-1]

    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    processed_strings = 0

    # Cycle the dataset until the time limit, one string at a time.
    for string in paced_items(itertools.cycle(strings), deadline_nanoseconds):
        _ = normalize_function(string)
        processed_strings += 1

    seconds = (now_nanoseconds() - start_time) / 1e9
    full_passes, remainder = divmod(processed_strings, len(strings))
    processed_bytes = full_passes * bytes_per_pass + cumulative_bytes[remainder]

    report_stats(name, "bytes", seconds, processed_strings, processed_bytes)


def normalize_stringzilla(form: str, s: str) -> bytes:
    """Normalize using StringZilla's utf8_norm() - returns raw UTF-8 bytes."""
    return sz.utf8_norm(s, form)


def normalize_stdlib(form: str, s: str) -> str:
    """Normalize using Python's unicodedata.normalize()."""
    return unicodedata.normalize(form, s)


def make_normalize_icu(form: str) -> Callable[[str], str]:
    """Build an ICU Normalizer2-backed normalizer for one form.

    The `Normalizer2` instance is constructed once here, outside the hot loop, so
    the benchmark measures normalization rather than instance lookup. NFC/NFKC use
    COMPOSE, NFD/NFKD use DECOMPOSE; the underlying data set is `nfc` for the
    canonical forms and `nfkc` for the compatibility forms.
    """
    data_name = "nfkc" if form in ("NFKC", "NFKD") else "nfc"
    mode = icu.UNormalizationMode2.COMPOSE if form in ("NFC", "NFKC") else icu.UNormalizationMode2.DECOMPOSE
    normalizer = icu.Normalizer2.getInstance(None, data_name, mode)

    def normalize(s: str) -> str:
        return normalizer.normalize(s)

    return normalize


_main_epilog = """
Examples:

  # Benchmark all case folding and normalization operations
  %(prog)s --dataset README.md --tokens lines

  # Test only case folding
  %(prog)s --dataset README.md --tokens lines -k "casefold"

  # Test only normalization
  %(prog)s --dataset README.md --tokens lines -k "normalize"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark Unicode case folding and normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern: re.Pattern | None = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load and tokenize dataset
    pythonic_str = load_dataset(args.dataset, as_bytes=False, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(pythonic_str, args.tokens)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Create adjacent pairs for comparison benchmarks
    pairs = make_pairs(tokens)

    # Select subset of tokens as needles for search benchmarks
    # Filter to needles with >= 3 UTF-8 bytes (matching Rust benchmark)
    candidates = [t for t in tokens if len(t.encode("utf-8")) >= 3]
    # Random sample with fixed seed for reproducibility
    random.seed(42)
    search_needles = random.sample(candidates, min(1000, len(candidates))) if candidates else []

    total_tokens = len(tokens)
    total_pairs = len(pairs)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens
    total_bytes = len(pythonic_str)

    print(f"Dataset: {total_tokens:,} tokens, {total_bytes:,} bytes, {mean_token_length:.1f} avg token length")
    print(f"Pairs: {total_pairs:,}, Search needles: {len(search_needles)}")
    log_system_info()

    # Case-insensitive comparison
    print("Case-Insensitive Comparison")
    if should_run("case-insensitive-compare/stringzilla.utf8_uncased_order", filter_pattern):
        bench_case_compare("stringzilla.utf8_uncased_order", pairs, compare_stringzilla, args.time_limit)
    if should_run("case-insensitive-compare/str.casefold.eq", filter_pattern):
        bench_case_compare("str.casefold.eq", pairs, compare_casefold, args.time_limit)
    if should_run("case-insensitive-compare/regex.fullmatch<fullcase>", filter_pattern):
        bench_case_compare("regex.fullmatch<fullcase>", pairs, compare_regex_fullcase, args.time_limit)
    if should_run("case-insensitive-compare/icu.CaseMap.foldCase.eq", filter_pattern):
        bench_case_compare("icu.CaseMap.foldCase.eq", pairs, compare_icu, args.time_limit)

    # Case-insensitive substring search
    print("\nCase-Insensitive Substring Search")
    if should_run("case-insensitive-find/stringzilla.utf8_uncased_find", filter_pattern):
        bench_case_find(
            "stringzilla.utf8_uncased_find", pythonic_str, search_needles, find_stringzilla, args.time_limit
        )
    if should_run("case-insensitive-find/str.casefold.find", filter_pattern):
        bench_case_find("str.casefold.find", pythonic_str, search_needles, find_casefold, args.time_limit)
    if should_run("case-insensitive-find/regex.search<fullcase>", filter_pattern):
        bench_case_find("regex.search<fullcase>", pythonic_str, search_needles, find_regex_fullcase, args.time_limit)
    if should_run("case-insensitive-find/icu.StringSearch", filter_pattern):
        bench_case_find("icu.StringSearch", pythonic_str, search_needles, find_icu, args.time_limit)

    # Case folding transformation
    print("\nCase Folding Transformation")
    if should_run("case-fold/stringzilla.utf8_uncased_fold", filter_pattern):
        bench_case_fold("stringzilla.utf8_uncased_fold", tokens, fold_stringzilla, args.time_limit)
    if should_run("case-fold/str.casefold", filter_pattern):
        bench_case_fold("str.casefold", tokens, fold_casefold, args.time_limit)
    if should_run("case-fold/icu.CaseMap.foldCase", filter_pattern):
        bench_case_fold("icu.CaseMap.foldCase", tokens, fold_icu, args.time_limit)

    # Unicode normalization (NFC / NFD / NFKC / NFKD) - all forms measured
    print("\nUnicode Normalization")
    for form in NORMALIZATION_FORMS:
        suffix = form.lower()
        if should_run(f"normalize-{suffix}/stringzilla.utf8_norm", filter_pattern):
            bench_normalize(
                f"stringzilla.utf8_norm<{suffix}>", tokens, partial(normalize_stringzilla, form), args.time_limit
            )
        if should_run(f"normalize-{suffix}/unicodedata.normalize", filter_pattern):
            bench_normalize(
                f"unicodedata.normalize<{suffix}>", tokens, partial(normalize_stdlib, form), args.time_limit
            )
        if should_run(f"normalize-{suffix}/icu.Normalizer2", filter_pattern):
            bench_normalize(f"icu.Normalizer2<{suffix}>", tokens, make_normalize_icu(form), args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
