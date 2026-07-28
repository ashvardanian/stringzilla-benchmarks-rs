# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "regex",
#   "PyICU",
# ]
# ///
"""Unicode normalization and case-insensitive comparison benchmarks in Python. Mirrors `normalization/bench.rs`."""

import argparse
import random
import sys
import unicodedata
from collections.abc import Callable
from functools import partial
from importlib.metadata import version as pkg_version

import icu
import pyunormalize
import regex
import stringzilla as sz

from utils import (
    MeasureSpec,
    add_common_args,
    finish,
    log_dataset,
    log_timing_overhead,
    measure,
    note_unavailable,
    pass_over,
    resolve_dataset,
    set_filter,
)


def log_system_info():
    """Log Python version and library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- regex: {pkg_version('regex')}")
    print(f"- PyICU: {pkg_version('PyICU')} (ICU {icu.ICU_VERSION})")
    print(f"- pyunormalize: {pkg_version('pyunormalize')} (Unicode {pyunormalize.UCD_VERSION})")
    print()


def bench_case_compare(
    name: str,
    lefts: list[str],
    rights: list[str],
    compare_function: Callable[[str, str], bool],
    total_bytes: int,
):
    """One pass compares every pair, so the pair mixture is identical in every sample."""
    if not lefts:
        note_unavailable(name, "fewer than two tokens to pair")
        return
    work = MeasureSpec(report="bytes", elements=len(lefts), total_bytes=total_bytes)
    measure(name, work, pass_over(compare_function, lefts, rights))


def compare_casefold(first_string: str, second_string: str) -> bool:
    return first_string.casefold() == second_string.casefold()


def compare_regex_fullcase(first_string: str, second_string: str) -> bool:
    """Escape, compile and full-match, all inside the timed pass — hence the row's name.

    Precompiling is not an option: the corpus holds millions of distinct words and `regex`
    caches only 500 patterns, so a hoisted variant would thrash that cache under its lock.
    `regex.escape` is a per-character Python loop and is paid on every call either way.
    """
    pattern = regex.compile(regex.escape(first_string), regex.IGNORECASE | regex.FULLCASE)
    return pattern.fullmatch(second_string) is not None


def compare_icu(first_string: str, second_string: str) -> bool:
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
    haystack_bytes: int,
):
    """One pass searches every needle across the whole haystack."""
    if not needles:
        print(f"{name}: no needles to search", file=sys.stderr)
        return
    work = MeasureSpec(
        report="bytes",
        elements=len(needles),
        total_bytes=haystack_bytes * len(needles),
    )
    measure(name, work, pass_over(partial(find_function, haystack), needles))


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
    """Count occurrences with IGNORECASE | FULLCASE, escaping and compiling per needle.

    Only 16 needles run here, so the compile is a cache hit; the escape is still a
    per-character Python loop, and the row's name says so.
    """
    if not needle:
        return 0
    pattern = regex.compile(regex.escape(needle), regex.IGNORECASE | regex.FULLCASE)
    # Counted lazily, so the Python loop overhead matches the StringZilla row.
    return sum(1 for _ in pattern.finditer(haystack))


def make_find_icu() -> Callable[[str, str], int]:
    """Build an ICU StringSearch counter over one reused collator.

    The collator does not depend on the needle, so it is built here rather than inside the
    timed pass; only the `StringSearch` binding a needle to the haystack stays per-call.
    """
    collator = icu.Collator.createInstance(icu.Locale.getRoot())
    collator.setStrength(icu.Collator.SECONDARY)  # Case-insensitive

    def find(haystack: str, needle: str) -> int:
        if not needle:
            return 0
        searcher = icu.StringSearch(needle, haystack, collator)
        count = 0
        pos = searcher.nextMatch()
        while pos != -1:
            count += 1
            pos = searcher.nextMatch()
        return count

    return find


def find_stringzilla(haystack: str, needle: str) -> int:
    """Count occurrences using StringZilla's utf8_uncased_matches."""
    if not needle:
        return 0
    return sum(1 for _ in sz.utf8_uncased_matches(haystack, needle))


def bench_case_fold(
    name: str,
    strings: list[str],
    fold_function: Callable[[str], str | bytes],
    total_bytes: int,
):
    """One pass folds every string."""
    if not strings:
        print(f"{name}: nothing to process", file=sys.stderr)
        return
    work = MeasureSpec(report="bytes", elements=len(strings), total_bytes=total_bytes)
    measure(name, work, pass_over(fold_function, strings))


def fold_casefold(s: str) -> str:
    """Fold using Python's str.casefold() - full Unicode."""
    return s.casefold()


def fold_stringzilla(s: str) -> bytes:
    """Fold using StringZilla's utf8_uncased_fold() - full Unicode."""
    return sz.utf8_uncased_fold(s)


def fold_icu(s: str) -> str:
    return str(icu.UnicodeString(s).foldCase())


NORMALIZATION_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


def bench_normalize(
    name: str,
    strings: list[str],
    normalize_function: Callable[[str], str | bytes],
    total_bytes: int,
):
    """One pass normalizes every string."""
    if not strings:
        print(f"{name}: nothing to process", file=sys.stderr)
        return
    work = MeasureSpec(report="bytes", elements=len(strings), total_bytes=total_bytes)
    measure(name, work, pass_over(normalize_function, strings))


def normalize_stringzilla(form: str, s: str) -> bytes:
    """Normalize using StringZilla's utf8_norm() - returns raw UTF-8 bytes."""
    return sz.utf8_norm(s, form)


def normalize_stdlib(form: str, s: str) -> str:
    """Normalize using Python's unicodedata.normalize()."""
    return unicodedata.normalize(form, s)


def normalize_pyunormalize(form: str, s: str) -> str:
    """Normalize using `pyunormalize`, a pure-Python implementation of UAX #15.

    Included as the reference point for what the algorithm costs without a native
    extension behind it - it ships its own Unicode tables rather than deferring to
    the interpreter's, so it also serves as an independent oracle for the forms.
    """
    return pyunormalize.normalize(form, s)


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

  %(prog)s --dataset README.md --tokens lines

  # Test only case folding
  %(prog)s --dataset data.txt --tokens lines -k "casefold"

  # Test only normalization
  %(prog)s --dataset data.txt --tokens lines -k "normalize"
"""


# One pass scans the haystack once per needle, matching `find` and the Rust side.
NEEDLES_PER_PASS = 16


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
    set_filter(args.filter)

    # Resolve the working set from the shared manifest, identically to `utils.rs`.
    dataset = resolve_dataset("normalization", as_bytes=False, dataset_path=args.dataset)
    tokens = dataset.tokens
    pythonic_str = "".join(tokens)
    log_dataset(dataset)
    log_timing_overhead()

    # The manifest gives this suite `tokens = "file"`, so `tokens` is one 16 MB string.
    # Case-folding and normalization want that, but the find and compare groups need
    # real needles - searching for the whole haystack inside itself matches once, and
    # a single token yields no pairs at all. Rust splits the same buffer; mirror it.
    words = pythonic_str.split()
    lefts, rights = words[:-1], words[1:]

    # >= 3 UTF-8 bytes, as in Rust. Codepoint length is a lower bound on byte length,
    # so the encode only runs for the short ones.
    candidates = [w for w in words if len(w) >= 3 or len(w.encode("utf-8")) >= 3]
    random.seed(42)
    search_needles = random.sample(candidates, min(NEEDLES_PER_PASS, len(candidates))) if candidates else []

    total_tokens = len(tokens)
    total_pairs = len(lefts)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens
    # `token_bytes` is the denominator every bytes/s figure is quoted against, and what
    # `log_dataset` has already printed; `len(pythonic_str)` is codepoints.
    total_bytes = dataset.token_bytes
    pair_bytes = sum(len(item.encode("utf-8")) for item in lefts) + sum(len(item.encode("utf-8")) for item in rights)

    print(f"Dataset: {total_tokens:,} tokens, {total_bytes:,} bytes, {mean_token_length:.1f} avg token length")
    print(f"Pairs: {total_pairs:,}, Search needles: {len(search_needles)}")
    log_system_info()

    # Case-insensitive comparison
    print("Case-Insensitive Comparison")
    bench_case_compare(
        "case-insensitive-compare/stringzilla.utf8_uncased_order", lefts, rights, compare_stringzilla, pair_bytes
    )
    bench_case_compare("case-insensitive-compare/str.casefold.eq", lefts, rights, compare_casefold, pair_bytes)
    bench_case_compare(
        "case-insensitive-compare/regex.fullmatch<compile+match>",
        lefts,
        rights,
        compare_regex_fullcase,
        pair_bytes,
    )
    bench_case_compare("case-insensitive-compare/icu.CaseMap.foldCase.eq", lefts, rights, compare_icu, pair_bytes)

    # Case-insensitive substring search
    print("\nCase-Insensitive Substring Search")
    # The row is named for the function actually called, `utf8_uncased_matches`.
    bench_case_find(
        "case-insensitive-find/stringzilla.utf8_uncased_matches",
        pythonic_str,
        search_needles,
        find_stringzilla,
        total_bytes,
    )
    bench_case_find("case-insensitive-find/str.casefold.find", pythonic_str, search_needles, find_casefold, total_bytes)
    bench_case_find(
        "case-insensitive-find/regex.finditer<compile+match>",
        pythonic_str,
        search_needles,
        find_regex_fullcase,
        total_bytes,
    )
    bench_case_find(
        "case-insensitive-find/icu.StringSearch", pythonic_str, search_needles, make_find_icu(), total_bytes
    )

    # Case folding transformation
    print("\nCase Folding Transformation")
    bench_case_fold("case-fold/stringzilla.utf8_uncased_fold", tokens, fold_stringzilla, total_bytes)
    bench_case_fold("case-fold/str.casefold", tokens, fold_casefold, total_bytes)
    bench_case_fold("case-fold/icu.CaseMap.foldCase", tokens, fold_icu, total_bytes)

    # Unicode normalization (NFC / NFD / NFKC / NFKD) - all forms measured
    print("\nUnicode Normalization")
    for form in NORMALIZATION_FORMS:
        suffix = form.lower()
        bench_normalize(
            f"normalize-{suffix}/stringzilla.utf8_norm", tokens, partial(normalize_stringzilla, form), total_bytes
        )
        bench_normalize(
            f"normalize-{suffix}/unicodedata.normalize", tokens, partial(normalize_stdlib, form), total_bytes
        )
        bench_normalize(f"normalize-{suffix}/icu.Normalizer2", tokens, make_normalize_icu(form), total_bytes)
        bench_normalize(
            f"normalize-{suffix}/pyunormalize.normalize", tokens, partial(normalize_pyunormalize, form), total_bytes
        )

    finish()
    return 0


if __name__ == "__main__":
    exit(main())
