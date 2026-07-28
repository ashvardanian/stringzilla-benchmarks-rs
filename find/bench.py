# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "pyahocorasick",
# ]
# ///
"""Substring, reverse-substring and byte-set search benchmarks in Python. Mirrors `find/bench.rs`."""

import argparse
import re
import sys
from collections.abc import Callable
from functools import partial
from importlib.metadata import version as pkg_version

import ahocorasick as ahoc
import stringzilla as sz

from utils import (
    MeasureSpec,
    add_common_args,
    finish,
    log_dataset,
    log_timing_overhead,
    measure,
    pass_over,
    resolve_dataset,
    set_filter,
)


def log_system_info():
    """Log Python version and find library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- PyAhoCorasick: {pkg_version('pyahocorasick')}")
    print()  # Add blank line


NEEDLES_PER_PASS = 16

# The three bytesets `find/bench.rs` scans, so a pass covers the haystack three times.
BYTESETS = ("\n\r\x0b\x0c", "</>&'\"=[]", "0123456789")
BYTESET_PATTERNS = tuple(re.compile("[" + re.escape(byteset) + "]") for byteset in BYTESETS)


def bench_op(name: str, haystack, haystack_bytes: int, patterns, operation: Callable[..., int]):
    """
    One pass scans every pattern across the whole haystack.

    The pattern set is fixed, so every implementation scans identical work. The old
    loop ran against a deadline, so a faster engine got through more patterns than a
    slower one — and pattern cost varies by more than an order of magnitude.

    `haystack_bytes` is passed rather than measured: `len()` on a `str` counts
    codepoints, which published the four stdlib rows at 61% of their true rate on a
    Cyrillic corpus while the `sz.Str` rows, whose `len()` is bytes, were correct.
    """
    work = MeasureSpec(
        report="bytes",
        elements=len(patterns),
        total_bytes=haystack_bytes * len(patterns),
    )
    measure(name, work, pass_over(partial(operation, haystack), patterns))


def count_find(haystack, pattern) -> int:
    # Non-overlapping, as `find/bench.rs` counts. `sz.Str` indexes bytes and `str`
    # codepoints, so the stride is the needle measured in the haystack's own unit.
    stride = len(pattern) if isinstance(haystack, str) else len(pattern.encode())
    count, start = 0, 0
    while True:
        index = haystack.find(pattern, start)
        if index == -1:
            break
        count += 1
        start = index + stride
    return count


def count_rfind(haystack, pattern) -> int:
    count, start = 0, len(haystack) - 1
    while True:
        index = haystack.rfind(pattern, 0, start + 1)
        if index == -1:
            break
        count += 1
        start = index - 1
    return count


def count_regex(haystack: str, regex: re.Pattern) -> int:
    return sum(1 for _ in regex.finditer(haystack))


def count_aho(haystack: str, automaton) -> int:
    return sum(1 for _ in automaton.iter(haystack))


def build_automaton(needle: str):
    """Built before the clock starts, as `find/bench.rs` builds its FSAs."""
    automaton = ahoc.Automaton()
    automaton.add_word(needle, 1)
    automaton.make_automaton()
    return automaton


def count_byteset(haystack: sz.Str, characters: str) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find_first_of(characters, start)
        if index == -1:
            break
        count += 1
        start = index + 1
    return count


_main_epilog = """
Examples:

  %(prog)s --dataset README.md --tokens lines

  # Test only substring search operations
  %(prog)s --dataset data.txt --tokens lines -k "str.find|sz.Str.find"

  %(prog)s --dataset large.txt --tokens words -k "find_first_of"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark StringZilla find operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    set_filter(args.filter)

    # Resolve the working set from the shared manifest, identically to `utils.rs`.
    dataset = resolve_dataset("find", as_bytes=False, dataset_path=args.dataset)
    log_timing_overhead()
    tokens = dataset.tokens
    # The haystack is the concatenation of the resolved tokens, so the byte count the
    # rate is divided by is exactly `dataset.token_bytes`.
    pythonic_str = "".join(tokens)
    stringzilla_str = sz.Str(pythonic_str)
    # A fixed, evenly spaced needle sample, matching `find/bench.rs`, so every row
    # scans identical work rather than however far the deadline reached.
    stride = max(len(tokens) // NEEDLES_PER_PASS, 1)
    sample = tokens[::stride][:NEEDLES_PER_PASS]
    automatons = [build_automaton(needle) for needle in sample]
    haystack_bytes = dataset.token_bytes

    log_dataset(dataset)
    log_system_info()

    print("\nSubstring Search Benchmarks")
    bench_op("substring-forward/str.find", pythonic_str, haystack_bytes, sample, count_find)
    bench_op("substring-forward/stringzilla.Str.find", stringzilla_str, haystack_bytes, sample, count_find)
    bench_op("substring-backward/str.rfind", pythonic_str, haystack_bytes, sample, count_rfind)
    bench_op("substring-backward/stringzilla.Str.rfind", stringzilla_str, haystack_bytes, sample, count_rfind)
    bench_op("substring-forward/pyahocorasick.iter", pythonic_str, haystack_bytes, automatons, count_aho)

    print("\nCharacter Set Search")
    bench_op("byteset-forward/re.finditer", pythonic_str, haystack_bytes, BYTESET_PATTERNS, count_regex)
    bench_op(
        "byteset-forward/stringzilla.Str.find_first_of",
        stringzilla_str,
        haystack_bytes,
        BYTESETS,
        count_byteset,
    )

    finish()
    return 0


if __name__ == "__main__":
    exit(main())
