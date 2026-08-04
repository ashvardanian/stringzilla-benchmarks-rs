# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "pyahocorasick",
# ]
# ///
"""
Python substring, byteset, and Aho–Corasick benches.

- Substring: str.find/rfind, sz.Str.find/rfind (per token)
- Byteset: re.finditer, sz.Str.find_first_of
- Aho–Corasick: per-token (build per pattern) and multi-token (one pass)

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run find/bench.py --dataset README.md --tokens lines
  uv run find/bench.py --dataset data/xlsum/xlsum.csv --tokens words -k "str.find"
  STRINGWARS_DATASET=README.md STRINGWARS_TOKENS=lines uv run find/bench.py

Timing via time.monotonic_ns.; throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import re
import sys
from collections.abc import Callable
from functools import partial
from importlib.metadata import version as pkg_version

import ahocorasick as ahoc
import stringzilla as sz

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    reduce_in_windows,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
)


def log_system_info():
    """Log Python version and find library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- PyAhoCorasick: {pkg_version('pyahocorasick')}")
    print()  # Add blank line


def bench_op(name: str, haystack, patterns, operation: Callable[..., int], time_limit_seconds: float = 10.0):
    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    haystack_length = len(haystack)

    # Haystack is fixed per call, so bind it with a (C-level) partial and reduce the
    # per-pattern results in C windows.
    received_results, requested_queries = reduce_in_windows(
        partial(operation, haystack),
        patterns,
        deadline_nanoseconds=deadline_nanoseconds,
    )

    end_time = now_nanoseconds()
    seconds = (end_time - start_time) / 1e9

    print(
        f"{name}: {received_results:,} results over {requested_queries:,} queries",
        file=sys.stderr,
    )
    report_stats(
        name,
        "bytes",
        seconds,
        requested_queries,
        haystack_length * requested_queries,
    )


def count_find(haystack, pattern) -> int:
    count, start = 0, 0
    while True:
        index = haystack.find(pattern, start)
        if index == -1:
            break
        count += 1
        start = index + 1
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


def count_aho_multi(haystack: str, automaton) -> int:
    # Count all matches over all tokens in a single pass
    return sum(1 for _ in automaton.iter(haystack))


def count_aho(haystack: str, pattern: str) -> int:
    # Build automaton for a single pattern and count all inclusions
    automaton = ahoc.Automaton()
    automaton.add_word(pattern, 1)
    automaton.make_automaton()
    return sum(1 for _ in automaton.iter(haystack))


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

  # Benchmark all find operations with default settings
  %(prog)s --dataset README.md --tokens lines

  # Test only substring search operations
  %(prog)s --dataset README.md --tokens lines -k "str.find|sz.Str.find"

  # Benchmark character set searches
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
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load and tokenize dataset
    pythonic_str = load_dataset(args.dataset, as_bytes=False, size_limit=args.dataset_limit)
    tokens_mode = resolve_tokens(args.tokens, "words")
    tokens = tokenize_dataset(pythonic_str, tokens_mode)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    stringzilla_str = sz.Str(pythonic_str)
    total_tokens = len(tokens)
    mean_token_length = sum(len(t) for t in tokens) / total_tokens

    print(f"Dataset: {total_tokens:,} tokens, {len(pythonic_str):,} bytes, {mean_token_length:.1f} avg token length")
    log_system_info()

    print("\nSubstring Search Benchmarks")
    if should_run("substring-forward/str.find", filter_pattern):
        bench_op("str.find", pythonic_str, tokens[::-1], count_find, args.time_limit)
    if should_run("substring-forward/stringzilla.Str.find", filter_pattern):
        bench_op("stringzilla.Str.find", stringzilla_str, tokens[::-1], count_find, args.time_limit)
    if should_run("substring-backward/str.rfind", filter_pattern):
        bench_op("str.rfind", pythonic_str, tokens, count_rfind, args.time_limit)
    if should_run("substring-backward/stringzilla.Str.rfind", filter_pattern):
        bench_op("stringzilla.Str.rfind", stringzilla_str, tokens, count_rfind, args.time_limit)
    if should_run("substring-forward/pyahocorasick.iter", filter_pattern):
        bench_op("pyahocorasick.iter", pythonic_str, tokens[::-1], count_aho, args.time_limit)

    print("\nCharacter Set Search")
    if args.tokens == "lines":
        re_chars = re.compile(r"[\n\r]")  # newlines: LF, CR
        sz_chars = "\n\r"
    else:
        re_chars = re.compile(r"[\t\n\r ]")  # whitespace: space, tab, LF, CR
        sz_chars = " \t\n\r"
    if should_run("byteset-forward/re.finditer", filter_pattern):
        bench_op("re.finditer", pythonic_str, [re_chars], count_regex, args.time_limit)
    if should_run("byteset-forward/stringzilla.Str.find_first_of", filter_pattern):
        bench_op("stringzilla.Str.find_first_of", stringzilla_str, [sz_chars], count_byteset, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
