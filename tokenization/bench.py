# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "regex",
#   "PyICU",
#   "uniseg",
#   "grapheme",
# ]
# ///
"""
Python UTF-8 tokenization and iteration benchmarks.

Benchmarks segmentation and codepoint operations:
- TR29 word, grapheme, and sentence segmentation (lazy iterators), per document line
- UAX#14 line-break opportunity segmentation (lazy iterators), per document line
- Unicode whitespace and newline splitting, per document line
- UTF-8 codepoint counting, over the whole document

The splitter benches process one document line per call, cycling through the lines (default
tokenization mode is 'lines'); the codepoint counter scans the whole document buffer.

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file'); defaults to 'lines'

Examples:
  uv run tokenization/bench.py --dataset README.md
  STRINGWARS_DATASET=README.md uv run tokenization/bench.py

Timing via time.monotonic_ns; throughput in decimal GB/s. Filter with -k/--filter.
"""

import argparse
import itertools
import re
import sys
from collections.abc import Callable
from importlib.metadata import version as pkg_version

import grapheme
import icu
import regex
import stringzilla as sz
import uniseg.graphemecluster
import uniseg.linebreak
import uniseg.sentencebreak
import uniseg.wordbreak

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    paced_items,
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
    print(f"- uniseg: {pkg_version('uniseg')}")
    print(f"- grapheme: {pkg_version('grapheme')}")
    print()


def bench_tokenize(
    name: str,
    text: str | bytes,
    count_function: Callable[[str | bytes], int],
    time_limit_seconds: float = 10.0,
):
    """Benchmark a whole-text tokenizer/scanner by counting what it yields.

    `count_function` consumes the entire `text` once per call and returns an integer
    count (token count, codepoint count, ...). The StringZilla and regex/ICU paths
    count lazily via `sum(1 for _ in ...)` so no token list is materialized.
    Throughput is reported as input bytes per second.
    """
    text_byte_length = len(text.encode("utf-8")) if isinstance(text, str) else len(text)

    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    passes = 0
    token_total = 0
    # Each yielded sentinel triggers one full-text pass; a pass is expensive, so we
    # check the deadline every iteration (step=1) rather than overshoot a large file.
    for _ in paced_items(itertools.count(), deadline_nanoseconds, step=1):
        token_total += count_function(text)
        passes += 1

    seconds = (now_nanoseconds() - start_time) / 1e9
    tokens_per_pass = token_total // passes if passes else 0

    print(f"{name}: {tokens_per_pass:,} tokens per pass over {passes:,} passes", file=sys.stderr)
    report_stats(name, "bytes", seconds, passes, text_byte_length * passes)


def bench_split_lines(
    name: str,
    lines: list[str],
    count_function: Callable[[str], int],
    time_limit_seconds: float = 10.0,
):
    """Benchmark a splitter by processing one document line per call, cycling the lines.

    Unlike the whole-text `bench_tokenize`, each call splits a single line, so the working set
    is one line rather than the entire file. Splitting is compute-bound, so the per-byte rate
    still mirrors a whole-file pass; only the working set changes. Throughput is reported as the
    sum of the byte lengths of every line processed.
    """
    line_byte_lengths = [len(line.encode("utf-8")) for line in lines]

    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    lines_processed = 0
    token_total = 0
    bytes_total = 0
    for line, line_byte_length in paced_items(
        itertools.cycle(zip(lines, line_byte_lengths, strict=True)), deadline_nanoseconds
    ):
        token_total += count_function(line)
        bytes_total += line_byte_length
        lines_processed += 1

    seconds = (now_nanoseconds() - start_time) / 1e9
    tokens_per_line = token_total / lines_processed if lines_processed else 0

    print(
        f"{name}: {tokens_per_line:,.1f} tokens per line over {lines_processed:,} lines",
        file=sys.stderr,
    )
    report_stats(name, "bytes", seconds, lines_processed, bytes_total)


def count_words_stringzilla(text: str) -> int:
    """Count TR29 words lazily via StringZilla's utf8_wordbreaks (no list built).

    `utf8_wordbreaks` tiles the input contiguously (every segment, spaces and punctuation included),
    matching `unicode-segmentation::split_word_bounds` and ICU rather than the word-like-only filter.
    """
    return sum(1 for _ in sz.utf8_wordbreaks(text))


def count_words_uniseg(text: str) -> int:
    """Count TR29 words lazily via uniseg's word_break iterator."""
    return sum(1 for _ in uniseg.wordbreak.words(text))


def make_count_words_icu() -> Callable[[str], int]:
    """Build an ICU word BreakIterator counter, reusing one iterator instance.

    Iterating the break iterator yields every boundary segment (words, punctuation,
    and whitespace), mirroring the Rust ICU WordSegmenter baseline which also counts
    boundary segments rather than only word-like ones.
    """
    break_iterator = icu.BreakIterator.createWordInstance(icu.Locale.getRoot())

    def count(text: str) -> int:
        break_iterator.setText(text)
        segments = 0
        previous = 0
        for boundary in break_iterator:
            segment = text[previous:boundary]  # materialize each segment, matching StringZilla's per-unit Str output
            _ = segment
            previous = boundary
            segments += 1
        return segments

    return count


def count_graphemes_stringzilla(text: str) -> int:
    """Count grapheme clusters lazily via StringZilla's utf8_graphemes (no list built)."""
    return sum(1 for _ in sz.utf8_graphemes(text, skip_empty=True))


def count_graphemes_regex(text: str) -> int:
    """Count grapheme clusters via the `regex` module's \\X meta-sequence (lazy)."""
    return sum(1 for _ in regex.finditer(r"\X", text))


def count_graphemes_grapheme(text: str) -> int:
    """Count grapheme clusters via the `grapheme` package (lazy)."""
    return sum(1 for _ in grapheme.graphemes(text))


def count_graphemes_uniseg(text: str) -> int:
    """Count grapheme clusters lazily via uniseg's grapheme_clusters iterator."""
    return sum(1 for _ in uniseg.graphemecluster.grapheme_clusters(text))


def make_count_graphemes_icu() -> Callable[[str], int]:
    """Build an ICU character (grapheme) BreakIterator counter, reusing one iterator instance."""
    break_iterator = icu.BreakIterator.createCharacterInstance(icu.Locale.getRoot())

    def count(text: str) -> int:
        break_iterator.setText(text)
        segments = 0
        previous = 0
        for boundary in break_iterator:
            segment = text[previous:boundary]  # materialize each segment, matching StringZilla's per-unit Str output
            _ = segment
            previous = boundary
            segments += 1
        return segments

    return count


def count_sentences_stringzilla(text: str) -> int:
    """Count sentences lazily via StringZilla's utf8_sentences (no list built)."""
    return sum(1 for _ in sz.utf8_sentences(text, skip_empty=True))


def count_sentences_uniseg(text: str) -> int:
    """Count TR29 sentences lazily via uniseg's sentence_break iterator."""
    return sum(1 for _ in uniseg.sentencebreak.sentences(text))


def make_count_sentences_icu() -> Callable[[str], int]:
    """Build an ICU sentence BreakIterator counter, reusing one iterator instance."""
    break_iterator = icu.BreakIterator.createSentenceInstance(icu.Locale.getRoot())

    def count(text: str) -> int:
        break_iterator.setText(text)
        segments = 0
        previous = 0
        for boundary in break_iterator:
            segment = text[previous:boundary]  # materialize each segment, matching StringZilla's per-unit Str output
            _ = segment
            previous = boundary
            segments += 1
        return segments

    return count


def count_lines_stringzilla(text: str) -> int:
    """Count UAX#14 line-break opportunities lazily via StringZilla's utf8_linebreaks."""
    return sum(1 for _ in sz.utf8_linebreaks(text, skip_empty=True))


def count_lines_uniseg(text: str) -> int:
    """Count UAX#14 line-break units lazily via uniseg's line_break iterator."""
    return sum(1 for _ in uniseg.linebreak.line_break_units(text))


def make_count_lines_icu() -> Callable[[str], int]:
    """Build an ICU line BreakIterator counter, reusing one iterator instance."""
    break_iterator = icu.BreakIterator.createLineInstance(icu.Locale.getRoot())

    def count(text: str) -> int:
        break_iterator.setText(text)
        segments = 0
        previous = 0
        for boundary in break_iterator:
            segment = text[previous:boundary]  # materialize each segment, matching StringZilla's per-unit Str output
            _ = segment
            previous = boundary
            segments += 1
        return segments

    return count


def count_whitespace_stringzilla(text: str) -> int:
    """Count whitespace-delimited tokens lazily via StringZilla's utf8_split_whitespaces.

    `skip_empty=True` drops the empty segments between whitespace runs, matching `str.split()`.
    """
    return sum(1 for _ in sz.utf8_split_whitespaces(text, skip_empty=True))


def count_whitespace_regex(text: str) -> int:
    """Count whitespace-delimited tokens via the `regex` module's \\s+ split (allocates a list)."""
    return len(regex.split(r"\s+", text))


def count_newlines_stringzilla(text: str) -> int:
    """Count lines lazily via StringZilla's utf8_split_newlines (no list built)."""
    return sum(1 for _ in sz.utf8_split_newlines(text))


def count_codepoints_stringzilla(data: bytes) -> int:
    """Count UTF-8 codepoints in raw bytes via StringZilla's utf8_count (SIMD scan)."""
    return sz.utf8_count(data)


def count_codepoints_decode(data: bytes) -> int:
    """Count UTF-8 codepoints by decoding the bytes to a str, then len() (allocates)."""
    return len(data.decode("utf-8"))


_main_epilog = """
Examples:

  # Benchmark all tokenization and iteration operations
  %(prog)s --dataset README.md --tokens file

  # Test only word segmentation
  %(prog)s --dataset README.md --tokens file -k "words"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark UTF-8 tokenization and iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser, default_dataset="data/xlsum/xlsum.csv", default_dataset_limit="64mb")

    args = parser.parse_args()

    # Compile filter pattern
    filter_pattern: re.Pattern | None = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load the dataset and tokenize it into document lines (the per-line splitter work items).
    tokens_mode = args.tokens
    pythonic_str = load_dataset(args.dataset, as_bytes=False, size_limit=args.dataset_limit)
    lines = tokenize_dataset(pythonic_str, tokens_mode)

    if not lines:
        print("No tokens found in dataset")
        return 1

    total_tokens = len(lines)
    mean_token_length = sum(len(t) for t in lines) / total_tokens
    total_bytes = len(pythonic_str)

    print(f"Dataset: {total_tokens:,} tokens, {total_bytes:,} bytes, {mean_token_length:.1f} avg token length")
    log_system_info()

    # UTF-8 word segmentation (TR29) per document line, cycling the lines.
    print("Word Segmentation (TR29)")
    if should_run("tokenize-words/stringzilla.utf8_wordbreaks", filter_pattern):
        bench_split_lines("stringzilla.utf8_wordbreaks", lines, count_words_stringzilla, args.time_limit)
    if should_run("tokenize-words/uniseg.words", filter_pattern):
        bench_split_lines("uniseg.words", lines, count_words_uniseg, args.time_limit)
    if should_run("tokenize-words/icu.BreakIterator", filter_pattern):
        bench_split_lines("icu.BreakIterator", lines, make_count_words_icu(), args.time_limit)

    # UTF-8 grapheme cluster segmentation (TR29) per document line, cycling the lines.
    print("\nGrapheme Cluster Segmentation (TR29)")
    if should_run("tokenize-graphemes-tr29/stringzilla.utf8_graphemes", filter_pattern):
        bench_split_lines("stringzilla.utf8_graphemes", lines, count_graphemes_stringzilla, args.time_limit)
    if should_run("tokenize-graphemes-tr29/regex.finditer", filter_pattern):
        bench_split_lines("regex.finditer", lines, count_graphemes_regex, args.time_limit)
    if should_run("tokenize-graphemes-tr29/grapheme.graphemes", filter_pattern):
        bench_split_lines("grapheme.graphemes", lines, count_graphemes_grapheme, args.time_limit)
    if should_run("tokenize-graphemes-tr29/uniseg.grapheme_clusters", filter_pattern):
        bench_split_lines("uniseg.grapheme_clusters", lines, count_graphemes_uniseg, args.time_limit)
    if should_run("tokenize-graphemes-tr29/icu.BreakIterator", filter_pattern):
        bench_split_lines("icu.BreakIterator", lines, make_count_graphemes_icu(), args.time_limit)

    # UTF-8 sentence segmentation (TR29) per document line, cycling the lines.
    print("\nSentence Segmentation (TR29)")
    if should_run("tokenize-sentences-tr29/stringzilla.utf8_sentences", filter_pattern):
        bench_split_lines("stringzilla.utf8_sentences", lines, count_sentences_stringzilla, args.time_limit)
    if should_run("tokenize-sentences-tr29/uniseg.sentences", filter_pattern):
        bench_split_lines("uniseg.sentences", lines, count_sentences_uniseg, args.time_limit)
    if should_run("tokenize-sentences-tr29/icu.BreakIterator", filter_pattern):
        bench_split_lines("icu.BreakIterator", lines, make_count_sentences_icu(), args.time_limit)

    # UTF-8 line-break opportunity segmentation (UAX#14) per document line, cycling the lines.
    print("\nLine-Break Segmentation (UAX#14)")
    if should_run("tokenize-lines-uax14/stringzilla.utf8_linebreaks", filter_pattern):
        bench_split_lines("stringzilla.utf8_linebreaks", lines, count_lines_stringzilla, args.time_limit)
    if should_run("tokenize-lines-uax14/uniseg.line_break", filter_pattern):
        bench_split_lines("uniseg.line_break", lines, count_lines_uniseg, args.time_limit)
    if should_run("tokenize-lines-uax14/icu.BreakIterator", filter_pattern):
        bench_split_lines("icu.BreakIterator", lines, make_count_lines_icu(), args.time_limit)

    # UTF-8 whitespace splitting per document line, cycling the lines.
    print("\nWhitespace Splitting")
    if should_run("tokenize-whitespace/stringzilla.utf8_split_whitespaces", filter_pattern):
        bench_split_lines("stringzilla.utf8_split_whitespaces", lines, count_whitespace_stringzilla, args.time_limit)
    if should_run("tokenize-whitespace/regex.split", filter_pattern):
        bench_split_lines("regex.split", lines, count_whitespace_regex, args.time_limit)

    # UTF-8 newline splitting per document line, cycling the lines.
    print("\nNewline Splitting")
    if should_run("tokenize-newlines/stringzilla.utf8_split_newlines", filter_pattern):
        bench_split_lines("stringzilla.utf8_split_newlines", lines, count_newlines_stringzilla, args.time_limit)

    # UTF-8 codepoint counting over the raw bytes (fair O(n)-from-bytes comparison;
    # `len(str)` is O(1) in CPython, so we decode-and-count as the stdlib baseline).
    print("\nCodepoint Counting")
    document_bytes = pythonic_str.encode("utf-8")
    if should_run("utf8-count/stringzilla.utf8_count", filter_pattern):
        bench_tokenize("stringzilla.utf8_count", document_bytes, count_codepoints_stringzilla, args.time_limit)
    if should_run("utf8-count/str.decode.len", filter_pattern):
        bench_tokenize("str.decode.len", document_bytes, count_codepoints_decode, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
