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
"""Tokenization benchmarks in Python: UTF-8 iteration and segmentation. Mirrors `tokenization/bench.rs`."""

import argparse
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
):
    """Benchmark a whole-text tokenizer/scanner by counting what it yields.

    `count_function` consumes the entire `text` once per call and returns an integer
    count (token count, codepoint count, ...). The StringZilla and regex/ICU paths
    count lazily via `sum(1 for _ in ...)` so no token list is materialized.
    Throughput is reported as input bytes per second.
    """
    text_byte_length = len(text.encode("utf-8")) if isinstance(text, str) else len(text)
    work = MeasureSpec(report="bytes", elements=1, total_bytes=text_byte_length)
    measure(name, work, lambda: count_function(text))


def bench_split_lines(
    name: str,
    lines: list[str],
    count_function: Callable[[str], int],
):
    """Benchmark a splitter by processing one document line per call, cycling the lines.

    Unlike the whole-text `bench_tokenize`, each call splits a single line, so the working set
    is one line rather than the entire file. Splitting is compute-bound, so the per-byte rate
    still mirrors a whole-file pass; only the working set changes. Throughput is reported as the
    sum of the byte lengths of every line processed.
    """
    work = MeasureSpec(
        report="bytes",
        elements=len(lines),
        total_bytes=sum(len(line.encode("utf-8")) for line in lines),
    )
    measure(name, work, pass_over(count_function, lines))


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

  %(prog)s --dataset README.md --tokens file

  # Test only word segmentation
  %(prog)s --dataset data.txt --tokens file -k "words"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark UTF-8 tokenization and iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    set_filter(args.filter)

    # Resolve the working set from the shared manifest, identically to `utils.rs`.
    dataset = resolve_dataset("tokenization", as_bytes=False, dataset_path=args.dataset)
    lines = dataset.tokens
    pythonic_str = "".join(lines)

    log_dataset(dataset)
    log_timing_overhead()
    log_system_info()

    # UTF-8 word segmentation (TR29) per document line, cycling the lines.
    print("Word Segmentation (TR29)")
    bench_split_lines("tokenize-words/stringzilla.utf8_wordbreaks", lines, count_words_stringzilla)
    bench_split_lines("tokenize-words/uniseg.words", lines, count_words_uniseg)
    bench_split_lines("tokenize-words/icu.BreakIterator", lines, make_count_words_icu())

    # UTF-8 grapheme cluster segmentation (TR29) per document line, cycling the lines.
    print("\nGrapheme Cluster Segmentation (TR29)")
    bench_split_lines("tokenize-graphemes-tr29/stringzilla.utf8_graphemes", lines, count_graphemes_stringzilla)
    bench_split_lines("tokenize-graphemes-tr29/regex.finditer", lines, count_graphemes_regex)
    bench_split_lines("tokenize-graphemes-tr29/grapheme.graphemes", lines, count_graphemes_grapheme)
    bench_split_lines("tokenize-graphemes-tr29/uniseg.grapheme_clusters", lines, count_graphemes_uniseg)
    bench_split_lines("tokenize-graphemes-tr29/icu.BreakIterator", lines, make_count_graphemes_icu())

    # UTF-8 sentence segmentation (TR29) per document line, cycling the lines.
    print("\nSentence Segmentation (TR29)")
    bench_split_lines("tokenize-sentences-tr29/stringzilla.utf8_sentences", lines, count_sentences_stringzilla)
    bench_split_lines("tokenize-sentences-tr29/uniseg.sentences", lines, count_sentences_uniseg)
    bench_split_lines("tokenize-sentences-tr29/icu.BreakIterator", lines, make_count_sentences_icu())

    # UTF-8 line-break opportunity segmentation (UAX#14) per document line, cycling the lines.
    print("\nLine-Break Segmentation (UAX#14)")
    bench_split_lines("tokenize-lines-uax14/stringzilla.utf8_linebreaks", lines, count_lines_stringzilla)
    bench_split_lines("tokenize-lines-uax14/uniseg.line_break", lines, count_lines_uniseg)
    bench_split_lines("tokenize-lines-uax14/icu.BreakIterator", lines, make_count_lines_icu())

    # UTF-8 whitespace splitting per document line, cycling the lines.
    print("\nWhitespace Splitting")
    bench_split_lines("tokenize-whitespace/stringzilla.utf8_split_whitespaces", lines, count_whitespace_stringzilla)
    bench_split_lines("tokenize-whitespace/regex.split", lines, count_whitespace_regex)

    # UTF-8 newline splitting per document line, cycling the lines.
    print("\nNewline Splitting")
    bench_split_lines("tokenize-newlines/stringzilla.utf8_split_newlines", lines, count_newlines_stringzilla)

    # UTF-8 codepoint counting over the raw bytes (fair O(n)-from-bytes comparison;
    # `len(str)` is O(1) in CPython, so we decode-and-count as the stdlib baseline).
    print("\nCodepoint Counting")
    document_bytes = pythonic_str.encode("utf-8")
    bench_tokenize("utf8-count/stringzilla.utf8_count", document_bytes, count_codepoints_stringzilla)
    bench_tokenize("utf8-count/str.decode.len", document_bytes, count_codepoints_decode)

    finish()
    return 0


if __name__ == "__main__":
    exit(main())
