# UTF-8 Tokenization & Iteration Benchmarks

Benchmarks for UTF-8 segmentation and codepoint iteration — whitespace, newline, and TR29 word splitting, UTF-8 character counting and decoding, and locating the Nth codepoint — across different languages and hardware platforms.
Different scripts stress UTF-8 processing in different ways.
Sections run from the coarsest splits to the finest units.
Tables below cover:

- __Korean__: 3-byte Hangul syllables with single-byte whitespace between words - representative for tokenization workloads
- __Chinese__: 3-byte CJK characters with rare whitespace - tests raw byte throughput
- __Arabic__: 2-byte Arabic script with regular punctuation - good for newline splitting benchmarks
- __French__: Mixed 1-2 byte Latin with high diacritic density
- __English__: Mostly 1-byte ASCII baseline

## Newlines

There are 8 characters classified as forced newline delimiters in Unicode.
Five overlap with the ASCII whitespace set: line feed _U+000A_, vertical tab _U+000B_, form feed _U+000C_, carriage return _U+000D_, and the carriage return plus line feed combination _U+000D U+000A_ treated as a single break.
The remaining three are the Next Line control _U+0085_, the line separator _U+2028_, and the paragraph separator _U+2029_.
Unlike the optional break opportunities of UAX#14, these always end a line.

### Intel Xeon4 Sapphire Rapids

| Library                            |       English |       Chinese |        Arabic |        French |        Korean |
| ---------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                               |               |               |               |               |               |
| `stringzilla::utf8_split_newlines` | __2.51 GB/s__ | __1.95 GB/s__ | __3.06 GB/s__ | __2.23 GB/s__ | __2.52 GB/s__ |
| `std::split<is_unicode_newline>`   |     0.98 GB/s |     1.17 GB/s |     0.57 GB/s |     0.77 GB/s |     0.69 GB/s |
|                                    |               |               |               |               |               |
| Python                             |               |               |               |               |               |
| `stringzilla.utf8_split_newlines`  |     0.32 GB/s |     0.27 GB/s |     0.39 GB/s |     0.27 GB/s |     0.36 GB/s |

> Rust measured June 23, 2026; Python June 28, 2026.

### Apple M5 Pro

| Library                            |       English |       Chinese |        Arabic |        French |        Korean |
| ---------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                               |               |               |               |               |               |
| `stringzilla::utf8_split_newlines` | __2.88 GB/s__ | __2.47 GB/s__ | __3.31 GB/s__ | __2.70 GB/s__ | __2.84 GB/s__ |
| `std::split<is_unicode_newline>`   |     1.85 GB/s |     2.21 GB/s |     1.15 GB/s |     1.38 GB/s |     1.59 GB/s |
|                                    |               |               |               |               |               |
| Python                             |               |               |               |               |               |
| `stringzilla.utf8_split_newlines`  |     2.42 GB/s |     2.02 GB/s |     2.87 GB/s |     2.24 GB/s |     2.36 GB/s |

> Measured July 29, 2026.

## UAX#14 Line Break Opportunities

Marks every position a renderer may wrap.
Mandatory breaks (classes BK, CR, LF, NL — the same forced breaks as the Newlines section) are separated from the allowed soft-wrap points.
Around 40 rules (LB1–LB31) govern soft wraps after spaces and hyphens, around CJK ideographs, and East-Asian-Width cases such as LB19.
It emits far more opportunities than the hard newline set.

### Intel Xeon4 Sapphire Rapids

| Library                         |       English |       Chinese |        Arabic |        French |        Korean |
| ------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                            |               |               |               |               |               |
| `stringzilla::utf8_linebreaks`  | __0.36 GB/s__ |     0.19 GB/s |     0.26 GB/s | __0.29 GB/s__ |     0.20 GB/s |
| `unicode-linebreak::linebreaks` |     0.20 GB/s | __0.50 GB/s__ | __0.45 GB/s__ |     0.21 GB/s | __0.50 GB/s__ |
| `icu::LineSegmenter`            |     0.08 GB/s |     0.13 GB/s |     0.15 GB/s |     0.08 GB/s |     0.12 GB/s |
|                                 |               |               |               |               |               |
| Python                          |               |               |               |               |               |
| `stringzilla.utf8_linebreaks`   | __0.09 GB/s__ | __0.06 GB/s__ | __0.10 GB/s__ | __0.08 GB/s__ | __0.06 GB/s__ |
| `uniseg.line_break`             |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |
| `icu.BreakIterator`             |     0.05 GB/s |     0.04 GB/s |     0.08 GB/s |     0.04 GB/s |     0.04 GB/s |

> Rust measured June 23, 2026; Python June 28, 2026.

### Apple M5 Pro

| Library                         |       English |       Chinese |        Arabic |        French |        Korean |
| ------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                            |               |               |               |               |               |
| `stringzilla::utf8_linebreaks`  | __0.25 GB/s__ |     0.21 GB/s |     0.25 GB/s | __0.24 GB/s__ |     0.21 GB/s |
| `unicode-linebreak::linebreaks` |     0.22 GB/s | __0.57 GB/s__ | __0.59 GB/s__ |     0.22 GB/s | __0.63 GB/s__ |
| `icu::LineSegmenter`            |     0.16 GB/s |     0.27 GB/s |     0.29 GB/s |     0.16 GB/s |     0.25 GB/s |
|                                 |               |               |               |               |               |
| Python                          |               |               |               |               |               |
| `stringzilla.utf8_linebreaks`   | __0.16 GB/s__ |     0.11 GB/s |     0.19 GB/s | __0.16 GB/s__ |     0.11 GB/s |
| `icu.BreakIterator`             |     0.13 GB/s | __0.13 GB/s__ | __0.22 GB/s__ |     0.13 GB/s | __0.12 GB/s__ |

> Measured July 29, 2026.

## TR29 Sentences

UAX#29 sentence boundaries use 12 Sentence_Break classes (STerm, ATerm, Close, SContinue, Sep, CR, LF, Sp, Lower, Upper, OLetter, Numeric, Extend, Format).
Sentences end at `.!?` and the Unicode terminators.
Abbreviations (U.S.A.), decimals (3.14), and a following continuation do not break.

### Intel Xeon4 Sapphire Rapids

| Library                                       |       English |       Chinese |        Arabic |        French |        Korean |
| --------------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                                          |               |               |               |               |               |
| `stringzilla::utf8_sentences`                 | __0.64 GB/s__ | __0.35 GB/s__ | __0.63 GB/s__ | __0.62 GB/s__ | __0.61 GB/s__ |
| `unicode-segmentation::split_sentence_bounds` |     0.04 GB/s |     0.18 GB/s |     0.09 GB/s |     0.05 GB/s |     0.15 GB/s |
| `icu::SentenceSegmenter`                      |     0.20 GB/s |     0.25 GB/s |     0.27 GB/s |     0.20 GB/s |     0.21 GB/s |
|                                               |               |               |               |               |               |
| Python                                        |               |               |               |               |               |
| `stringzilla.utf8_sentences`                  | __0.20 GB/s__ |     0.14 GB/s | __0.23 GB/s__ | __0.18 GB/s__ |     0.21 GB/s |
| `uniseg.sentences`                            |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |
| `icu.BreakIterator`                           |     0.14 GB/s | __0.18 GB/s__ |     0.23 GB/s |     0.14 GB/s | __0.22 GB/s__ |

> Rust measured June 23, 2026; Python June 28, 2026.

### Apple M5 Pro

| Library                                       |       English |       Chinese |        Arabic |        French |        Korean |
| --------------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                                          |               |               |               |               |               |
| `stringzilla::utf8_sentences`                 | __0.44 GB/s__ | __0.55 GB/s__ | __0.51 GB/s__ | __0.41 GB/s__ | __0.56 GB/s__ |
| `unicode-segmentation::split_sentence_bounds` |     0.09 GB/s |     0.34 GB/s |     0.20 GB/s |     0.09 GB/s |     0.36 GB/s |
| `icu::SentenceSegmenter`                      |     0.41 GB/s |     0.55 GB/s |     0.47 GB/s |     0.39 GB/s |     0.44 GB/s |
|                                               |               |               |               |               |               |
| Python                                        |               |               |               |               |               |
| `stringzilla.utf8_sentences`                  | __0.42 GB/s__ |     0.51 GB/s |     0.48 GB/s |     0.39 GB/s |     0.52 GB/s |
| `icu.BreakIterator`                           |     0.41 GB/s | __0.73 GB/s__ | __0.69 GB/s__ | __0.41 GB/s__ | __0.85 GB/s__ |

> Measured July 29, 2026.

## Whitespaces

There are 25 characters classified as whitespace in Unicode.
Six are ASCII separators: horizontal tab _U+0009_, line feed _U+000A_, vertical tab _U+000B_, form feed _U+000C_, carriage return _U+000D_, and space _U+0020_.
Two more are the Next Line control _U+0085_ and the no-break space _U+00A0_.
Then the Ogham space mark _U+1680_, followed by the eleven General Punctuation separators: en quad _U+2000_, em quad _U+2001_, en space _U+2002_, em space _U+2003_, three-per-em space _U+2004_, four-per-em space _U+2005_, six-per-em space _U+2006_, figure space _U+2007_, punctuation space _U+2008_, thin space _U+2009_, and hair space _U+200A_.
The same block adds the line and paragraph separators _U+2028_ and _U+2029_, the narrow no-break space _U+202F_, the medium mathematical space _U+205F_, and the CJK ideographic space _U+3000_.
The zero-width space _U+200B_, despite its name, is not among them — its `White_Space` property is `No`.

### Intel Xeon4 Sapphire Rapids

| Library                               |       English |       Chinese |        Arabic |        French |        Korean |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                                  |               |               |               |               |               |
| `stringzilla::utf8_split_whitespaces` | __0.70 GB/s__ | __1.55 GB/s__ | __1.05 GB/s__ | __0.69 GB/s__ | __0.97 GB/s__ |
| `std::split<is_whitespace>`           |     0.37 GB/s |     0.92 GB/s |     0.47 GB/s |     0.35 GB/s |     0.59 GB/s |
| `icu::WhiteSpace`                     |     0.09 GB/s |     0.28 GB/s |     0.18 GB/s |     0.09 GB/s |     0.23 GB/s |
|                                       |               |               |               |               |               |
| Python                                |               |               |               |               |               |
| `stringzilla.utf8_split_whitespaces`  | __0.11 GB/s__ | __0.24 GB/s__ | __0.17 GB/s__ | __0.11 GB/s__ | __0.16 GB/s__ |
| `regex.split`                         |     0.02 GB/s |     0.03 GB/s |     0.03 GB/s |     0.02 GB/s |     0.02 GB/s |
| `regex.finditer`                      |             — |             — |             — |             — |             — |

> Rust measured June 23, 2026; Python June 28, 2026.
> `str.split()` is omitted: it also treats U+001C–U+001F (C0 separators) as whitespace, a different set than the Unicode `White_Space` property StringZilla, `char::is_whitespace`, and `regex \s` use.

### Apple M5 Pro

| Library                               |       English |       Chinese |        Arabic |        French |        Korean |
| ------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                                  |               |               |               |               |               |
| `stringzilla::utf8_split_whitespaces` |     0.59 GB/s | __1.83 GB/s__ |     0.79 GB/s |     0.58 GB/s |     0.81 GB/s |
| `std::split<is_whitespace>`           | __0.71 GB/s__ |     1.67 GB/s | __1.04 GB/s__ | __0.68 GB/s__ | __1.44 GB/s__ |
| `icu::WhiteSpace`                     |     0.19 GB/s |     0.58 GB/s |     0.43 GB/s |     0.19 GB/s |     0.52 GB/s |
|                                       |               |               |               |               |               |
| Python                                |               |               |               |               |               |
| `stringzilla.utf8_split_whitespaces`  | __0.27 GB/s__ | __1.33 GB/s__ | __0.41 GB/s__ | __0.27 GB/s__ | __0.39 GB/s__ |
| `regex.finditer`                      |     0.06 GB/s |     0.43 GB/s |     0.10 GB/s |     0.06 GB/s |     0.10 GB/s |

> Measured July 29, 2026.

## TR29 Words

UAX#29 word boundaries use 18 Word_Break classes (ALetter, Hebrew_Letter, Numeric, Katakana, ExtendNumLet, MidLetter, MidNum, MidNumLet, Single_Quote, Double_Quote, WSegSpace, Format, Extend, ZWJ, Regional_Indicator, CR, LF, Newline).
Letter and number runs stay together while punctuation and spaces split.
Space-less scripts (Chinese, Japanese, Thai) need a dictionary, hence ICU's `WordSegmenter::new_dictionary`.
`stringzilla::utf8_wordbreaks` emits every boundary segment (tiling the input contiguously), so it is compared only against the baselines that also tile — `split_word_bounds` and ICU — not the word-like-only `unicode_words` filter.

### Intel Xeon4 Sapphire Rapids

| Library                                   |       English |       Chinese |        Arabic |        French |        Korean |
| ----------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                                      |               |               |               |               |               |
| `stringzilla::utf8_wordbreaks`            | __0.35 GB/s__ | __0.22 GB/s__ | __0.35 GB/s__ | __0.28 GB/s__ | __0.31 GB/s__ |
| `unicode-segmentation::split_word_bounds` |     0.06 GB/s |     0.13 GB/s |     0.12 GB/s |     0.06 GB/s |     0.17 GB/s |
| `icu::WordSegmenter`                      |     0.11 GB/s |     0.02 GB/s |     0.22 GB/s |     0.11 GB/s |     0.17 GB/s |
| `std::split_whitespace`                   |             — |             — |             — |             — |             — |
|                                           |               |               |               |               |               |
| Python                                    |               |               |               |               |               |
| `stringzilla.utf8_wordbreaks`             | __0.06 GB/s__ | __0.06 GB/s__ | __0.09 GB/s__ | __0.06 GB/s__ | __0.09 GB/s__ |
| `uniseg.words`                            |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |
| `icu.BreakIterator`                       |     0.03 GB/s |     0.01 GB/s |     0.05 GB/s |     0.03 GB/s |     0.03 GB/s |

> Rust measured June 23, 2026; Python June 28, 2026.

### Apple M5 Pro

| Library                                   |       English |       Chinese |        Arabic |        French |        Korean |
| ----------------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                                      |               |               |               |               |               |
| `stringzilla::utf8_wordbreaks`            |     0.46 GB/s |     0.34 GB/s |     0.35 GB/s |     0.35 GB/s |     0.35 GB/s |
| `unicode-segmentation::split_word_bounds` |     0.12 GB/s |     0.25 GB/s |     0.24 GB/s |     0.12 GB/s |     0.42 GB/s |
| `icu::WordSegmenter`                      |     0.21 GB/s |     0.04 GB/s |     0.33 GB/s |     0.22 GB/s |     0.42 GB/s |
| `std::split_whitespace`                   | __0.68 GB/s__ | __1.65 GB/s__ | __1.02 GB/s__ | __0.65 GB/s__ | __1.35 GB/s__ |
|                                           |               |               |               |               |               |
| Python                                    |               |               |               |               |               |
| `stringzilla.utf8_wordbreaks`             | __0.13 GB/s__ | __0.12 GB/s__ | __0.17 GB/s__ | __0.13 GB/s__ | __0.16 GB/s__ |
| `icu.BreakIterator`                       |     0.08 GB/s |     0.03 GB/s |     0.14 GB/s |     0.09 GB/s |     0.08 GB/s |

> Measured July 29, 2026.

## TR29 Graphemes

A grapheme cluster is one user-perceived character: a base plus combining marks, a Hangul L/V/T syllable, a Regional_Indicator pair (flag), or an Extended_Pictographic base joined by ZWJ _U+200D_.
UAX#29 assigns 13 Grapheme_Cluster_Break classes (CR, LF, Control, Extend, ZWJ, Regional_Indicator, Prepend, SpacingMark, L, V, T, LV, LVT).
It breaks between every pair except where rules GB3–GB11 forbid it.

### Intel Xeon4 Sapphire Rapids

| Library                           |       English |       Chinese |        Arabic |        French |        Korean |
| --------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                              |               |               |               |               |               |
| `stringzilla::utf8_graphemes`     | __0.23 GB/s__ | __0.21 GB/s__ | __0.26 GB/s__ | __0.19 GB/s__ | __0.26 GB/s__ |
| `unicode-segmentation::graphemes` |     0.07 GB/s |     0.13 GB/s |     0.10 GB/s |     0.07 GB/s |     0.08 GB/s |
| `icu::GraphemeClusterSegmenter`   |     0.13 GB/s |     0.20 GB/s |     0.19 GB/s |     0.13 GB/s |     0.17 GB/s |
|                                   |               |               |               |               |               |
| Python                            |               |               |               |               |               |
| `stringzilla.utf8_graphemes`      | __0.03 GB/s__ | __0.05 GB/s__ | __0.04 GB/s__ | __0.03 GB/s__ | __0.05 GB/s__ |
| `regex.finditer`                  |    <0.01 GB/s |     0.01 GB/s |     0.01 GB/s |    <0.01 GB/s |     0.01 GB/s |
| `grapheme.graphemes`              |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |
| `uniseg.grapheme_clusters`        |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |
| `icu.BreakIterator`               |     0.02 GB/s |     0.04 GB/s |     0.03 GB/s |     0.02 GB/s |     0.04 GB/s |

> Rust measured June 23, 2026; Python June 28, 2026.

### Apple M5 Pro

| Library                           |       English |       Chinese |        Arabic |        French |        Korean |
| --------------------------------- | ------------: | ------------: | ------------: | ------------: | ------------: |
| Rust                              |               |               |               |               |               |
| `stringzilla::utf8_graphemes`     |     0.22 GB/s |     0.41 GB/s |     0.29 GB/s |     0.22 GB/s |     0.29 GB/s |
| `unicode-segmentation::graphemes` |     0.15 GB/s |     0.23 GB/s |     0.22 GB/s |     0.15 GB/s |     0.17 GB/s |
| `icu::GraphemeClusterSegmenter`   | __0.29 GB/s__ | __0.46 GB/s__ | __0.36 GB/s__ | __0.29 GB/s__ | __0.36 GB/s__ |
|                                   |               |               |               |               |               |
| Python                            |               |               |               |               |               |
| `stringzilla.utf8_graphemes`      | __0.05 GB/s__ | __0.12 GB/s__ | __0.08 GB/s__ | __0.05 GB/s__ |     0.10 GB/s |
| `regex.finditer`                  |     0.01 GB/s |     0.03 GB/s |     0.02 GB/s |     0.01 GB/s |     0.03 GB/s |
| `grapheme.graphemes`              |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |    <0.01 GB/s |
| `icu.BreakIterator`               |     0.04 GB/s |     0.11 GB/s |     0.08 GB/s |     0.05 GB/s | __0.11 GB/s__ |

> Measured July 29, 2026.

## Codepoint Indexing

UTF-8 encodes each codepoint in 1–4 bytes (lead bytes `0xxxxxxx`, `110xxxxx`, `1110xxxx`, `11110xxx`).
`find_nth_utf8` returns the byte offset of the Nth codepoint by scanning lead bytes; targeting the last codepoint forces a full-buffer scan.
CPython stores `str` in a fixed-width representation (PEP 393), so `s[n]` and `len(s)` are O(1) — there is no comparable Python byte scan, so this section is Rust-only.

### Intel Xeon4 Sapphire Rapids

| Library                      |        English |        Chinese |         Arabic |         French |         Korean |
| ---------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
| Rust                         |                |                |                |                |                |
| `stringzilla::find_nth_utf8` | __16.12 GB/s__ | __18.10 GB/s__ | __13.94 GB/s__ | __17.25 GB/s__ | __15.20 GB/s__ |
| `std::char_indices.nth`      |      1.61 GB/s |      1.07 GB/s |      0.73 GB/s |      1.12 GB/s |      0.71 GB/s |

> Measured June 23, 2026.

### Apple M5 Pro

| Library                      |        English |        Chinese |         Arabic |         French |         Korean |
| ---------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
| Rust                         |                |                |                |                |                |
| `stringzilla::find_nth_utf8` | __34.14 GB/s__ | __34.27 GB/s__ | __34.28 GB/s__ | __34.26 GB/s__ | __34.27 GB/s__ |
| `std::char_indices.nth`      |      2.67 GB/s |      1.12 GB/s |      0.66 GB/s |      1.79 GB/s |      0.96 GB/s |

> Measured July 29, 2026.

To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_tokenization --features bench_tokenization
bin=$(find target/release/deps -name 'bench_tokenization-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="tokenize" "$bin"
done
```

---

See [README.md](../README.md) for dataset information and replication instructions.
