#![doc = r#"# StringWars: Tokenization

Tokenization benchmarks: UTF-8 iteration, word/grapheme/sentence/line segmentation.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_tokenization --bench bench_tokenization
```
"#]
use std::hint::black_box;

use stringtape::BytesCowsAuto;

use icu::properties::props::WhiteSpace;
use icu::properties::CodePointSetData;
use icu::segmenter::{GraphemeClusterSegmenter, LineSegmenter, SentenceSegmenter, WordSegmenter};
use stringzilla::sz;
use stringzilla::sz::StringZillableUnary;
use unicode_linebreak::linebreaks;
use unicode_segmentation::UnicodeSegmentation;

use stringwars::{
    finish, install_panic_hook, log_stringzilla_metadata, log_timing_overhead, measure,
    resolve_dataset, MeasureSpec, ResultExt, Unit, WorkUnits,
};

/// File-local helper: cycles through `needles` byte slices, passes each to `count`, and reports
/// throughput as `WorkUnits::bytes(line.len())` — the bytes of that one line per call.
///
/// Only use this for benchmarks whose body is exactly "pick a line, count something,
/// WorkUnits::bytes(line.len())". Blocks that operate on `&str` slices, use `unsafe`, or scan the
/// whole haystack in one shot are left inline.
fn measure_line_tokenizer<Count: FnMut(&[u8]) -> usize>(
    name: &str,
    work: WorkUnits,
    needles: &BytesCowsAuto,
    mut count: Count,
) {
    measure(name, MeasureSpec::new(Unit::Bytes, work), || {
        for line in needles.iter() {
            black_box(count(black_box(line)));
        }
    });
}

/// Benchmarks Unicode whitespace splitting using ICU, stdlib, and StringZilla.
///
/// Each call splits a single document line, cycling through the line tokens. Throughput is
/// reported as the bytes of that one line, so the per-byte rate still reflects the splitter's
/// compute cost while the working set stays a single line rather than the whole file.
fn bench_tokenize_whitespace(work: WorkUnits, needles: &BytesCowsAuto, lines_str: &[&str]) {
    measure_line_tokenizer(
        "tokenize-whitespace/stringzilla::utf8_split_whitespaces",
        work,
        needles,
        |line| {
            let count: usize = line.sz_utf8_split_whitespaces().count();
            black_box(count);
            count
        },
    );

    {
        measure(
            "tokenize-whitespace/std::split<is_whitespace>",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let count: usize = black_box(*line)
                        .split(char::is_whitespace)
                        .filter(|segment| !segment.is_empty())
                        .count();
                    black_box(count);
                }
            },
        );
    }

    {
        let white_space = CodePointSetData::new::<WhiteSpace>();
        measure(
            "tokenize-whitespace/icu::WhiteSpace.split",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let count: usize = black_box(*line)
                        .split(|character: char| white_space.contains(character))
                        .filter(|segment: &&str| !segment.is_empty())
                        .count();
                    black_box(count);
                }
            },
        );
    }
}

/// Benchmarks Unicode newline splitting using custom predicates and StringZilla.
///
/// Each call splits a single document line, cycling through the line tokens; throughput is the
/// bytes of that one line. (Per-line newline splitting is degenerate when lines were split on `\n`,
/// but the kernels still exercise the full Unicode newline set across the seven characters.)
fn bench_tokenize_newlines(work: WorkUnits, needles: &BytesCowsAuto, lines_str: &[&str]) {
    // Custom newline predicate matching StringZilla's 7 newline characters.
    fn is_unicode_newline(character: char) -> bool {
        matches!(
            character,
            '\n' | '\r' | '\x0B' | '\x0C' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
    }

    measure_line_tokenizer(
        "tokenize-newlines/stringzilla::utf8_split_newlines",
        work,
        needles,
        |line| {
            let count: usize = line.sz_utf8_split_newlines().count();
            black_box(count);
            count
        },
    );

    {
        measure(
            "tokenize-newlines/custom::split<is_unicode_newline>",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let count: usize = black_box(*line)
                        .split(is_unicode_newline)
                        .filter(|segment| !segment.is_empty())
                        .count();
                    black_box(count);
                }
            },
        );
    }
}

/// Benchmarks Unicode TR29 (UAX#29) word segmentation.
///
/// TR29 defines linguistically-aware word boundaries that handle complex cases like
/// contractions ("can't"), numeric sequences ("3.14"), and scripts without spaces.
/// `stringzilla::utf8_wordbreaks` yields UAX#29 words that tile the input contiguously (every
/// segment, punctuation and spaces included), so the apples-to-apples baselines are the ones that
/// also tile — `unicode-segmentation::split_word_bounds()` and `icu::segmenter::WordSegmenter`. The
/// filtering `unicode_words()` (word-like segments only, dropping spaces/punctuation) is a different
/// operation and is intentionally not compared here.
fn bench_tokenize_words_tr29(work: WorkUnits, needles: &BytesCowsAuto, lines_str: &[&str]) {
    // Benchmark for StringZilla's single-pass TR29 word iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-words-tr29/stringzilla::utf8_wordbreaks",
        work,
        needles,
        |line| {
            let count: usize = line.sz_utf8_wordbreaks().count();
            black_box(count);
            count
        },
    );

    {
        measure(
            "tokenize-words-tr29/unicode-segmentation::split_word_bounds",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    let count: usize = line.split_word_bounds().count();
                    black_box(count);
                }
            },
        );
    }

    {
        let segmenter = WordSegmenter::new_dictionary(Default::default());
        measure(
            "tokenize-words-tr29/icu::WordSegmenter::new_dictionary.segment_str",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    // WordSegmenter returns boundary indices; count segments = boundaries - 1
                    let boundaries: usize = segmenter.segment_str(line).count();
                    black_box(boundaries);
                }
            },
        );
    }

    {
        measure(
            "tokenize-words-tr29/std::split_whitespace",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    let count: usize = line.split_whitespace().count();
                    black_box(count);
                }
            },
        );
    }
}

/// Benchmarks Unicode TR29 (UAX#29) grapheme cluster segmentation.
///
/// Grapheme clusters are user-perceived characters: a base codepoint plus any combining marks,
/// emoji ZWJ sequences, and regional-indicator pairs count as one cluster.
/// - `unicode-segmentation::graphemes(true)`: extended grapheme clusters
/// - `icu::segmenter::GraphemeClusterSegmenter`: ICU4X implementation
fn bench_tokenize_graphemes(work: WorkUnits, needles: &BytesCowsAuto, lines_str: &[&str]) {
    // Benchmark for StringZilla's single-pass grapheme iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-graphemes-tr29/stringzilla::utf8_graphemes",
        work,
        needles,
        |line| {
            let count: usize = line.sz_utf8_graphemes().count();
            black_box(count);
            count
        },
    );

    {
        measure(
            "tokenize-graphemes-tr29/unicode-segmentation::graphemes",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    let count: usize = UnicodeSegmentation::graphemes(line, true).count();
                    black_box(count);
                }
            },
        );
    }

    {
        let segmenter = GraphemeClusterSegmenter::new();
        measure(
            "tokenize-graphemes-tr29/icu::GraphemeClusterSegmenter.segment_str",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    // The segmenter returns boundary indices; count segments = boundaries - 1.
                    let boundaries: usize = segmenter.segment_str(line).count();
                    black_box(boundaries);
                }
            },
        );
    }
}

/// Benchmarks Unicode TR29 (UAX#29) sentence segmentation.
///
/// Sentence boundaries handle abbreviations, decimal numbers, and terminal punctuation across
/// scripts.
/// - `unicode-segmentation::split_sentence_bounds()`: raw UAX#29 sentence boundaries
/// - `icu::segmenter::SentenceSegmenter`: ICU4X implementation
fn bench_tokenize_sentences(work: WorkUnits, needles: &BytesCowsAuto, lines_str: &[&str]) {
    // Benchmark for StringZilla's single-pass sentence iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-sentences-tr29/stringzilla::utf8_sentences",
        work,
        needles,
        |line| {
            let count: usize = line.sz_utf8_sentences().count();
            black_box(count);
            count
        },
    );

    {
        measure(
            "tokenize-sentences-tr29/unicode-segmentation::split_sentence_bounds",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    let count: usize = line.split_sentence_bounds().count();
                    black_box(count);
                }
            },
        );
    }

    {
        let segmenter = SentenceSegmenter::new(Default::default());
        measure(
            "tokenize-sentences-tr29/icu::SentenceSegmenter.segment_str",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    // The segmenter returns boundary indices; count segments = boundaries - 1.
                    let boundaries: usize = segmenter.segment_str(line).count();
                    black_box(boundaries);
                }
            },
        );
    }
}

/// Benchmarks Unicode UAX#14 line-break opportunity segmentation.
///
/// UAX#14 locates positions where a renderer may wrap a line (after spaces, hyphens, etc.),
/// distinct from the hard newline splitting in `bench_tokenize_newlines`.
/// - `unicode-linebreak::linebreaks()`: mandatory and allowed break opportunities
/// - `icu::segmenter::LineSegmenter`: ICU4X implementation
fn bench_tokenize_lines_uax14(work: WorkUnits, needles: &BytesCowsAuto, lines_str: &[&str]) {
    // Benchmark for StringZilla's single-pass line-break iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-lines-uax14/stringzilla::utf8_linebreaks",
        work,
        needles,
        |line| {
            let count: usize = line.sz_utf8_linebreaks().count();
            black_box(count);
            count
        },
    );

    {
        measure(
            "tokenize-lines-uax14/unicode-linebreak::linebreaks",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    let count: usize = linebreaks(line).count();
                    black_box(count);
                }
            },
        );
    }

    {
        let segmenter = LineSegmenter::new_dictionary(Default::default());
        measure(
            "tokenize-lines-uax14/icu::LineSegmenter::new_dictionary.segment_str",
            MeasureSpec::new(Unit::Bytes, work),
            || {
                for line in lines_str {
                    let line = black_box(*line);
                    // The segmenter returns boundary indices; count segments = boundaries - 1.
                    let boundaries: usize = segmenter.segment_str(line).count();
                    black_box(boundaries);
                }
            },
        );
    }
}

/// Benchmarks UTF-8 character counting using StringZilla, simdutf, and stdlib.
fn bench_utf8_length(_work: WorkUnits, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    // Validate UTF-8 once, outside the timed closures (only the stdlib baseline needs it; the
    // StringZilla and simdutf counters operate directly on bytes).
    let haystack_str = std::str::from_utf8(haystack).ok();

    measure(
        "utf8-length/stringzilla::utf8_chars.len",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = haystack_bytes.sz_utf8_runes().len();
            black_box(count);
        },
    );

    // Benchmark for StringZilla's dedicated `count_utf8()` free function (direct SIMD scan,
    // without constructing a view object).
    measure(
        "utf8-length/stringzilla::count_utf8",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = sz::count_utf8(haystack_bytes);
            black_box(count);
        },
    );

    measure(
        "utf8-length/simdutf::count_utf8",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = simdutf::count_utf8(haystack_bytes);
            black_box(count);
        },
    );

    {
        let text = haystack_str.expect("UTF-8 text required for the stdlib codepoint counter");
        measure(
            "utf8-length/std::chars.count",
            MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
            || {
                let count: usize = black_box(text).chars().count();
                black_box(count);
            },
        );
    }
}

/// Benchmarks UTF-8 to UTF-32 decoding using StringZilla, simdutf, and stdlib.
fn bench_utf8_iterate(_work: WorkUnits, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    measure(
        "utf8-iterate/stringzilla::utf8_chars.iter",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let haystack_bytes = black_box(haystack);
            let mut sum: u32 = 0;
            for character in haystack_bytes.sz_utf8_runes().iter() {
                sum = sum.wrapping_add(character as u32);
            }
            black_box(sum);
        },
    );

    {
        // Pre-allocate buffer for UTF-32 output (worst case: same number of codepoints as bytes)
        let mut utf32_buffer = vec![0u32; haystack.len()];
        measure(
            "utf8-iterate/simdutf::convert_utf8_to_utf32",
            MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
            || {
                let haystack_bytes = black_box(haystack);
                let len = unsafe {
                    simdutf::convert_utf8_to_utf32(
                        haystack_bytes.as_ptr(),
                        haystack_bytes.len(),
                        utf32_buffer.as_mut_ptr(),
                    )
                };
                let mut sum: u32 = 0;
                for value in &utf32_buffer[..len] {
                    sum = sum.wrapping_add(*value);
                }
                black_box(sum);
            },
        );
    }

    measure(
        "utf8-iterate/std::chars",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            // Safety: the tokenization corpora are valid UTF-8 text; `from_utf8_unchecked` skips
            // re-validation on the hot path so the benchmark measures iteration, not UTF-8 checking.
            let haystack_str = black_box(unsafe { std::str::from_utf8_unchecked(haystack) });
            let mut sum: u32 = 0;
            for character in haystack_str.chars() {
                sum = sum.wrapping_add(character as u32);
            }
            black_box(sum);
        },
    );
}

/// Benchmarks locating the byte offset of the Nth UTF-8 codepoint.
///
/// - `stringzilla::find_nth_utf8()`: SIMD-accelerated scan to the Nth codepoint
/// - `std::str::char_indices().nth()`: scalar decode-and-count baseline
///
/// We target the *last* codepoint, so every implementation scans the whole buffer once —
/// a fair workload whose throughput is simply the input size.
fn bench_find_nth_utf8(_work: WorkUnits, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_str = match std::str::from_utf8(haystack) {
        Ok(text) => text,
        Err(_) => {
            eprintln!("Warning: Haystack is not valid UTF-8, skipping find-nth-utf8 benchmarks");
            return;
        }
    };

    let codepoint_count = sz::count_utf8(haystack);
    if codepoint_count == 0 {
        return;
    }
    let last_index = codepoint_count - 1;

    let haystack_length = haystack.len() as u64;

    measure(
        "find-nth-utf8/stringzilla::find_nth_utf8",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let offset = sz::find_nth_utf8(black_box(haystack), last_index);
            black_box(offset);
        },
    );

    measure(
        "find-nth-utf8/std::char_indices.nth",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let offset = black_box(haystack_str)
                .char_indices()
                .nth(last_index)
                .map(|(byte_offset, _)| byte_offset);
            black_box(offset);
        },
    );
}
fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    let tape = resolve_dataset("tokenization").unwrap_nice();

    // The tape's single token, not `parent()`: the parent is the raw read, cut at exactly
    // the budget and so liable to end mid-character, while the token carries the UTF-8
    // backoff. It is also what the Python side measures, so the two stay comparable.
    let haystack: &[u8] = tape.iter().next().expect("empty working set");
    let needles = &tape;
    let work = WorkUnits::new(tape.len() as u64, tape.iter().map(|t| t.len() as u64).sum());
    // Decoded once for every `&str` baseline in the suite. A lossy fallback would be
    // silent: in `file` mode the tape is a single token, so one undecodable byte would
    // hand a no-op row the whole working set and call it throughput.
    let lines_str: Vec<&str> = tape
        .iter()
        .map(|line| std::str::from_utf8(line).expect("dataset must be valid UTF-8"))
        .collect();
    log_timing_overhead();

    println!("# tokenize-whitespace");
    bench_tokenize_whitespace(work, needles, &lines_str);

    println!("# tokenize-newlines");
    bench_tokenize_newlines(work, needles, &lines_str);

    println!("# tokenize-words-tr29");
    bench_tokenize_words_tr29(work, needles, &lines_str);

    println!("# tokenize-graphemes-tr29");
    bench_tokenize_graphemes(work, needles, &lines_str);

    println!("# tokenize-sentences-tr29");
    bench_tokenize_sentences(work, needles, &lines_str);

    println!("# tokenize-lines-uax14");
    bench_tokenize_lines_uax14(work, needles, &lines_str);

    println!("# utf8-length");
    bench_utf8_length(work, haystack, needles);

    println!("# utf8-iterate");
    bench_utf8_iterate(work, haystack, needles);

    println!("# find-nth-utf8");
    bench_find_nth_utf8(work, haystack, needles);

    finish();
}
