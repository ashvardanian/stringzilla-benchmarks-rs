#![doc = r#"
# StringWars: UTF-8 Tokenization & Iteration Benchmarks

This file benchmarks UTF-8 segmentation and codepoint iteration:
- Unicode whitespace and newline splitting
- Unicode TR29 (UAX#29) word segmentation
- UTF-8 character counting and UTF-8 to UTF-32 decoding
- Locating the byte offset of the Nth UTF-8 codepoint

## Benchmark Groups

- `tokenize-whitespace`: Unicode whitespace splitting
- `tokenize-newlines`: Unicode newline splitting
- `tokenize-words-tr29`: Unicode TR29 word boundary segmentation
- `tokenize-graphemes-tr29`: Unicode TR29 grapheme cluster segmentation
- `tokenize-sentences-tr29`: Unicode TR29 sentence boundary segmentation
- `tokenize-lines-uax14`: Unicode UAX#14 line-break opportunity segmentation
- `utf8-length`: UTF-8 character counting
- `utf8-iterate`: UTF-8 to UTF-32 decoding
- `find-nth-utf8`: byte offset of the Nth UTF-8 codepoint

## Usage Examples

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_tokenization --bench bench_tokenization
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

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput, should_run,
    BenchBudget, ReportAs, ResultExt, WorkUnits, COMPUTE_BOUND_SLICE,
};

/// File-local helper: cycles through `needles` byte slices, passes each to `count`, and reports
/// throughput as `WorkUnits::bytes(line.len())` — the bytes of that one line per call.
///
/// Only use this for benchmarks whose body is exactly "pick a line, count something,
/// WorkUnits::bytes(line.len())". Blocks that operate on `&str` slices, use `unsafe`, or scan the
/// whole haystack in one shot are left inline.
fn measure_line_tokenizer<Count: FnMut(&[u8]) -> usize>(
    name: &str,
    budget: &BenchBudget,
    needles: &BytesCowsAuto,
    mut count: Count,
) {
    if !should_run(name) {
        return;
    }
    let mut lines = needles.iter().cycle();
    measure_throughput(name, ReportAs::Bytes, budget, || {
        let line = black_box(lines.next().unwrap());
        black_box(count(line));
        WorkUnits::bytes(line.len() as u64)
    });
}

/// Benchmarks Unicode whitespace splitting using ICU, stdlib, and StringZilla.
///
/// Each call splits a single document line, cycling through the line tokens. Throughput is
/// reported as the bytes of that one line, so the per-byte rate still reflects the splitter's
/// compute cost while the working set stays a single line rather than the whole file.
fn bench_tokenize_whitespace(budget: &BenchBudget, _haystack: &[u8], needles: &BytesCowsAuto) {
    // Pre-decode every line to `&str` once, outside the timed closures. The byte-based StringZilla
    // variant operates on the raw line bytes; the `str`-based baselines reuse these validated lines.
    let lines_str: Vec<&str> = needles
        .iter()
        .map(|line| std::str::from_utf8(line).unwrap_or(""))
        .collect();

    // Benchmark for StringZilla whitespace splits.
    measure_line_tokenizer(
        "tokenize-whitespace/stringzilla::utf8_split_whitespaces",
        budget,
        needles,
        |line| {
            let count: usize = line.sz_utf8_split_whitespaces().count();
            black_box(count);
            count
        },
    );

    // Benchmark for Rust stdlib char::is_whitespace.
    {
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-whitespace/std::split<is_whitespace>",
            ReportAs::Bytes,
            budget,
            || {
                let line = lines.next().unwrap();
                let count: usize = black_box(*line)
                    .split(char::is_whitespace)
                    .filter(|segment| !segment.is_empty())
                    .count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }

    // Benchmark for ICU4X WhiteSpace property.
    {
        let white_space = CodePointSetData::new::<WhiteSpace>();
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-whitespace/icu::WhiteSpace.split",
            ReportAs::Bytes,
            budget,
            || {
                let line = lines.next().unwrap();
                let count: usize = black_box(*line)
                    .split(|character: char| white_space.contains(character))
                    .filter(|segment: &&str| !segment.is_empty())
                    .count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }
}

/// Benchmarks Unicode newline splitting using custom predicates and StringZilla.
///
/// Each call splits a single document line, cycling through the line tokens; throughput is the
/// bytes of that one line. (Per-line newline splitting is degenerate when lines were split on `\n`,
/// but the kernels still exercise the full Unicode newline set across the seven characters.)
fn bench_tokenize_newlines(budget: &BenchBudget, _haystack: &[u8], needles: &BytesCowsAuto) {
    // Custom newline predicate matching StringZilla's 7 newline characters.
    fn is_unicode_newline(character: char) -> bool {
        matches!(
            character,
            '\n' | '\r' | '\x0B' | '\x0C' | '\u{0085}' | '\u{2028}' | '\u{2029}'
        )
    }

    // Pre-decode every line to `&str` once (only the custom `str` baseline needs it); the
    // StringZilla variant splits the raw line bytes directly.
    let lines_str: Vec<&str> = needles
        .iter()
        .map(|line| std::str::from_utf8(line).unwrap_or(""))
        .collect();

    // Benchmark for StringZilla newline splits.
    measure_line_tokenizer(
        "tokenize-newlines/stringzilla::utf8_split_newlines",
        budget,
        needles,
        |line| {
            let count: usize = line.sz_utf8_split_newlines().count();
            black_box(count);
            count
        },
    );

    // Benchmark for custom newline predicate.
    {
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-newlines/custom::split<is_unicode_newline>",
            ReportAs::Bytes,
            budget,
            || {
                let line = lines.next().unwrap();
                let count: usize = black_box(*line)
                    .split(is_unicode_newline)
                    .filter(|segment| !segment.is_empty())
                    .count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
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
fn bench_tokenize_words_tr29(budget: &BenchBudget, _haystack: &[u8], needles: &BytesCowsAuto) {
    // Pre-decode every line to `&str` once. StringZilla segments the raw line bytes directly; the
    // `unicode-segmentation`, ICU, and stdlib baselines reuse these validated lines per call.
    let lines_str: Vec<&str> = needles
        .iter()
        .map(|line| std::str::from_utf8(line).unwrap_or(""))
        .collect();

    // Benchmark for StringZilla's single-pass TR29 word iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-words-tr29/stringzilla::utf8_wordbreaks",
        budget,
        needles,
        |line| {
            let count: usize = line.sz_utf8_wordbreaks().count();
            black_box(count);
            count
        },
    );

    // Benchmark for unicode-segmentation: split_word_bounds() - all segments
    {
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-words-tr29/unicode-segmentation::split_word_bounds",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                let count: usize = line.split_word_bounds().count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }

    // Benchmark for ICU4X WordSegmenter with dictionary model
    {
        let segmenter = WordSegmenter::new_dictionary(Default::default());
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-words-tr29/icu::WordSegmenter::new_dictionary.segment_str",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                // WordSegmenter returns boundary indices; count segments = boundaries - 1
                let boundaries: usize = segmenter.segment_str(line).count();
                black_box(boundaries);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }

    // Benchmark for stdlib split_whitespace as baseline comparison
    {
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-words-tr29/std::split_whitespace",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                let count: usize = line.split_whitespace().count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
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
fn bench_tokenize_graphemes(budget: &BenchBudget, _haystack: &[u8], needles: &BytesCowsAuto) {
    // Pre-decode every line to `&str` once. StringZilla segments the raw line bytes directly; the
    // `unicode-segmentation` and ICU baselines reuse these validated lines per call.
    let lines_str: Vec<&str> = needles
        .iter()
        .map(|line| std::str::from_utf8(line).unwrap_or(""))
        .collect();

    // Benchmark for StringZilla's single-pass grapheme iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-graphemes-tr29/stringzilla::utf8_graphemes",
        budget,
        needles,
        |line| {
            let count: usize = line.sz_utf8_graphemes().count();
            black_box(count);
            count
        },
    );

    // Benchmark for unicode-segmentation: graphemes(true) - extended grapheme clusters
    {
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-graphemes-tr29/unicode-segmentation::graphemes",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                let count: usize = UnicodeSegmentation::graphemes(line, true).count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }

    // Benchmark for ICU4X GraphemeClusterSegmenter
    {
        let segmenter = GraphemeClusterSegmenter::new();
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-graphemes-tr29/icu::GraphemeClusterSegmenter.segment_str",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                // The segmenter returns boundary indices; count segments = boundaries - 1.
                let boundaries: usize = segmenter.segment_str(line).count();
                black_box(boundaries);
                WorkUnits::bytes(line.len() as u64)
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
fn bench_tokenize_sentences(budget: &BenchBudget, _haystack: &[u8], needles: &BytesCowsAuto) {
    // Pre-decode every line to `&str` once. StringZilla segments the raw line bytes directly; the
    // `unicode-segmentation` and ICU baselines reuse these validated lines per call.
    let lines_str: Vec<&str> = needles
        .iter()
        .map(|line| std::str::from_utf8(line).unwrap_or(""))
        .collect();

    // Benchmark for StringZilla's single-pass sentence iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-sentences-tr29/stringzilla::utf8_sentences",
        budget,
        needles,
        |line| {
            let count: usize = line.sz_utf8_sentences().count();
            black_box(count);
            count
        },
    );

    // Benchmark for unicode-segmentation: split_sentence_bounds() - raw UAX#29 boundaries
    {
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-sentences-tr29/unicode-segmentation::split_sentence_bounds",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                let count: usize = line.split_sentence_bounds().count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }

    // Benchmark for ICU4X SentenceSegmenter
    {
        let segmenter = SentenceSegmenter::new(Default::default());
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-sentences-tr29/icu::SentenceSegmenter.segment_str",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                // The segmenter returns boundary indices; count segments = boundaries - 1.
                let boundaries: usize = segmenter.segment_str(line).count();
                black_box(boundaries);
                WorkUnits::bytes(line.len() as u64)
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
fn bench_tokenize_lines_uax14(budget: &BenchBudget, _haystack: &[u8], needles: &BytesCowsAuto) {
    // Pre-decode every line to `&str` once. StringZilla segments the raw line bytes directly; the
    // `unicode-linebreak` and ICU baselines reuse these validated lines per call.
    let lines_str: Vec<&str> = needles
        .iter()
        .map(|line| std::str::from_utf8(line).unwrap_or(""))
        .collect();

    // Benchmark for StringZilla's single-pass line-break iterator. `.count()` consumes the
    // iterator without materializing the segments, so no allocation taints the measurement.
    measure_line_tokenizer(
        "tokenize-lines-uax14/stringzilla::utf8_linebreaks",
        budget,
        needles,
        |line| {
            let count: usize = line.sz_utf8_linebreaks().count();
            black_box(count);
            count
        },
    );

    // Benchmark for unicode-linebreak: linebreaks() - all break opportunities
    {
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-lines-uax14/unicode-linebreak::linebreaks",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                let count: usize = linebreaks(line).count();
                black_box(count);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }

    // Benchmark for ICU4X LineSegmenter
    {
        let segmenter = LineSegmenter::new_dictionary(Default::default());
        let mut lines = lines_str.iter().cycle();
        measure_throughput(
            "tokenize-lines-uax14/icu::LineSegmenter::new_dictionary.segment_str",
            ReportAs::Bytes,
            budget,
            || {
                let line = black_box(*lines.next().unwrap());
                // The segmenter returns boundary indices; count segments = boundaries - 1.
                let boundaries: usize = segmenter.segment_str(line).count();
                black_box(boundaries);
                WorkUnits::bytes(line.len() as u64)
            },
        );
    }
}

/// Benchmarks UTF-8 character counting using StringZilla, simdutf, and stdlib.
fn bench_utf8_length(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    // Validate UTF-8 once, outside the timed closures (only the stdlib baseline needs it; the
    // StringZilla and simdutf counters operate directly on bytes).
    let haystack_str = std::str::from_utf8(haystack).ok();

    // Benchmark for StringZilla UTF-8 character counting via the lazy view.
    measure_throughput(
        "utf8-length/stringzilla::utf8_chars.len",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = haystack_bytes.sz_utf8_runes().len();
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for StringZilla's dedicated `count_utf8()` free function (direct SIMD scan,
    // without constructing a view object).
    measure_throughput(
        "utf8-length/stringzilla::count_utf8",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = sz::count_utf8(haystack_bytes);
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for simdutf UTF-8 character counting.
    measure_throughput(
        "utf8-length/simdutf::count_utf8",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let count: usize = simdutf::count_utf8(haystack_bytes);
            black_box(count);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for stdlib UTF-8 character counting.
    {
        let text = haystack_str.expect("UTF-8 text required for the stdlib codepoint counter");
        measure_throughput(
            "utf8-length/std::chars.count",
            ReportAs::Bytes,
            budget,
            || {
                let count: usize = black_box(text).chars().count();
                black_box(count);
                WorkUnits::bytes(haystack_length)
            },
        );
    }
}

/// Benchmarks UTF-8 to UTF-32 decoding using StringZilla, simdutf, and stdlib.
fn bench_utf8_iterate(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;

    // Benchmark for StringZilla UTF-8 character iteration.
    measure_throughput(
        "utf8-iterate/stringzilla::utf8_chars.iter",
        ReportAs::Bytes,
        budget,
        || {
            let haystack_bytes = black_box(haystack);
            let mut sum: u32 = 0;
            for character in haystack_bytes.sz_utf8_runes().iter() {
                sum = sum.wrapping_add(character as u32);
            }
            black_box(sum);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for simdutf UTF-8 to UTF-32 conversion.
    {
        // Pre-allocate buffer for UTF-32 output (worst case: same number of codepoints as bytes)
        let mut utf32_buffer = vec![0u32; haystack.len()];
        measure_throughput(
            "utf8-iterate/simdutf::convert_utf8_to_utf32",
            ReportAs::Bytes,
            budget,
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
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // Benchmark for stdlib UTF-8 character iteration.
    measure_throughput("utf8-iterate/std::chars", ReportAs::Bytes, budget, || {
        // Safety: the tokenization corpora are valid UTF-8 text; `from_utf8_unchecked` skips
        // re-validation on the hot path so the benchmark measures iteration, not UTF-8 checking.
        let haystack_str = black_box(unsafe { std::str::from_utf8_unchecked(haystack) });
        let mut sum: u32 = 0;
        for character in haystack_str.chars() {
            sum = sum.wrapping_add(character as u32);
        }
        black_box(sum);
        WorkUnits::bytes(haystack_length)
    });
}

/// Benchmarks locating the byte offset of the Nth UTF-8 codepoint.
///
/// - `stringzilla::find_nth_utf8()`: SIMD-accelerated scan to the Nth codepoint
/// - `std::str::char_indices().nth()`: scalar decode-and-count baseline
///
/// We target the *last* codepoint, so every implementation scans the whole buffer once —
/// a fair workload whose throughput is simply the input size.
fn bench_find_nth_utf8(budget: &BenchBudget, haystack: &[u8], _needles: &BytesCowsAuto) {
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

    // Benchmark for StringZilla's SIMD `find_nth_utf8`.
    measure_throughput(
        "find-nth-utf8/stringzilla::find_nth_utf8",
        ReportAs::Bytes,
        budget,
        || {
            let offset = sz::find_nth_utf8(black_box(haystack), last_index);
            black_box(offset);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for the stdlib scalar baseline.
    measure_throughput(
        "find-nth-utf8/std::char_indices.nth",
        ReportAs::Bytes,
        budget,
        || {
            let offset = black_box(haystack_str)
                .char_indices()
                .nth(last_index)
                .map(|(byte_offset, _)| byte_offset);
            black_box(offset);
            WorkUnits::bytes(haystack_length)
        },
    );
}
fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables
    let tape = load_dataset("lines", COMPUTE_BOUND_SLICE, "data/xlsum/xlsum.csv").unwrap_nice();

    // Get the parent data directly from the tape (zero-copy)
    let haystack = tape.parent();
    let needles = &tape;

    let budget = BenchBudget::from_env(3.0, 20.0);

    println!("# tokenize-whitespace");
    bench_tokenize_whitespace(&budget, haystack, needles);

    println!("# tokenize-newlines");
    bench_tokenize_newlines(&budget, haystack, needles);

    println!("# tokenize-words-tr29");
    bench_tokenize_words_tr29(&budget, haystack, needles);

    println!("# tokenize-graphemes-tr29");
    bench_tokenize_graphemes(&budget, haystack, needles);

    println!("# tokenize-sentences-tr29");
    bench_tokenize_sentences(&budget, haystack, needles);

    println!("# tokenize-lines-uax14");
    bench_tokenize_lines_uax14(&budget, haystack, needles);

    println!("# utf8-length");
    bench_utf8_length(&budget, haystack, needles);

    println!("# utf8-iterate");
    bench_utf8_iterate(&budget, haystack, needles);

    println!("# find-nth-utf8");
    bench_find_nth_utf8(&budget, haystack, needles);
}
