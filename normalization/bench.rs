#![doc = r#"
# StringWars: Case Folding & Normalization Benchmarks

This file benchmarks Unicode case-insensitive operations and normalization:
- Case folding transformation
- Case-insensitive string comparison
- Case-insensitive substring search
- Unicode normalization (NFC / NFD / NFKC / NFKD)

## Benchmark Groups

- `case-fold`: Case folding transformation
- `normalize`: Unicode normalization (NFC / NFD / NFKC / NFKD)
- `case-insensitive-compare`: Case-insensitive equality
- `case-insensitive-find`: Case-insensitive substring search

## Usage Examples

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_normalization --bench bench_normalization
```
"#]
use std::hint::black_box;

use rand::prelude::IndexedRandom;
use rand::SeedableRng;
use stringtape::BytesCowsAuto;

use icu::casemap::CaseMapper;
use icu::normalizer::{ComposingNormalizerBorrowed, DecomposingNormalizerBorrowed};
use memchr::memmem;
use pcre2::bytes::RegexBuilder;
use stringzilla::sz;
use stringzilla::sz::Utf8NormalForm;
use unicase::UniCase;
use unicode_normalization::UnicodeNormalization;

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput, BenchBudget,
    ReportAs, ResultExt, WorkUnits, COMPUTE_BOUND_SLICE,
};

fn log_pcre2_metadata() {
    let (major, minor) = pcre2::version();
    println!("PCRE2 v{}.{}", major, minor);
    println!("- JIT available: {}", pcre2::is_jit_available());
}
/// Benchmarks case folding transformation throughput.
///
/// Unicode case folding may expand characters (e.g., German ß → ss).
/// - `stringzilla::utf8_uncased_fold()`: Full Unicode case folding per Unicode Standard
/// - `stdlib::to_lowercase()`: Full Unicode lowercasing (locale-independent, allocates)
fn bench_case_fold(budget: &BenchBudget, haystack: &[u8]) {
    let haystack_length = haystack.len() as u64;

    let haystack_str = std::str::from_utf8(haystack).unwrap();

    // Pre-allocate buffer for StringZilla case folding (3x for worst-case expansion)
    let mut fold_buffer = vec![0u8; haystack.len() * 3];

    // Benchmark for StringZilla case folding (full Unicode).
    measure_throughput(
        "case-fold/stringzilla::utf8_uncased_fold",
        ReportAs::Bytes,
        budget,
        || {
            let input = black_box(haystack);
            let len = sz::utf8_uncased_fold(input, &mut fold_buffer);
            black_box(len);
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for stdlib to_lowercase (full Unicode, allocates).
    measure_throughput(
        "case-fold/std::to_lowercase",
        ReportAs::Bytes,
        budget,
        || {
            let input = black_box(haystack_str);
            let lowered = input.to_lowercase();
            black_box(lowered);
            WorkUnits::bytes(haystack_length)
        },
    );
}
/// Benchmarks Unicode normalization (NFC / NFD / NFKC / NFKD) throughput.
///
/// - `stringzilla::utf8_norm()`: single-pass SIMD normalization into a caller buffer
/// - `unicode-normalization`: the de-facto Rust crate, iterator of normalized `char`s
/// - `icu::normalizer`: ICU4X Composing/Decomposing normalizers over `&str`
///
/// Normalization is most meaningful on Indic / Arabic / Vietnamese / Korean corpora; on
/// ASCII-heavy inputs every implementation degenerates to a near-passthrough copy.
fn bench_normalize(budget: &BenchBudget, haystack: &[u8]) {
    let haystack_str = match std::str::from_utf8(haystack) {
        Ok(text) => text,
        Err(_) => {
            eprintln!("Warning: Haystack is not valid UTF-8, skipping normalization benchmarks");
            return;
        }
    };

    let haystack_length = haystack.len() as u64;

    // Benchmark for StringZilla normalization across all four forms. The form is a
    // runtime enum value, so a single loop covers every form without duplication.
    let mut normalization_buffer = vec![0u8; haystack.len() * 3];
    let stringzilla_forms = [
        ("NFC", Utf8NormalForm::Nfc),
        ("NFD", Utf8NormalForm::Nfd),
        ("NFKC", Utf8NormalForm::Nfkc),
        ("NFKD", Utf8NormalForm::Nfkd),
    ];
    for (form_name, form) in stringzilla_forms {
        let identifier = format!(
            "normalize-{}/stringzilla::utf8_norm",
            form_name.to_lowercase()
        );
        measure_throughput(&identifier, ReportAs::Bytes, budget, || {
            let length = sz::utf8_norm(black_box(haystack), form, &mut normalization_buffer);
            black_box(length);
            WorkUnits::bytes(haystack_length)
        });
    }

    // A single reusable UTF-8 output buffer shared by the baseline implementations, so
    // every benchmarked path writes into pre-allocated capacity exactly like StringZilla's
    // `normalization_buffer` above. `clear()` keeps the allocation; `extend`/`normalize_to`
    // refill it without touching the heap. This keeps the comparison apples-to-apples:
    // we measure normalization work, not allocator throughput.
    let mut string_buffer = String::with_capacity(haystack.len() * 3);

    // Benchmark for the `unicode-normalization` crate. Each form returns a distinct
    // iterator type (Decompositions vs Recompositions), so the four cases are spelled out.
    // `String::extend` consumes the iterator into the reused buffer without reallocating.
    measure_throughput(
        "normalize-nfc/unicode-normalization::nfc",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfc());
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );
    measure_throughput(
        "normalize-nfd/unicode-normalization::nfd",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfd());
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );
    measure_throughput(
        "normalize-nfkc/unicode-normalization::nfkc",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfkc());
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );
    measure_throughput(
        "normalize-nfkd/unicode-normalization::nfkd",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfkd());
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );

    // Benchmark for ICU4X normalizers. Composing forms (NFC/NFKC) and decomposing forms
    // (NFD/NFKD) are different types; both expose `normalize_to(&str, &mut impl fmt::Write)`,
    // which streams into the reused `string_buffer` instead of allocating a fresh `Cow`.
    let icu_nfc = ComposingNormalizerBorrowed::new_nfc();
    let icu_nfkc = ComposingNormalizerBorrowed::new_nfkc();
    let icu_nfd = DecomposingNormalizerBorrowed::new_nfd();
    let icu_nfkd = DecomposingNormalizerBorrowed::new_nfkd();
    measure_throughput(
        "normalize-nfc/icu::ComposingNormalizer::normalize_to",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            icu_nfc
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );
    measure_throughput(
        "normalize-nfd/icu::DecomposingNormalizer::normalize_to",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            icu_nfd
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );
    measure_throughput(
        "normalize-nfkc/icu::ComposingNormalizer::normalize_to",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            icu_nfkc
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );
    measure_throughput(
        "normalize-nfkd/icu::DecomposingNormalizer::normalize_to",
        ReportAs::Bytes,
        budget,
        || {
            string_buffer.clear();
            icu_nfkd
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
            WorkUnits::bytes(haystack_length)
        },
    );
}
/// Benchmarks case-insensitive string equality comparison.
fn bench_case_insensitive_compare(budget: &BenchBudget, needles: &BytesCowsAuto) {
    // We compare each pair of adjacent tokens
    let pairs: Vec<(&[u8], &[u8])> = needles
        .iter()
        .zip(needles.iter().skip(1))
        .take(1000) // Limit to avoid excessive runtime
        .collect();

    if pairs.is_empty() {
        eprintln!("Warning: Not enough tokens for case-insensitive comparison benchmarks");
        return;
    }

    // Decode each pair to `&str` once, outside the timed closures, so the string-based baselines
    // do not re-validate UTF-8 on every iteration. StringZilla compares the raw bytes directly.
    let pairs_str: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(left, right)| {
            (
                std::str::from_utf8(left).unwrap_or(""),
                std::str::from_utf8(right).unwrap_or(""),
            )
        })
        .collect();

    // Benchmark for StringZilla case-insensitive comparison. One pair is compared per call,
    // cycling through the pairs; throughput is reported as the bytes spanned by both sides.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "case-insensitive-compare/stringzilla::utf8_uncased_order",
            ReportAs::Bytes,
            budget,
            || {
                let (left, right) = pairs[cursor % pairs.len()];
                cursor += 1;
                let pair_bytes = (left.len() + right.len()) as u64;
                let equal = sz::utf8_uncased_order(left, right) == std::cmp::Ordering::Equal;
                black_box(equal);
                WorkUnits::new(1, pair_bytes)
            },
        );
    }

    // Benchmark for unicase equality.
    {
        let mut cursor = 0usize;
        measure_throughput(
            "case-insensitive-compare/unicase::eq",
            ReportAs::Bytes,
            budget,
            || {
                let (left_str, right_str) = pairs_str[cursor % pairs_str.len()];
                cursor += 1;
                let pair_bytes = (left_str.len() + right_str.len()) as u64;
                let equal = UniCase::new(left_str) == UniCase::new(right_str);
                black_box(equal);
                WorkUnits::new(1, pair_bytes)
            },
        );
    }

    // Benchmark for stdlib lowercase + equality (baseline).
    {
        let mut cursor = 0usize;
        measure_throughput(
            "case-insensitive-compare/std::to_lowercase.eq",
            ReportAs::Bytes,
            budget,
            || {
                let (left_str, right_str) = pairs_str[cursor % pairs_str.len()];
                cursor += 1;
                let pair_bytes = (left_str.len() + right_str.len()) as u64;
                let equal = left_str.to_lowercase() == right_str.to_lowercase();
                black_box(equal);
                WorkUnits::new(1, pair_bytes)
            },
        );
    }
}
/// Benchmarks case-insensitive substring search.
fn bench_case_insensitive_find(budget: &BenchBudget, haystack: &[u8], needles: &BytesCowsAuto) {
    let haystack_length = haystack.len() as u64;
    let haystack_str = std::str::from_utf8(haystack).unwrap();

    // Collect candidate needles (valid UTF-8, length >= 3)
    let candidates: Vec<&str> = needles
        .iter()
        .filter_map(|needle_bytes| std::str::from_utf8(needle_bytes).ok())
        .filter(|candidate| candidate.len() >= 3)
        .collect();

    if candidates.is_empty() {
        eprintln!("Warning: No suitable needles for case-insensitive find benchmarks");
        return;
    }

    // Random-sample 100 needles with fixed seed for reproducibility
    let mut random_generator = rand::rngs::StdRng::seed_from_u64(42);
    let search_needles: Vec<&str> = candidates
        .sample(&mut random_generator, 100.min(candidates.len()))
        .copied()
        .collect();

    // Rotate through needles across iterations with a plain counter. The harness runs the
    // measured closure serially, so a captured `FnMut` counter suffices and avoids the
    // atomic read-modify-write overhead on the hot path. Each call scans the full haystack
    // once with a single needle; throughput is the haystack size.
    // Benchmark for StringZilla case-insensitive find (all matches for one needle).
    {
        let mut needle_index = 0usize;
        measure_throughput(
            "case-insensitive-find/stringzilla::utf8_uncased_search",
            ReportAs::Bytes,
            budget,
            || {
                let haystack_bytes = black_box(haystack);
                let index = needle_index % search_needles.len();
                needle_index += 1;
                let needle = sz::Utf8UncasedNeedle::new(search_needles[index].as_bytes());
                let mut matches = 0usize;
                let mut remaining = haystack_bytes;
                while let Some((offset, len)) = sz::utf8_uncased_search(remaining, &needle) {
                    matches += 1;
                    remaining = &remaining[offset + len.max(1)..];
                }
                black_box(matches);
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // PCRE2 benchmarks for case-insensitive search with full Unicode case folding.
    // NOTE: We use PCRE2 instead of Rust's `regex` crate because `regex` only supports
    // Unicode "simple" case folding (1:1 character mappings). PCRE2 with `.utf(true)`
    // supports full Unicode case folding, including expansions like ß→ss, İ→i̇, ﬁ→fi.
    // This makes it a fair comparison against StringZilla's full case folding.
    // See: https://github.com/rust-lang/regex/blob/master/UNICODE.md

    // Variant 1: Pre-compiled with JIT (compilation cost excluded from benchmark)
    {
        let regexes: Vec<_> = search_needles
            .iter()
            .filter_map(|needle| {
                RegexBuilder::new()
                    .caseless(true)
                    .utf(true)
                    .jit_if_available(true)
                    .build(&pcre2::escape(needle))
                    .ok()
            })
            .collect();

        let mut needle_index = 0usize;
        measure_throughput(
            "case-insensitive-find/pcre2::pre-jit",
            ReportAs::Bytes,
            budget,
            || {
                let haystack_bytes: &[u8] = black_box(haystack);
                let index = needle_index % regexes.len();
                needle_index += 1;
                black_box(regexes[index].find_iter(haystack_bytes).count());
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // Variant 2: JIT compilation included in benchmark (compile + search per iteration)
    {
        let mut needle_index = 0usize;
        measure_throughput(
            "case-insensitive-find/pcre2::jit-on-fly",
            ReportAs::Bytes,
            budget,
            || {
                let haystack_bytes: &[u8] = black_box(haystack);
                let index = needle_index % search_needles.len();
                needle_index += 1;
                let needle = search_needles[index];
                let regex = RegexBuilder::new()
                    .caseless(true)
                    .utf(true)
                    .jit_if_available(true)
                    .build(&pcre2::escape(needle))
                    .unwrap();
                black_box(regex.find_iter(haystack_bytes).count());
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // Variant 3: No JIT (interpreter mode only)
    {
        let regexes: Vec<_> = search_needles
            .iter()
            .filter_map(|needle| {
                RegexBuilder::new()
                    .caseless(true)
                    .utf(true)
                    // No .jit_if_available() - uses interpreter
                    .build(&pcre2::escape(needle))
                    .ok()
            })
            .collect();

        let mut needle_index = 0usize;
        measure_throughput(
            "case-insensitive-find/pcre2::no-jit",
            ReportAs::Bytes,
            budget,
            || {
                let haystack_bytes: &[u8] = black_box(haystack);
                let index = needle_index % regexes.len();
                needle_index += 1;
                black_box(regexes[index].find_iter(haystack_bytes).count());
                WorkUnits::bytes(haystack_length)
            },
        );
    }

    // Benchmark for ICU case-fold + memchr SIMD search.
    // Full Unicode case folding (ß→ss) + fast byte search.
    // Folding happens inside the loop for fair comparison.
    {
        let case_mapper = CaseMapper::new();
        let mut needle_index = 0usize;
        measure_throughput(
            "case-insensitive-find/memchr::Finder<icu-fold>",
            ReportAs::Bytes,
            budget,
            || {
                let haystack_text = black_box(haystack_str);
                let index = needle_index % search_needles.len();
                needle_index += 1;
                let needle = search_needles[index];
                // Fold both haystack and needle inside the benchmark
                let folded_haystack = case_mapper.fold_string(haystack_text);
                let folded_needle = case_mapper.fold_string(needle);
                let finder = memmem::Finder::new(folded_needle.as_bytes());
                black_box(finder.find_iter(folded_haystack.as_bytes()).count());
                WorkUnits::bytes(haystack_length)
            },
        );
    }
}
fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    log_pcre2_metadata();

    // Load the dataset defined by the environment variables
    let tape = load_dataset("lines", COMPUTE_BOUND_SLICE, "data/xlsum/xlsum.csv").unwrap_nice();

    // Get the parent data directly from the tape (zero-copy)
    let haystack = tape.parent();
    let needles = &tape;

    let budget = BenchBudget::from_env(3.0, 20.0);

    println!("# case-fold");
    bench_case_fold(&budget, haystack);

    println!("# normalize");
    bench_normalize(&budget, haystack);

    println!("# case-insensitive-compare");
    bench_case_insensitive_compare(&budget, needles);

    println!("# case-insensitive-find");
    bench_case_insensitive_find(&budget, haystack, needles);
}
