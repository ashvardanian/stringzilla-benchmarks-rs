#![doc = r#"# StringWars: Normalization

Unicode normalization and case-insensitive comparison benchmarks.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_normalization --bench bench_normalization
```
"#]
use std::hint::black_box;

/// Needles scanned per pass. Matches `find/bench.rs`: one pass covers the whole set
/// so the needle mixture cancels instead of varying between samples.
const NEEDLES_PER_PASS: usize = 16;

use rand::prelude::IndexedRandom;
use rand::SeedableRng;

use icu::casemap::CaseMapper;
use icu::normalizer::{ComposingNormalizerBorrowed, DecomposingNormalizerBorrowed};
use memchr::memmem;
use pcre2::bytes::RegexBuilder;
use stringzilla::sz;
use stringzilla::sz::Utf8NormalForm;
use unicase::UniCase;
use unicode_normalization::UnicodeNormalization;

use stringwars::{
    finish, install_panic_hook, log_stringzilla_metadata, log_timing_overhead, measure,
    note_unavailable, resolve_dataset, token_ranges, MeasureSpec, ResultExt, Unit, WorkUnits,
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
fn bench_case_fold(haystack: &[u8]) {
    let haystack_length = haystack.len() as u64;

    let haystack_str = std::str::from_utf8(haystack).unwrap();

    // Pre-allocate buffer for StringZilla case folding (3x for worst-case expansion)
    let mut fold_buffer = vec![0u8; haystack.len() * 3];

    measure(
        "case-fold/stringzilla::utf8_uncased_fold",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let input = black_box(haystack);
            let len = sz::utf8_uncased_fold(input, &mut fold_buffer);
            black_box(len);
        },
    );

    measure(
        "case-fold/std::to_lowercase",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            let input = black_box(haystack_str);
            let lowered = input.to_lowercase();
            black_box(lowered);
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
fn bench_normalize(haystack: &[u8]) {
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
        measure(
            &identifier,
            MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
            || {
                let length = sz::utf8_norm(black_box(haystack), form, &mut normalization_buffer);
                black_box(length);
            },
        );
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
    measure(
        "normalize-nfc/unicode-normalization::nfc",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfc());
            black_box(string_buffer.len());
        },
    );
    measure(
        "normalize-nfd/unicode-normalization::nfd",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfd());
            black_box(string_buffer.len());
        },
    );
    measure(
        "normalize-nfkc/unicode-normalization::nfkc",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfkc());
            black_box(string_buffer.len());
        },
    );
    measure(
        "normalize-nfkd/unicode-normalization::nfkd",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            string_buffer.extend(black_box(haystack_str).nfkd());
            black_box(string_buffer.len());
        },
    );

    // Benchmark for ICU4X normalizers. Composing forms (NFC/NFKC) and decomposing forms
    // (NFD/NFKD) are different types; both expose `normalize_to(&str, &mut impl fmt::Write)`,
    // which streams into the reused `string_buffer` instead of allocating a fresh `Cow`.
    let icu_nfc = ComposingNormalizerBorrowed::new_nfc();
    let icu_nfkc = ComposingNormalizerBorrowed::new_nfkc();
    let icu_nfd = DecomposingNormalizerBorrowed::new_nfd();
    let icu_nfkd = DecomposingNormalizerBorrowed::new_nfkd();
    measure(
        "normalize-nfc/icu::ComposingNormalizer::normalize_to",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            icu_nfc
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
        },
    );
    measure(
        "normalize-nfd/icu::DecomposingNormalizer::normalize_to",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            icu_nfd
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
        },
    );
    measure(
        "normalize-nfkc/icu::ComposingNormalizer::normalize_to",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            icu_nfkc
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
        },
    );
    measure(
        "normalize-nfkd/icu::DecomposingNormalizer::normalize_to",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(haystack_length)),
        || {
            string_buffer.clear();
            icu_nfkd
                .normalize_to(black_box(haystack_str), &mut string_buffer)
                .unwrap();
            black_box(string_buffer.len());
        },
    );
}
/// Benchmarks case-insensitive string equality comparison.
fn bench_case_insensitive_compare(needles: &[&[u8]]) {
    // We compare each pair of adjacent tokens
    // Adjacent pairs; two slices rather than a Vec of tuples.
    let pairs: Vec<(&[u8], &[u8])> = needles
        .iter()
        .zip(needles.iter().skip(1))
        .take(1000)
        .map(|(left, right)| (*left, *right))
        .collect();

    if pairs.is_empty() {
        eprintln!("Warning: Not enough tokens for case-insensitive comparison benchmarks");
        return;
    }

    // A pass compares every pair once, so the pair mixture is identical in every sample.
    // All three rows are priced by this one declaration rather than by re-deriving it
    // from whichever representation each row happens to iterate.
    let pair_work = WorkUnits::new(
        pairs.len() as u64,
        pairs
            .iter()
            .map(|(left, right)| (left.len() + right.len()) as u64)
            .sum(),
    );

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
        measure(
            "case-insensitive-compare/stringzilla::utf8_uncased_order",
            MeasureSpec::new(Unit::Bytes, pair_work),
            || {
                for &(left, right) in pairs.iter() {
                    let equal = sz::utf8_uncased_order(left, right) == std::cmp::Ordering::Equal;
                    black_box(equal);
                }
            },
        );
    }

    {
        measure(
            "case-insensitive-compare/unicase::eq",
            MeasureSpec::new(Unit::Bytes, pair_work),
            || {
                for &(left_str, right_str) in pairs_str.iter() {
                    let equal = UniCase::new(left_str) == UniCase::new(right_str);
                    black_box(equal);
                }
            },
        );
    }

    {
        measure(
            "case-insensitive-compare/std::to_lowercase.eq",
            MeasureSpec::new(Unit::Bytes, pair_work),
            || {
                for &(left_str, right_str) in pairs_str.iter() {
                    let equal = left_str.to_lowercase() == right_str.to_lowercase();
                    black_box(equal);
                }
            },
        );
    }
}
/// Benchmarks case-insensitive substring search.
fn bench_case_insensitive_find(haystack: &[u8], needles: &[&[u8]]) {
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
        .sample(
            &mut random_generator,
            NEEDLES_PER_PASS.min(candidates.len()),
        )
        .copied()
        .collect();
    // One pass scans the haystack once per needle, so every sample does identical
    // work. Rotating one needle per pass made the cost depend on which needle came
    // up, and the rows never converged.
    let scan_work = WorkUnits::bytes(haystack_length * search_needles.len() as u64);

    // Rotate through needles across iterations with a plain counter. The harness runs the
    // measured closure serially, so a captured `FnMut` counter suffices and avoids the
    // atomic read-modify-write overhead on the hot path. Each call scans the full haystack
    // once with a single needle; throughput is the haystack size.
    {
        measure(
            "case-insensitive-find/stringzilla::utf8_uncased_search",
            MeasureSpec::new(Unit::Bytes, scan_work),
            || {
                for text in &search_needles {
                    let haystack_bytes = black_box(haystack);
                    let needle = sz::Utf8UncasedNeedle::new(text.as_bytes());
                    let mut matches = 0usize;
                    let mut remaining = haystack_bytes;
                    while let Some((offset, len)) = sz::utf8_uncased_search(remaining, &needle) {
                        matches += 1;
                        remaining = &remaining[offset + len.max(1)..];
                    }
                    black_box(matches);
                }
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

        // `filter_map(..ok())` drops every needle PCRE2 rejects. If it drops them
        // all, the `% regexes.len()` below divides by zero — which is how this row
        // used to abort the whole suite instead of reporting a skip.
        if regexes.is_empty() {
            note_unavailable(
                "case-insensitive-find/pcre2::pre-jit",
                "no needle compiled under PCRE2",
            );
        } else {
            measure(
                "case-insensitive-find/pcre2::pre-jit",
                MeasureSpec::new(Unit::Bytes, scan_work),
                || {
                    for regex in &regexes {
                        let haystack_bytes: &[u8] = black_box(haystack);
                        black_box(regex.find_iter(haystack_bytes).count());
                    }
                },
            );
        }
    }

    // Variant 2: JIT compilation included in benchmark (compile + search per iteration)
    {
        measure(
            "case-insensitive-find/pcre2::jit-on-fly",
            MeasureSpec::new(Unit::Bytes, scan_work),
            || {
                for needle in &search_needles {
                    let haystack_bytes: &[u8] = black_box(haystack);
                    let Ok(regex) = RegexBuilder::new()
                        .caseless(true)
                        .utf(true)
                        .jit_if_available(true)
                        .build(&pcre2::escape(needle))
                    else {
                        continue;
                    };
                    black_box(regex.find_iter(haystack_bytes).count());
                }
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

        // `filter_map(..ok())` drops every needle PCRE2 rejects. If it drops them
        // all, the `% regexes.len()` below divides by zero — which is how this row
        // used to abort the whole suite instead of reporting a skip.
        if regexes.is_empty() {
            note_unavailable(
                "case-insensitive-find/pcre2::no-jit",
                "no needle compiled under PCRE2",
            );
        } else {
            measure(
                "case-insensitive-find/pcre2::no-jit",
                MeasureSpec::new(Unit::Bytes, scan_work),
                || {
                    for regex in &regexes {
                        let haystack_bytes: &[u8] = black_box(haystack);
                        black_box(regex.find_iter(haystack_bytes).count());
                    }
                },
            );
        }
    }

    // Benchmark for ICU case-fold + memchr SIMD search.
    // Full Unicode case folding (ß→ss) + fast byte search.
    // Folding happens inside the loop for fair comparison.
    {
        let case_mapper = CaseMapper::new();
        measure(
            "case-insensitive-find/memchr::Finder<icu-fold>",
            MeasureSpec::new(Unit::Bytes, scan_work),
            || {
                for needle in &search_needles {
                    let haystack_text = black_box(haystack_str);
                    // Folding stays inside the loop: it is what this row measures.
                    let folded_haystack = case_mapper.fold_string(haystack_text);
                    let folded_needle = case_mapper.fold_string(needle);
                    let finder = memmem::Finder::new(folded_needle.as_bytes());
                    black_box(finder.find_iter(folded_haystack.as_bytes()).count());
                }
            },
        );
    }
}
fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    log_pcre2_metadata();

    let tape = resolve_dataset("normalization").unwrap_nice();
    log_timing_overhead();

    // The tape's single token, not `parent()`: the parent is the raw read, cut at exactly
    // the budget and so liable to end mid-character, while the token carries the UTF-8
    // backoff. It is also what the Python side measures, so the two stay comparable.
    let haystack: &[u8] = tape.iter().next().expect("empty working set");
    // The manifest gives this suite `tokens = "file"`, so the tape is one 16 MB
    // token. Case-folding and normalization want exactly that, but the find and
    // compare groups need real needles - searching for the whole haystack inside
    // itself matched once and made PCRE2 refuse a 16 MB literal. Split words off
    // the same buffer with the shared rule.
    let needles: Vec<&[u8]> = token_ranges(haystack, "words", haystack.len() as u64)
        .into_iter()
        .map(|range| &haystack[range])
        .collect();

    println!("# case-fold");
    bench_case_fold(haystack);

    println!("# normalize");
    bench_normalize(haystack);

    println!("# case-insensitive-compare");
    bench_case_insensitive_compare(&needles);

    println!("# case-insensitive-find");
    bench_case_insensitive_find(haystack, &needles);

    finish();
}
