#![doc = r#"# StringWars: Find

Substring, reverse-substring and byte-set search benchmarks.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_find --bench bench_find
```
"#]
use std::hint::black_box;

use stringtape::BytesCowsAuto;

use aho_corasick::AhoCorasick;
use bstr::ByteSlice;
use memchr::memmem;
use regex::bytes::Regex;
use stringzilla::sz;

use stringwars::{
    finish, install_panic_hook, log_stringzilla_metadata, log_timing_overhead, measure,
    resolve_dataset, MeasureSpec, ResultExt, Unit, WorkUnits,
};

/// How many needles make one pass.
///
/// The needle set has to be fixed and identical for every implementation. The old
/// loop cycled the whole token tape against the deadline, so a faster engine got
/// through more needles than a slower one — and since needle cost varies by more
/// than an order of magnitude (a frequent short word restarts the scan constantly;
/// a rare long line matches once), the engines were not scanning comparable work.
/// Sixteen keeps a pass to ~0.1-0.8s on a 256 MB haystack while spanning a range of
/// needle lengths.
const NEEDLES_PER_PASS: usize = 16;

/// Picks a deterministic, evenly spaced needle sample so every row scans the same set.
fn needle_sample<'a>(needles: &'a BytesCowsAuto) -> Vec<&'a [u8]> {
    let total = needles.len();
    if total == 0 {
        return Vec::new();
    }
    let stride = (total / NEEDLES_PER_PASS).max(1);
    (0..total)
        .step_by(stride)
        .take(NEEDLES_PER_PASS)
        .filter_map(|index| needles.get(index))
        .collect()
}

/// One pass scans every needle in the sample across the whole haystack, so the
/// needle mixture is identical in every sample and cancels exactly.
fn measure_search<Search: FnMut(&[u8])>(
    name: &str,
    sample: &[&[u8]],
    haystack_bytes: u64,
    mut search: Search,
) {
    let work = WorkUnits::new(sample.len() as u64, haystack_bytes * sample.len() as u64);
    measure(name, MeasureSpec::new(Unit::Bytes, work), || {
        for needle in sample {
            search(black_box(needle));
        }
    });
}

/// Benchmarks forward substring search using "StringZilla", "MemMem", and standard strings.
///
/// Each call cycles to the next needle and scans the whole haystack for every occurrence of it,
/// so the per-call work is one full haystack pass (`haystack.len()` bytes), matching the original
/// `Throughput::Bytes(haystack.len())` accounting.
fn bench_substring_forward(haystack: &[u8], sample: &[&[u8]]) {
    let haystack_bytes = haystack.len() as u64;

    measure_search(
        "substring-forward/stringzilla::find",
        sample,
        haystack_bytes,
        |needle| {
            let mut position: usize = 0;
            while let Some(found) = sz::find(&haystack[position..], needle) {
                position += found + needle.len();
            }
        },
    );

    measure_search(
        "substring-forward/memmem::find",
        sample,
        haystack_bytes,
        |needle| {
            let mut position: usize = 0;
            while let Some(found) = memmem::find(&haystack[position..], needle) {
                position += found + needle.len();
            }
        },
    );

    measure_search(
        "substring-forward/memmem::Finder",
        sample,
        haystack_bytes,
        |needle| {
            let finder = memmem::Finder::new(needle);
            let mut position: usize = 0;
            while let Some(found) = finder.find(&haystack[position..]) {
                position += found + needle.len();
            }
        },
    );

    measure_search(
        "substring-forward/std::str::find",
        sample,
        haystack_bytes,
        |needle| {
            let mut position = 0;
            while let Some(found) = haystack[position..].find(needle) {
                position += found + needle.len();
            }
        },
    );
}

/// Benchmarks backward substring search using "StringZilla", "MemMem", and standard strings.
///
/// Each call cycles to the next needle and scans the whole haystack backward, so the per-call
/// work is one full haystack pass, matching the original `Throughput::Bytes(haystack.len())`.
fn bench_substring_backward(haystack: &[u8], sample: &[&[u8]]) {
    let haystack_bytes = haystack.len() as u64;

    measure_search(
        "substring-backward/stringzilla::rfind",
        sample,
        haystack_bytes,
        |needle| {
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = sz::rfind(&haystack[..end], needle) {
                    position = Some(found);
                } else {
                    break;
                }
            }
        },
    );

    measure_search(
        "substring-backward/memmem::rfind",
        sample,
        haystack_bytes,
        |needle| {
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = memmem::rfind(&haystack[..end], needle) {
                    position = Some(found);
                } else {
                    break;
                }
            }
        },
    );

    measure_search(
        "substring-backward/memmem::FinderRev",
        sample,
        haystack_bytes,
        |needle| {
            let finder = memmem::FinderRev::new(needle);
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = finder.rfind(&haystack[..end]) {
                    position = Some(found);
                } else {
                    break;
                }
            }
        },
    );

    measure_search(
        "substring-backward/std::str::rfind",
        sample,
        haystack_bytes,
        |needle| {
            let mut position: Option<usize> = Some(haystack.len());
            while let Some(end) = position {
                if let Some(found) = haystack[..end].rfind(needle) {
                    position = Some(found);
                } else {
                    break;
                }
            }
        },
    );
}

/// Benchmarks byteset search using "StringZilla", "bstr", "RegEx", and "AhoCorasick".
///
/// Each call cycles to the next needle token and runs all three bytesets over it. The original
/// looped over every needle in one iteration with `Throughput::Bytes(3 * haystack.len())`; since
/// the needles collectively span the haystack, the per-token equivalent is `3 * token.len()`.
fn bench_byteset_forward(needles: &BytesCowsAuto) {
    // Each token is scanned once per byteset, so a pass covers three times the tape.
    let byteset_work = WorkUnits::new(
        needles.len() as u64,
        3 * needles.iter().map(|t| t.len() as u64).sum::<u64>(),
    );
    // Define the three bytesets we will analyze.
    const BYTES_TABS: &[u8] = b"\n\r\x0B\x0C";
    const BYTES_HTML: &[u8] = b"</>&'\"=[]";
    const BYTES_DIGITS: &[u8] = b"0123456789";

    // Benchmark for StringZilla forward search.
    let sz_tabs = sz::Byteset::from(BYTES_TABS);
    let sz_html = sz::Byteset::from(BYTES_HTML);
    let sz_digits = sz::Byteset::from(BYTES_DIGITS);
    measure(
        "byteset-forward/stringzilla::find_byteset",
        MeasureSpec::new(Unit::Bytes, byteset_work),
        || {
            for token in needles.iter() {
                let token = black_box(token);
                let mut position: usize = 0;
                while let Some(found) = sz::find_byteset(&token[position..], sz_tabs) {
                    position += found + 1;
                }
                position = 0;
                while let Some(found) = sz::find_byteset(&token[position..], sz_html) {
                    position += found + 1;
                }
                position = 0;
                while let Some(found) = sz::find_byteset(&token[position..], sz_digits) {
                    position += found + 1;
                }
            }
        },
    );

    measure(
        "byteset-forward/bstr::find_byteset",
        MeasureSpec::new(Unit::Bytes, byteset_work),
        || {
            for token in needles.iter() {
                let token = black_box(token);
                let mut position: usize = 0;
                while let Some(found) = token[position..].find_byteset(BYTES_TABS) {
                    position += found + 1;
                }
                position = 0;
                while let Some(found) = token[position..].find_byteset(BYTES_HTML) {
                    position += found + 1;
                }
                position = 0;
                while let Some(found) = token[position..].find_byteset(BYTES_DIGITS) {
                    position += found + 1;
                }
            }
        },
    );

    // Benchmark for Regex-based byteset search.
    let re_tabs = Regex::new("[\n\r\x0B\x0C]").unwrap();
    let re_html = Regex::new("[</>&'\"=\\[\\]]").unwrap();
    let re_digits = Regex::new("[0-9]").unwrap();
    measure(
        "byteset-forward/regex::find_iter",
        MeasureSpec::new(Unit::Bytes, byteset_work),
        || {
            for token in needles.iter() {
                let token = black_box(token);
                black_box(re_tabs.find_iter(token).count());
                black_box(re_html.find_iter(token).count());
                black_box(re_digits.find_iter(token).count());
            }
        },
    );

    // Benchmark for Aho–Corasick-based byteset search.
    let ac_tabs = AhoCorasick::new(BYTES_TABS.chunks(1)).expect("failed to create AhoCorasick FSA");
    let ac_html = AhoCorasick::new(BYTES_HTML.chunks(1)).expect("failed to create AhoCorasick FSA");
    let ac_digits =
        AhoCorasick::new(BYTES_DIGITS.chunks(1)).expect("failed to create AhoCorasick FSA");
    measure(
        "byteset-forward/aho_corasick::find_iter",
        MeasureSpec::new(Unit::Bytes, byteset_work),
        || {
            for token in needles.iter() {
                let token = black_box(token);
                black_box(ac_tabs.find_iter(token).count());
                black_box(ac_html.find_iter(token).count());
                black_box(ac_digits.find_iter(token).count());
            }
        },
    );
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    let tape = resolve_dataset("find").unwrap_nice();
    log_timing_overhead();

    // The resolved tokens concatenated, not `tape.parent()`. The parent is the raw read:
    // it carries the separators between tokens and whatever trailing bytes the reader
    // pulled past the budget, so it is neither `token_bytes` long nor the same haystack
    // `find/bench.py` scans, which is exactly this concatenation. One startup allocation
    // buys a denominator both languages agree on.
    let joined: Vec<u8> = tape.iter().flatten().copied().collect();
    let haystack: &[u8] = &joined;
    let needles = &tape;
    let sample = needle_sample(needles);

    println!("# substring-forward");
    bench_substring_forward(haystack, &sample);

    println!("# substring-backward");
    bench_substring_backward(haystack, &sample);

    println!("# byteset-forward");
    bench_byteset_forward(needles);

    finish();
}
