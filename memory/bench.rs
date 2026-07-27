#![doc = r#"# StringWars: Memory

Low-level memory benchmarks: lookup-table transforms, PRNG fills, memset, memcpy, memmove.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_memory --bench bench_memory
```
"#]
use std::error::Error;
use std::hint::black_box;
use std::ptr;
use std::slice;

use rand::{Rng, SeedableRng};
use stringzilla::sz;
use zeroize::Zeroize;

use stringwars::{
    finish, install_panic_hook, log_stringzilla_metadata, log_timing_overhead, measure,
    note_working_set, read_within_budget, suite_settings, token_ranges, MeasureSpec, ResultExt,
    Unit, WorkUnits,
};

/// Cycles `tokens` by a local cursor, calls `work` on the current mutable token, and
/// reports throughput as bytes equal to the token length.  This avoids repeating the
/// `{ let mut cursor = 0usize; measure_throughput(…) }` block for every single-buffer
/// variant that transforms one token in place.
fn measure_mut_token<Work: FnMut(&mut [u8])>(name: &str, tokens: &mut [&mut [u8]], mut work: Work) {
    let pass = WorkUnits::new(
        tokens.len() as u64,
        tokens.iter().map(|token| token.len() as u64).sum(),
    );
    measure(name, MeasureSpec::new(Unit::Bytes, pass), || {
        for token in tokens.iter_mut() {
            work(token);
        }
    });
}

/// Reads the raw dataset bytes named by `STRINGWARS_DATASET`.
///
/// The in-place LUT/translate/PRNG benchmarks mutate their tokens, so they need owned,
/// mutable bytes and mutable token slices; the shared `stringwars::load_dataset` returns an
/// immutable, leaked tape and cannot be used here.
pub fn load_dataset_bytes() -> Result<Vec<u8>, Box<dyn Error>> {
    let settings = suite_settings("memory");
    let path = settings.dataset.ok_or("No dataset for suite 'memory'")?;
    Ok(read_within_budget(
        &path,
        settings.budget_bytes,
        &settings.tokens,
    )?)
}

/// Borrows the working set as mutable token slices.
///
/// The split rule itself lives in `stringwars::token_ranges`; this only turns the
/// ranges into disjoint `&mut` slices. The suite used to carry its own copy of the
/// rule, and the copy had drifted: it ignored `STRINGWARS_UNIQUE` and its `file`
/// arm applied neither the byte budget nor the UTF-8 backoff.
pub fn tokenize_mut(haystack: &mut [u8]) -> Result<Vec<&mut [u8]>, Box<dyn Error>> {
    let settings = suite_settings("memory");
    let ranges = token_ranges(haystack, &settings.tokens, settings.budget_bytes);

    // Hand out one disjoint `&mut` per range by splitting the tail repeatedly.
    let mut rest = haystack;
    let mut consumed = 0usize;
    let mut tokens = Vec::with_capacity(ranges.len());
    for range in ranges {
        let (_, tail) = rest.split_at_mut(range.start - consumed);
        let (token, tail) = tail.split_at_mut(range.end - range.start);
        tokens.push(token);
        consumed = range.end;
        rest = tail;
    }
    Ok(tokens)
}

/// Benchmarks in-place lookup-table transforms, transforming one token per call and cycling the
/// dataset. Throughput is reported as bytes/s, matching the original `Throughput::Bytes` over the
/// sum of token lengths.
fn bench_lookup_table(tokens: &mut [&mut [u8]]) {
    let mut lookup_invert_case: [u8; 256] = core::array::from_fn(|index| index as u8);
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[upper as usize] = lower as u8;
    }
    for (upper, lower) in ('A'..='Z').zip('a'..='z') {
        lookup_invert_case[lower as usize] = upper as u8;
    }

    measure_mut_token(
        "lookup-table/stringzilla::lookup_inplace",
        tokens,
        |token| {
            sz::lookup_inplace(token, lookup_invert_case);
            black_box(token);
        },
    );

    measure_mut_token("lookup-table/serial", tokens, |token| {
        for byte in token.iter_mut() {
            *byte = lookup_invert_case[*byte as usize];
        }
        black_box(&token);
    });
}

/// Benchmarks random-string generation, filling one token per call and cycling the dataset.
/// Throughput is reported as bytes/s, matching the original `Throughput::Bytes` over the sum of
/// token lengths.
fn bench_generate_random(tokens: &mut [&mut [u8]]) {
    measure_mut_token(
        "generate-random/stringzilla::fill_random",
        tokens,
        |token| {
            sz::fill_random(token, 0);
        },
    );

    measure_mut_token("generate-random/zeroize::zeroize", tokens, |token| {
        token.zeroize();
        black_box(&token);
    });

    measure_mut_token("generate-random/getrandom::fill", tokens, |token| {
        getrandom::fill(token).expect("getrandom failed");
        black_box(&token);
    });

    // Benchmark using `rand_chacha::ChaCha20Rng`. The generator is seeded once, outside
    // the timed region: `sz::fill_random` constructs nothing per token, so seeding per
    // token would charge these rows for setup their contender never pays.
    {
        let mut random_generator = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        measure_mut_token(
            "generate-random/rand_chacha::ChaCha20Rng",
            tokens,
            |token| {
                random_generator.fill_bytes(token);
                black_box(&token);
            },
        );
    }

    {
        let mut random_generator = rand_xoshiro::Xoshiro128Plus::from_seed([0u8; 16]);
        measure_mut_token(
            "generate-random/rand_xoshiro::Xoshiro128Plus",
            tokens,
            |token| {
                random_generator.fill_bytes(token);
                black_box(&token);
            },
        );
    }
}

/// Benchmarks memory-fill operations, filling one buffer per call and cycling the dataset.
fn bench_memset(tokens: &mut [&mut [u8]]) {
    const FILL_VALUE: u8 = 0xAA;
    let templates: Vec<Vec<u8>> = tokens.iter().map(|token| (**token).to_vec()).collect();
    let total_bytes: usize = templates.iter().map(|buffer| buffer.len()).sum();
    if total_bytes == 0 {
        return;
    }
    let buffer_count = templates.len();

    {
        let mut buffers = templates.clone();
        measure(
            "memset/stringzilla::fill",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, total_bytes as u64),
            ),
            || {
                for buffer in buffers.iter_mut() {
                    sz::fill(buffer, FILL_VALUE);
                    black_box(&buffer);
                }
            },
        );
    }

    {
        let mut buffers = templates.clone();
        measure(
            "memset/std::ptr::write_bytes",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, total_bytes as u64),
            ),
            || {
                for buffer in buffers.iter_mut() {
                    unsafe {
                        ptr::write_bytes(buffer.as_mut_ptr(), FILL_VALUE, buffer.len());
                    }
                    black_box(&buffer);
                }
            },
        );
    }

    {
        let mut buffers = templates.clone();
        measure(
            "memset/slice::fill",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, total_bytes as u64),
            ),
            || {
                for buffer in buffers.iter_mut() {
                    buffer.fill(FILL_VALUE);
                    black_box(&buffer);
                }
            },
        );
    }
}

/// Benchmarks memory-copy operations, copying one buffer per call and cycling the dataset.
fn bench_memcpy(tokens: &mut [&mut [u8]]) {
    let sources: Vec<Vec<u8>> = tokens.iter().map(|token| (**token).to_vec()).collect();
    let dest_template: Vec<Vec<u8>> = sources.iter().map(|src| vec![0u8; src.len()]).collect();
    let total_bytes: usize = sources.iter().map(|buffer| buffer.len()).sum();
    if total_bytes == 0 {
        return;
    }
    let buffer_count = sources.len();

    {
        let mut dests = dest_template.clone();
        measure(
            "memcpy/stringzilla::copy",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, total_bytes as u64),
            ),
            || {
                for index in 0..buffer_count {
                    let source = &sources[index];
                    let dest = &mut dests[index];
                    sz::copy(dest, source);
                    black_box(&dest);
                }
            },
        );
    }

    {
        let mut dests = dest_template.clone();
        measure(
            "memcpy/slice::copy_from_slice",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, total_bytes as u64),
            ),
            || {
                for index in 0..buffer_count {
                    let source = &sources[index];
                    let dest = &mut dests[index];
                    dest.copy_from_slice(source);
                    black_box(&dest);
                }
            },
        );
    }

    {
        let mut dests = dest_template.clone();
        measure(
            "memcpy/std::ptr::copy_nonoverlapping",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, total_bytes as u64),
            ),
            || {
                for index in 0..buffer_count {
                    let source = &sources[index];
                    let dest = &mut dests[index];
                    unsafe {
                        ptr::copy_nonoverlapping(source.as_ptr(), dest.as_mut_ptr(), source.len());
                    }
                    black_box(&dest);
                }
            },
        );
    }
}

/// Benchmarks memory-move operations, shifting one buffer per call and cycling the dataset.
/// Only tokens longer than `SHIFT` participate, and the per-call byte count is `len - SHIFT`,
/// matching the original `Throughput::Bytes(sum(len - SHIFT))` accounting.
fn bench_memmove(tokens: &mut [&mut [u8]]) {
    const SHIFT: usize = 8;
    let templates: Vec<Vec<u8>> = tokens
        .iter()
        .filter_map(|token| {
            let slice = &**token;
            if slice.len() <= SHIFT {
                None
            } else {
                Some(slice.to_vec())
            }
        })
        .collect();
    if templates.is_empty() {
        return;
    }
    let buffer_count = templates.len();
    let moved_bytes: usize = templates
        .iter()
        .map(|b| b.len().saturating_sub(SHIFT))
        .sum();

    {
        let mut buffers = templates.clone();
        measure(
            "memmove/stringzilla::move_",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, moved_bytes as u64),
            ),
            || {
                for buffer in buffers.iter_mut() {
                    let move_len = buffer.len() - SHIFT;
                    unsafe {
                        let source = slice::from_raw_parts(buffer.as_ptr(), move_len);
                        let dest =
                            slice::from_raw_parts_mut(buffer.as_mut_ptr().add(SHIFT), move_len);
                        sz::move_(dest, &source);
                    }
                    black_box(&buffer);
                }
            },
        );
    }

    {
        let mut buffers = templates.clone();
        measure(
            "memmove/std::ptr::copy",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, moved_bytes as u64),
            ),
            || {
                for buffer in buffers.iter_mut() {
                    let move_len = buffer.len() - SHIFT;
                    unsafe {
                        ptr::copy(buffer.as_ptr(), buffer.as_mut_ptr().add(SHIFT), move_len);
                    }
                    black_box(&buffer);
                }
            },
        );
    }

    {
        let mut buffers = templates.clone();
        measure(
            "memmove/slice::copy_within",
            MeasureSpec::new(
                Unit::Bytes,
                WorkUnits::new(buffer_count as u64, moved_bytes as u64),
            ),
            || {
                for buffer in buffers.iter_mut() {
                    let move_len = buffer.len() - SHIFT;
                    buffer.copy_within(0..move_len, SHIFT);
                    black_box(&buffer);
                }
            },
        );
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    let mut dataset = load_dataset_bytes().unwrap_nice();
    let mut tokens = tokenize_mut(&mut dataset).unwrap_nice();
    if tokens.is_empty() {
        panic!("No tokens found in the dataset.");
    }
    note_working_set(
        "memory",
        &tokens.iter().map(|token| &**token).collect::<Vec<_>>(),
    );
    log_timing_overhead();

    println!("# lookup-table");
    bench_lookup_table(&mut tokens[..]);

    println!("# generate-random");
    bench_generate_random(&mut tokens[..]);

    println!("# memset");
    bench_memset(&mut tokens[..]);

    println!("# memcpy");
    bench_memcpy(&mut tokens[..]);

    println!("# memmove");
    bench_memmove(&mut tokens[..]);

    finish();
}
