#![doc = r#"# StringWars: Hash

Hashing benchmarks: stateless, stateful and checksum digests over the working set.

- `STRINGWARS_COLLISIONS=1` also reports collision rates; off by default because
  deduplicating a large corpus costs gigabytes.

  `cityhash` is x86_64-only and is skipped elsewhere.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_hash --bench bench_hash
```
"#]
use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use std::hint::black_box;

use bit_set::BitSet;
use stringtape::{BytesCowsAuto, BytesTape};

use ahash::RandomState as AHashState;
use ring::digest as ring_digest;
use sha2::{Digest, Sha256};
use stringzilla::sz;
use wyhash::wyhash;
use xxhash_rust::xxh3::xxh3_64;

use stringwars::{
    finish, get_env_bool, install_panic_hook, log_stringzilla_metadata, log_timing_overhead,
    measure, note_unavailable, resolve_dataset, should_run, MeasureSpec, ResultExt, Unit,
    WorkUnits,
};

/// Benchmarks one stateless hash that produces a `u64` result: runs `bench_each_token` for
/// throughput and, when unique tokens are available, prints the collision rate.
///
/// Exceptions left inline: `crc32fast::hash` (returns `u32`, closure shape differs) and
/// the `#[cfg(target_arch = "x86_64")]` `cityhash` block (cfg-gated, must stay inline).
fn bench_stateless_hash<HashFn: Fn(&[u8]) -> u64 + Copy>(
    name: &str,
    work: WorkUnits,
    slices: &[&[u8]],
    unique_tokens: &[&[u8]],
    hash_fn: HashFn,
) {
    bench_each_token(name, slices, work, |token| {
        let _ = black_box(hash_fn(token));
    });
    if !unique_tokens.is_empty() && should_run(name) {
        print_collision_rate(unique_tokens, hash_fn);
    }
}

/// Times one stateless hash over the whole working set.
///
/// A pass is one complete traversal, so the mixture of token lengths is identical
/// in every sample and cancels exactly, and the work is declared once instead of
/// being accumulated per call.
fn bench_each_token<HashOne: FnMut(&[u8])>(
    name: &str,
    tokens: &[&[u8]],
    work: WorkUnits,
    mut hash_one: HashOne,
) {
    measure(name, MeasureSpec::new(Unit::Bytes, work), || {
        for token in tokens {
            hash_one(black_box(token));
        }
    });
}

/// Counts collisions for a given hash function using a bitset sized to the number of unique tokens
fn count_collisions<F>(unique_tokens: &[&[u8]], hash_fn: F) -> usize
where
    F: Fn(&[u8]) -> u64,
{
    if unique_tokens.is_empty() {
        return 0;
    }

    let table_size = unique_tokens.len();
    let mut bitset = BitSet::with_capacity(table_size);
    let mut collisions = 0;

    for token in unique_tokens {
        let hash = hash_fn(token);
        let bit_pos = (hash as usize) % table_size;
        collisions += !bitset.insert(bit_pos) as usize;
    }

    collisions
}

/// Calculate and print collision rate for a hash function using a bitset matching the unique token count
fn print_collision_rate<F>(unique_tokens: &[&[u8]], hash_fn: F)
where
    F: Fn(&[u8]) -> u64,
{
    let n_unique = unique_tokens.len();
    if n_unique == 0 {
        return;
    }

    let collisions = count_collisions(unique_tokens, hash_fn);
    let rate = (collisions as f64 / n_unique as f64) * 100.0;

    println!(
        "                        collisions: {:.2}% ({} collisions across {} buckets)",
        rate, collisions, n_unique
    );
}

/// Benchmarks stateless hashes, hashing one token per call and cycling the dataset.
fn bench_stateless(work: WorkUnits, tokens: &BytesCowsAuto) {
    // Collision detection is opt-in via STRINGWARS_COLLISIONS environment variable.
    // This avoids OOM on large datasets (can use GBs of RAM for deduplication).
    let enable_collision_detection = get_env_bool("STRINGWARS_COLLISIONS");
    let unique_tokens: Vec<&[u8]> = if enable_collision_detection {
        println!("\nComputing unique tokens for collision detection...");
        let unique_set: HashSet<&[u8]> = tokens.iter().collect();
        let unique: Vec<&[u8]> = unique_set.into_iter().collect();
        println!(
            "Collision statistics for {} unique tokens (from {} total):",
            unique.len(),
            tokens.len()
        );
        unique
    } else {
        Vec::new()
    };

    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let slices: Vec<&[u8]> = (&view).into_iter().collect();

    bench_stateless_hash(
        "stateless/stringzilla::hash",
        work,
        &slices,
        &unique_tokens,
        |token| sz::hash(token),
    );

    // Benchmark: SipHash via `std::DefaultHasher`
    let std_builder = std::collections::hash_map::RandomState::new();
    bench_stateless_hash(
        "stateless/std::DefaultHasher::hash_one",
        work,
        &slices,
        &unique_tokens,
        |token| std_builder.hash_one(token),
    );

    // Benchmark: aHash
    let hash_builder = AHashState::with_seed(42);
    bench_stateless_hash(
        "stateless/ahash::hash_one",
        work,
        &slices,
        &unique_tokens,
        |token| hash_builder.hash_one(token),
    );

    bench_stateless_hash(
        "stateless/xxh3::xxh3_64",
        work,
        &slices,
        &unique_tokens,
        xxh3_64,
    );

    bench_stateless_hash(
        "stateless/wyhash::wyhash",
        work,
        &slices,
        &unique_tokens,
        |token| wyhash(token, 42),
    );

    // Benchmark: FoldHash
    let foldhash_builder = foldhash::fast::RandomState::default();
    bench_stateless_hash(
        "stateless/foldhash::hash_one",
        work,
        &slices,
        &unique_tokens,
        |token| foldhash_builder.hash_one(token),
    );

    // Benchmark: CRC32 — left inline because `crc32fast::hash` returns `u32`, so the
    // bench closure black-boxes a `u32` while the collision closure casts to `u64`;
    // the two-closure shapes differ from `bench_stateless_hash`.
    bench_each_token("stateless/crc32fast::hash", &slices, work, |token| {
        let _ = black_box(crc32fast::hash(token));
    });
    if !unique_tokens.is_empty() && should_run("stateless/crc32fast::hash") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            crc32fast::hash(token_bytes) as u64
        });
    }

    bench_stateless_hash(
        "stateless/murmurhash32::murmurhash3",
        work,
        &slices,
        &unique_tokens,
        |token| murmurhash32::murmurhash3(token) as u64,
    );

    // Inline because a cfg gate cannot sit on a bare call expression.
    #[cfg(not(target_arch = "x86_64"))]
    note_unavailable("stateless/cityhash::city_hash_64", "x86_64 only");
    #[cfg(target_arch = "x86_64")]
    {
        bench_each_token("stateless/cityhash::city_hash_64", &slices, work, |token| {
            let _ = black_box(cityhash::city_hash_64(token));
        });
        if !unique_tokens.is_empty() && should_run("stateless/cityhash::city_hash_64") {
            print_collision_rate(&unique_tokens, |token_bytes| {
                cityhash::city_hash_64(token_bytes)
            });
        }
    }

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Benchmarks checksum hashes including cryptographic hashes and reference bounds.
fn bench_checksum(work: WorkUnits, tokens: &BytesCowsAuto) {
    let enable_collision_detection = get_env_bool("STRINGWARS_COLLISIONS");
    let unique_tokens: Vec<&[u8]> = if enable_collision_detection {
        let unique_set: HashSet<&[u8]> = tokens.iter().collect();
        let unique: Vec<&[u8]> = unique_set.into_iter().collect();
        println!(
            "Collision statistics for {} unique tokens (from {} total):",
            unique.len(),
            tokens.len()
        );
        unique
    } else {
        Vec::new()
    };

    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let slices: Vec<&[u8]> = (&view).into_iter().collect();

    bench_each_token("checksum/stringzilla::bytesum", &slices, work, |token| {
        let _ = black_box(sz::bytesum(token));
    });

    bench_each_token("checksum/blake3::hash", &slices, work, |token| {
        let _ = black_box(blake3::hash(token));
    });
    if !unique_tokens.is_empty() && should_run("checksum/blake3::hash") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let hash = blake3::hash(token_bytes);
            let bytes = hash.as_bytes();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
    }

    bench_each_token("checksum/sha2::Sha256", &slices, work, |token| {
        let mut hasher = Sha256::new();
        hasher.update(token);
        let _ = black_box(hasher.finalize());
    });
    if !unique_tokens.is_empty() && should_run("checksum/sha2::Sha256") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let mut hasher = Sha256::new();
            hasher.update(token_bytes);
            let result = hasher.finalize();
            u64::from_le_bytes([
                result[0], result[1], result[2], result[3], result[4], result[5], result[6],
                result[7],
            ])
        });
    }

    bench_each_token("checksum/ring::SHA256", &slices, work, |token| {
        let _ = black_box(ring_digest::digest(&ring_digest::SHA256, token));
    });
    if !unique_tokens.is_empty() && should_run("checksum/ring::SHA256") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let digest = ring_digest::digest(&ring_digest::SHA256, token_bytes);
            let bytes = digest.as_ref();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
    }

    bench_each_token("checksum/stringzilla::Sha256", &slices, work, |token| {
        let _ = black_box(sz::Sha256::hash(token));
    });
    if !unique_tokens.is_empty() && should_run("checksum/stringzilla::Sha256") {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let digest = sz::Sha256::hash(token_bytes);
            u64::from_le_bytes([
                digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                digest[7],
            ])
        });
    }

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Benchmarks stateful hashes, streaming the whole dataset through one hasher per pass and
/// cycling passes for the budget. The per-call unit is one full streaming pass, so the deadline
/// check after each call bounds overshoot to a single pass.
fn bench_stateful(tokens: &BytesCowsAuto) {
    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let total_bytes: u64 = (&view).into_iter().map(|token| token.len() as u64).sum();

    measure(
        "stateful/stringzilla::Hasher",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(total_bytes)),
        || {
            let mut hasher = sz::Hasher::new(0);
            for token in &view {
                hasher.write(token);
            }
            black_box(hasher.finish());
        },
    );

    // Benchmark: SipHash via `std::DefaultHasher`
    let std_builder = std::collections::hash_map::RandomState::new();
    measure(
        "stateful/std::DefaultHasher",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(total_bytes)),
        || {
            let mut aggregate = std_builder.build_hasher();
            for token in &view {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
        },
    );

    // Benchmark: aHash
    let ahash_state = AHashState::with_seed(42);
    measure(
        "stateful/ahash::AHasher",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(total_bytes)),
        || {
            let mut aggregate = ahash_state.build_hasher();
            for token in &view {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
        },
    );

    // Benchmark: FoldHash
    let foldhash_state = foldhash::fast::RandomState::default();
    measure(
        "stateful/foldhash::FoldHasher",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(total_bytes)),
        || {
            let mut aggregate = foldhash_state.build_hasher();
            for token in &view {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
        },
    );

    measure(
        "stateful/crc32fast::Hasher",
        MeasureSpec::new(Unit::Bytes, WorkUnits::bytes(total_bytes)),
        || {
            let mut hasher = crc32fast::Hasher::new();
            for token in &view {
                hasher.update(token);
            }
            black_box(hasher.finalize());
        },
    );
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    let tape = resolve_dataset("hash").unwrap_nice();

    // One pass is one traversal of the working set; declared once, never accumulated.
    let work = WorkUnits::new(tape.len() as u64, tape.iter().map(|t| t.len() as u64).sum());
    log_timing_overhead();

    println!("# stateless");
    bench_stateless(work, &tape);

    println!("# stateful");
    bench_stateful(&tape);

    println!("# checksum");
    bench_checksum(work, &tape);

    finish();
}
