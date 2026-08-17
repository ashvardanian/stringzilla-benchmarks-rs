#![doc = r#"
# StringWars: String Hashing Benchmarks

This file contains benchmarks for various Rust hashing libraries, treating the inputs as binary strings without any
UTF-8 validity constraints. For accurate stats aggregation, on each iteration, the whole file is scanned. Be warned, for
large files, it may take a while!

The benchmarks are organized into three categories:

**Stateless Hashes** (hash each input independently):
- StringZilla `hash`
- Standard `Hash` implementation (SipHash)
- aHash
- xxHash (xxh3)
- FoldHash
- CRC32 (IEEE) via `crc32fast`
- MurmurHash32 via `murmurhash32`
- CityHash64 via `cityhash` (x86_64 & Clang only)

**Stateful Hashes** (incremental/streaming):
- StringZilla `Hasher`
- Standard `DefaultHasher` (SipHash)
- aHash `AHasher`
- FoldHash `FoldHasher`
- CRC32 via `crc32fast::Hasher`

**Checksum Hashes** (cryptographic and reference bounds):
- StringZilla `bytesum` (reference lower bound)
- Blake3 (cryptographic)
- SHA256 via `sha2` (cryptographic)
- SHA256 via `ring` (cryptographic)
- SHA256 via `stringzilla` (cryptographic, stateless and stateful)

## System Dependencies

Before running these benchmarks, ensure the following system packages are installed:

```sh
sudo apt install -y build-essential llvm-18-dev libclang-18-dev clang-18 # for Ubuntu/Debian
sudo dnf install -y gcc llvm-devel clang-devel # for RHEL/Fedora
brew install llvm clang # for macOS
```

## Usage Examples

The benchmarks use environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.
  - `file`: Process the entire file as a single token.
- `STRINGWARS_COLLISIONS`: Set to `1` or `true` to enable collision detection (disabled by default to avoid OOM on large
  datasets).
- `STRINGWARS_FILTER`: Regex pattern to filter which benchmarks to run (e.g., `sha` for SHA benchmarks,
  `stateless/.*hash` for stateless hashes).

To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=data/xlsum/xlsum.csv \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_hash --bench bench_hash
```

`lines` is the default mode here, and a corpus of article-length rows such as XLSum is the one to reach for.
Word-tokenized input makes every row shorter than a single 64-byte block, so the whole benchmark measures per-call overhead and the padding block, and never reaches the compression loop the backends actually differ in.

Note: `cityhash` is only compiled on x86_64 targets as it requires x86-specific instructions.
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

#[path = "../utils.rs"]
mod utils;
use utils::{
    get_env_bool, install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput,
    should_run, BenchBudget, ReportAs, ResultExt, WorkUnits,
};

/// Benchmarks one stateless hash that produces a `u64` result: runs `bench_each_token` for
/// throughput and, when unique tokens are available, prints the collision rate.
///
/// Exceptions left inline: `crc32fast::hash` (returns `u32`, closure shape differs) and
/// the `#[cfg(target_arch = "x86_64")]` `cityhash` block (cfg-gated, must stay inline).
fn bench_stateless_hash<HashFn: Fn(&[u8]) -> u64 + Copy>(
    name: &str,
    budget: &BenchBudget,
    slices: &[&[u8]],
    unique_tokens: &[&[u8]],
    hash_fn: HashFn,
) {
    bench_each_token(name, budget, slices, |token| {
        let _ = black_box(hash_fn(token));
    });
    if !unique_tokens.is_empty() && should_run(name) {
        print_collision_rate(unique_tokens, hash_fn);
    }
}

/// Time one stateless hash over the dataset by cycling tokens for the budget. The kernel
/// `hash_one` is called on one token per iteration; throughput is reported as bytes/s.
fn bench_each_token<HashOne: FnMut(&[u8])>(
    name: &str,
    budget: &BenchBudget,
    tokens: &[&[u8]],
    mut hash_one: HashOne,
) {
    let mut cursor = 0usize;
    measure_throughput(name, ReportAs::Bytes, budget, || {
        let token = tokens[cursor % tokens.len()];
        cursor += 1;
        hash_one(black_box(token));
        WorkUnits::new(1, token.len() as u64)
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
fn bench_stateless(budget: &BenchBudget, tokens: &BytesCowsAuto) {
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

    // Use BytesTape to colocate strings and reduce memory access overhead.
    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let slices: Vec<&[u8]> = (&view).into_iter().collect();

    // Benchmark: StringZilla
    bench_stateless_hash(
        "stateless/stringzilla::hash",
        budget,
        &slices,
        &unique_tokens,
        |token| sz::hash(token),
    );

    // Benchmark: SipHash via `std::DefaultHasher`
    let std_builder = std::collections::hash_map::RandomState::new();
    bench_stateless_hash(
        "stateless/std::DefaultHasher::hash_one",
        budget,
        &slices,
        &unique_tokens,
        |token| std_builder.hash_one(token),
    );

    // Benchmark: aHash
    let hash_builder = AHashState::with_seed(42);
    bench_stateless_hash(
        "stateless/ahash::hash_one",
        budget,
        &slices,
        &unique_tokens,
        |token| hash_builder.hash_one(token),
    );

    // Benchmark: xxHash
    bench_stateless_hash(
        "stateless/xxh3::xxh3_64",
        budget,
        &slices,
        &unique_tokens,
        xxh3_64,
    );

    // Benchmark: wyhash
    bench_stateless_hash(
        "stateless/wyhash::wyhash",
        budget,
        &slices,
        &unique_tokens,
        |token| wyhash(token, 42),
    );

    // Benchmark: FoldHash
    let foldhash_builder = foldhash::fast::RandomState::default();
    bench_stateless_hash(
        "stateless/foldhash::hash_one",
        budget,
        &slices,
        &unique_tokens,
        |token| foldhash_builder.hash_one(token),
    );

    // Benchmark: CRC32 — left inline because `crc32fast::hash` returns `u32`, so the
    // bench closure black-boxes a `u32` while the collision closure casts to `u64`;
    // the two-closure shapes differ from `bench_stateless_hash`.
    let name = "stateless/crc32fast::hash";
    bench_each_token(name, budget, &slices, |token| {
        let _ = black_box(crc32fast::hash(token));
    });
    if !unique_tokens.is_empty() && should_run(name) {
        print_collision_rate(&unique_tokens, |token_bytes| {
            crc32fast::hash(token_bytes) as u64
        });
    }

    // Benchmark: MurmurHash32 via `murmurhash32` (stateless) — cast to u64 at call site.
    bench_stateless_hash(
        "stateless/murmurhash32::murmurhash3",
        budget,
        &slices,
        &unique_tokens,
        |token| murmurhash32::murmurhash3(token) as u64,
    );

    // Benchmark: CityHash64 via `cityhash` (stateless, x86_64 only) — left inline because
    // the cfg gate cannot be placed on a single function call expression without a block.
    #[cfg(target_arch = "x86_64")]
    {
        let name = "stateless/cityhash::city_hash_64";
        bench_each_token(name, budget, &slices, |token| {
            let _ = black_box(cityhash::city_hash_64(token));
        });
        if !unique_tokens.is_empty() && should_run(name) {
            print_collision_rate(&unique_tokens, |token_bytes| {
                cityhash::city_hash_64(token_bytes)
            });
        }
    }

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Picks a stride coprime to `count`, near the golden-ratio fraction of it, so repeatedly stepping
/// by it visits every index once before repeating and samples the whole range early.
fn stride_coprime_to(count: usize) -> usize {
    fn greatest_common_divisor(first: usize, second: usize) -> usize {
        if second == 0 {
            first
        } else {
            greatest_common_divisor(second, first % second)
        }
    }

    if count <= 2 {
        return 1;
    }
    let mut stride = ((count as f64) * 0.6180339887) as usize | 1;
    while stride > 1 && greatest_common_divisor(stride, count) != 1 {
        stride -= 2;
    }
    stride.max(1)
}

/// How many messages the multi-state SHA256 hasher advances together, one per lane.
const SHA256_LANES: usize = 16;

/// How many consecutive batches one window covers.
const SHA256_BATCHES_PER_WINDOW: usize = 64;

/// Hands out the first token index of each successive batch, stepping whole windows on a stride
/// coprime to the window count and walking each window front to back.
///
/// The budget expires long before a full pass over a corpus of this size, and on length-ordered
/// tokens a forward walk would spend all of it among the shortest. The stride reaches every length,
/// while the run inside a window keeps the reads contiguous: striding one batch at a time would
/// scatter reads across the corpus and report the cache rather than the kernel.
struct BatchWalker {
    batches_count: usize,
    windows_count: usize,
    stride: usize,
    windows_visited: usize,
    batch_in_window: usize,
}

impl BatchWalker {
    fn new(tokens_count: usize) -> Self {
        let batches_count = tokens_count / SHA256_LANES;
        let windows_count = batches_count.div_ceil(SHA256_BATCHES_PER_WINDOW);
        Self {
            batches_count,
            windows_count,
            stride: stride_coprime_to(windows_count),
            windows_visited: 0,
            batch_in_window: 0,
        }
    }

    fn next_batch_start(&mut self) -> usize {
        if self.batch_in_window == SHA256_BATCHES_PER_WINDOW {
            self.batch_in_window = 0;
            self.windows_visited += 1;
        }
        let window_index = self.windows_visited.wrapping_mul(self.stride) % self.windows_count;
        let batch_index =
            (window_index * SHA256_BATCHES_PER_WINDOW + self.batch_in_window) % self.batches_count;
        self.batch_in_window += 1;
        batch_index * SHA256_LANES
    }
}

/// Times SHA256 over a whole batch at once, one lane per token. Hashing one message is a serial
/// dependency chain that bounds the per-token rows however wide the machine is; independent
/// messages compress in parallel lanes instead.
///
/// A batch retires when its longest member does, so the caller decides whether `tokens` arrives in
/// dataset order or length order, and the distance between those two rows is what length skew costs.
fn bench_lane_batches(name: &str, budget: &BenchBudget, tokens: &[&[u8]]) {
    if tokens.len() < SHA256_LANES {
        return;
    }
    let mut walker = BatchWalker::new(tokens.len());
    let initial_hasher = sz::Sha256::new();
    let mut lane_hashers = vec![initial_hasher; SHA256_LANES];
    let mut digests = vec![[0u8; sz::SHA256_DIGEST_LENGTH]; SHA256_LANES];
    measure_throughput(name, ReportAs::Bytes, budget, || {
        let first_token = walker.next_batch_start();
        let batch = &tokens[first_token..first_token + SHA256_LANES];

        lane_hashers.fill(initial_hasher);
        sz::sha256_multistate_update(&mut lane_hashers, black_box(batch)).unwrap();
        sz::sha256_multistate_digest(&lane_hashers, &mut digests).unwrap();

        let bytes: u64 = batch.iter().map(|token| token.len() as u64).sum();
        WorkUnits::new(SHA256_LANES as u64, bytes)
    });
}

/// Benchmarks cryptographic hashes, plus the reference bounds they are read against.
fn bench_crypto(budget: &BenchBudget, tokens: &BytesCowsAuto) {
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

    // Benchmark: StringZilla `bytesum`, the floor a 64-bit pass over the bytes cannot beat
    bench_each_token("reference/stringzilla::bytesum", budget, &slices, |token| {
        let _ = black_box(sz::bytesum(token));
    });

    // Benchmark: Blake3, a different construction with its own security argument
    let name = "reference/blake3::hash";
    bench_each_token(name, budget, &slices, |token| {
        let _ = black_box(blake3::hash(token));
    });
    if !unique_tokens.is_empty() && should_run(name) {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let hash = blake3::hash(token_bytes);
            let bytes = hash.as_bytes();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
    }

    // Benchmark: SHA256 via sha2
    let name = "crypto/sha2::Sha256";
    bench_each_token(name, budget, &slices, |token| {
        let mut hasher = Sha256::new();
        hasher.update(token);
        let _ = black_box(hasher.finalize());
    });
    if !unique_tokens.is_empty() && should_run(name) {
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

    // Benchmark: SHA256 via ring
    let name = "crypto/ring::SHA256";
    bench_each_token(name, budget, &slices, |token| {
        let _ = black_box(ring_digest::digest(&ring_digest::SHA256, token));
    });
    if !unique_tokens.is_empty() && should_run(name) {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let digest = ring_digest::digest(&ring_digest::SHA256, token_bytes);
            let bytes = digest.as_ref();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
    }

    // Benchmark: SHA256 via stringzilla
    let name = "crypto/stringzilla::Sha256";
    bench_each_token(name, budget, &slices, |token| {
        let _ = black_box(sz::Sha256::hash(token));
    });
    if !unique_tokens.is_empty() && should_run(name) {
        print_collision_rate(&unique_tokens, |token_bytes| {
            let digest = sz::Sha256::hash(token_bytes);
            u64::from_le_bytes([
                digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6],
                digest[7],
            ])
        });
    }

    // Benchmark: SHA256 over whole batches, in dataset order and in length order
    bench_lane_batches("crypto/stringzilla::Sha256s", budget, &slices);
    let mut sorted_by_length: Vec<&[u8]> = slices.clone();
    sorted_by_length.sort_unstable_by_key(|token| token.len());
    bench_lane_batches(
        "crypto/stringzilla::Sha256s<sorted>",
        budget,
        &sorted_by_length,
    );

    if !unique_tokens.is_empty() {
        println!();
    }
}

/// Benchmarks stateful hashes, streaming the whole dataset through one hasher per pass and
/// cycling passes for the budget. The per-call unit is one full streaming pass, so the deadline
/// check after each call bounds overshoot to a single pass.
fn bench_stateful(budget: &BenchBudget, tokens: &BytesCowsAuto) {
    let mut tokens_tape = BytesTape::<u64>::new();
    tokens_tape
        .extend(tokens.iter())
        .expect("Failed to create BytesTape");
    let view = tokens_tape.view();
    let total_bytes: u64 = (&view).into_iter().map(|token| token.len() as u64).sum();

    // Benchmark: StringZilla `Hasher`
    measure_throughput(
        "stateful/stringzilla::Hasher",
        ReportAs::Bytes,
        budget,
        || {
            let mut hasher = sz::Hasher::new(0);
            for token in &view {
                hasher.write(token);
            }
            black_box(hasher.finish());
            WorkUnits::bytes(total_bytes)
        },
    );

    // Benchmark: SipHash via `std::DefaultHasher`
    let std_builder = std::collections::hash_map::RandomState::new();
    measure_throughput(
        "stateful/std::DefaultHasher",
        ReportAs::Bytes,
        budget,
        || {
            let mut aggregate = std_builder.build_hasher();
            for token in &view {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
            WorkUnits::bytes(total_bytes)
        },
    );

    // Benchmark: aHash
    let ahash_state = AHashState::with_seed(42);
    measure_throughput("stateful/ahash::AHasher", ReportAs::Bytes, budget, || {
        let mut aggregate = ahash_state.build_hasher();
        for token in &view {
            aggregate.write(token);
        }
        black_box(aggregate.finish());
        WorkUnits::bytes(total_bytes)
    });

    // Benchmark: FoldHash
    let foldhash_state = foldhash::fast::RandomState::default();
    measure_throughput(
        "stateful/foldhash::FoldHasher",
        ReportAs::Bytes,
        budget,
        || {
            let mut aggregate = foldhash_state.build_hasher();
            for token in &view {
                aggregate.write(token);
            }
            black_box(aggregate.finish());
            WorkUnits::bytes(total_bytes)
        },
    );

    // Benchmark: CRC32
    measure_throughput(
        "stateful/crc32fast::Hasher",
        ReportAs::Bytes,
        budget,
        || {
            let mut hasher = crc32fast::Hasher::new();
            for token in &view {
                hasher.update(token);
            }
            black_box(hasher.finalize());
            WorkUnits::bytes(total_bytes)
        },
    );
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables.
    // Lines rather than words: a word is around five bytes, so every hash is one padding block and the
    // row measures call overhead instead of the compression function. Article-length lines from a corpus
    // like XLSum span several blocks, which is where the kernels differ.
    let tape = load_dataset("lines", "0", "data/xlsum/xlsum.csv").unwrap_nice();

    let budget = BenchBudget::from_env(2.0, 10.0);

    println!("# stateless");
    bench_stateless(&budget, &tape);

    println!("# stateful");
    bench_stateful(&budget, &tape);

    println!("# crypto");
    bench_crypto(&budget, &tape);
}
