use std::env;
use std::fs;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hash::{BuildHasher, Hasher};

use stringzilla::sz::{checksum as sz_checksum, hash as sz_hash};
use stringzilla::StringZilla;

use ahash::AHasher;
use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3;
use xxhash_rust::xxh3::xxh3_64;

// Mode: "lines", "words", "file"
// STRINGWARS_MODE controls how we interpret the input data.
fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(1000) // Number of iterations per benchmark.
        .warm_up_time(std::time::Duration::from_secs(10)) // Let CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(120)) // Actual measurement time.
}

fn bench_hash(c: &mut Criterion) {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_MODE").unwrap_or_else(|_| "lines".to_string());

    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");
    let units: Vec<&str> = match mode.as_str() {
        "lines" => content.lines().collect(),
        "words" => content.split_whitespace().collect(),
        "file" => {
            // In "file" mode, treat the entire content as a single unit.
            vec![&content]
        }
        other => panic!(
            "Unknown STRINGWARS_MODE: {}. Use 'lines', 'words', or 'file'.",
            other
        ),
    };

    if units.is_empty() {
        panic!("No data found for hashing in the provided dataset.");
    }

    // Calculate total bytes processed for throughput reporting
    let total_bytes: usize = units.iter().map(|u| u.len()).sum();

    let mut g = c.benchmark_group("hash");
    g.throughput(Throughput::Bytes(total_bytes as u64));

    perform_hashing_benchmarks(&mut g, &units);

    g.finish();
}

fn perform_hashing_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    units: &[&str],
) {
    // Benchmark StringZilla checksums
    let mut index = 0;
    g.bench_function("stringzilla::checksum", |b| {
        b.iter(|| {
            let unit = units[index];
            let _hash = sz_checksum(unit.as_bytes());
            index = (index + 1) % units.len();
        })
    });

    // Benchmark StringZilla hashing
    let mut index = 0;
    g.bench_function("stringzilla::hash", |b| {
        b.iter(|| {
            let unit = units[index];
            let _hash = sz_hash(unit.as_bytes());
            index = (index + 1) % units.len();
        })
    });

    // Benchmark aHash
    let mut index = 0;
    let ahash_builder = ahash::RandomState::new();
    g.bench_function("aHash", |b| {
        b.iter(|| {
            let unit = units[index];
            let mut hasher = ahash_builder.build_hasher();
            hasher.write(unit.as_bytes());
            let _hash = hasher.finish();
            index = (index + 1) % units.len();
        })
    });

    // Benchmark xxHash (xxh3)
    let mut index = 0;
    g.bench_function("xxh3", |b| {
        b.iter(|| {
            let unit = units[index];
            let _hash = xxh3_64(unit.as_bytes());
            index = (index + 1) % units.len();
        })
    });
}

criterion_group! {
    name = bench_hash_group;
    config = configure_bench();
    targets = bench_hash
}
criterion_main!(bench_hash_group);
