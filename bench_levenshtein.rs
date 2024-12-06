use std::env;
use std::fs;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use rapidfuzz::distance::levenshtein;
use stringzilla::StringZilla;

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(1000) // Number of iterations for each benchmark.
        .warm_up_time(std::time::Duration::from_secs(10)) // Let the CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(120)) // Actual measurement time.
}

fn bench_levenshtein(c: &mut Criterion) {
    // Get the dataset path from the environment variable.
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_MODE").unwrap_or_else(|_| "lines".to_string());
    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");

    // Depending on the mode, split the input differently.
    let units: Vec<&str> = match mode.as_str() {
        "words" => content.split_whitespace().collect(),
        "lines" => content.lines().collect(),
        other => panic!(
            "Unknown STRINGWARS_MODE: {}. Use 'lines' or 'words'.",
            other
        ),
    };

    if units.len() < 2 {
        panic!("Dataset must contain at least two items for comparisons.");
    }

    // Pair up the units in twos.
    let pairs: Vec<(&str, &str)> = units
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some((chunk[0], chunk[1]))
            } else {
                None
            }
        })
        .collect();

    let data_size = pairs.len();

    let mut g = c.benchmark_group("levenshtein");
    g.throughput(Throughput::Elements(data_size as u64));

    perform_levenshtein_benchmarks(&mut g, &pairs);

    g.finish();
}

fn perform_levenshtein_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
) {
    // Benchmark for StringZilla Levenshtein distance
    let mut pair_index: usize = 0;
    g.bench_function("stringzilla::levenshtein", |b| {
        b.iter(|| {
            let (a, b) = pairs[pair_index];
            let _distance = a.sz_edit_distance(b);
            pair_index = (pair_index + 1) % pairs.len();
        })
    });

    // Benchmark for RapidFuzz Levenshtein distance
    let mut pair_index: usize = 0;
    g.bench_function("rapidfuzz::levenshtein", |b| {
        b.iter(|| {
            let (a, b) = pairs[pair_index];
            let _distance = levenshtein::distance(a.chars(), b.chars());
            pair_index = (pair_index + 1) % pairs.len();
        })
    });
}

criterion_group! {
    name = bench_levenshtein_group;
    config = configure_bench();
    targets = bench_levenshtein
}
criterion_main!(bench_levenshtein_group);
