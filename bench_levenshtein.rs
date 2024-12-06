use std::env;
use std::fs;

use criterion::{criterion_group, criterion_main, Criterion};
use rapidfuzz::distance::levenshtein;
use stringzilla::StringZilla;

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(1000)
        .warm_up_time(std::time::Duration::from_secs(10))
        .measurement_time(std::time::Duration::from_secs(120))
}

fn bench_levenshtein(c: &mut Criterion) {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_MODE").unwrap_or_else(|_| "lines".to_string());
    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");

    let bound_percent = env::var("STRINGWARS_ERROR_BOUND")
        .unwrap_or_else(|_| "15".to_string())
        .parse::<u64>()
        .expect("STRINGWARS_ERROR_BOUND must be a number");

    let max_pairs = env::var("STRINGWARS_MAX_PAIRS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100);

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

    let mut pairs: Vec<(&str, &str)> = units
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some((chunk[0], chunk[1]))
            } else {
                None
            }
        })
        .collect();

    if pairs.is_empty() {
        panic!("No pairs could be formed from the dataset.");
    }

    if pairs.len() > max_pairs {
        pairs.truncate(max_pairs);
    }

    let pair_bounds: Vec<usize> = pairs
        .iter()
        .map(|(a, b)| {
            let max_len = a.len().max(b.len());
            ((max_len as u64 * bound_percent) / 100) as usize
        })
        .collect();

    let mut g = c.benchmark_group("levenshtein");

    perform_levenshtein_benchmarks(&mut g, &pairs, &pair_bounds);

    g.finish();
}

fn perform_levenshtein_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    pairs: &[(&str, &str)],
    pair_bounds: &[usize],
) {
    // StringZilla, bytes-based, unbounded
    {
        let mut pair_index = 0;
        g.bench_function("stringzilla::levenshtein_bytes_unbounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let _distance = a.sz_edit_distance(b_str.as_bytes());
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }

    // StringZilla, bytes-based, bounded
    {
        let mut pair_index = 0;
        g.bench_function("stringzilla::levenshtein_bytes_bounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let bound = pair_bounds[pair_index];
                let _distance = a
                    .as_bytes()
                    .sz_edit_distance_bounded(b_str.as_bytes(), bound);
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }

    // StringZilla, UTF-8, unbounded
    {
        let mut pair_index = 0;
        g.bench_function("stringzilla::levenshtein_utf8_unbounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let _distance = a.as_bytes().sz_edit_distance_utf8(b_str.as_bytes());
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }

    // StringZilla, UTF-8, bounded
    {
        let mut pair_index = 0;
        g.bench_function("stringzilla::levenshtein_utf8_bounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let bound = pair_bounds[pair_index];
                let _distance = a
                    .as_bytes()
                    .sz_edit_distance_utf8_bounded(b_str.as_bytes(), bound);
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }

    // RapidFuzz, ASCII (bytes) unbounded
    {
        let mut pair_index = 0;
        g.bench_function("rapidfuzz::levenshtein_bytes_unbounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let _distance = levenshtein::distance(a.bytes(), b_str.bytes());
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }

    // RapidFuzz, ASCII (bytes) bounded
    {
        let mut pair_index = 0;
        g.bench_function("rapidfuzz::levenshtein_bytes_bounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let bound = pair_bounds[pair_index];
                let _distance = levenshtein::distance_with_args(
                    a.bytes(),
                    b_str.bytes(),
                    &levenshtein::Args::default().score_cutoff(bound),
                );
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }

    // RapidFuzz, UTF-8 (chars) unbounded
    {
        let mut pair_index = 0;
        g.bench_function("rapidfuzz::levenshtein_utf8_unbounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let _distance = levenshtein::distance(a.chars(), b_str.chars());
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }

    // RapidFuzz, UTF-8 (chars) bounded
    {
        let mut pair_index = 0;
        g.bench_function("rapidfuzz::levenshtein_utf8_bounded", |b| {
            b.iter(|| {
                let (a, b_str) = pairs[pair_index];
                let bound = pair_bounds[pair_index];
                let _distance = levenshtein::distance_with_args(
                    a.chars(),
                    b_str.chars(),
                    &levenshtein::Args::default().score_cutoff(bound),
                );
                pair_index = (pair_index + 1) % pairs.len();
            })
        });
    }
}

criterion_group! {
    name = bench_levenshtein_group;
    config = configure_bench();
    targets = bench_levenshtein
}
criterion_main!(bench_levenshtein_group);
