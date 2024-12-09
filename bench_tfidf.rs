use std::env;
use std::fs;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use memchr::memmem;
use stringzilla::StringZilla;

fn configure_bench() -> Criterion {
    Criterion::default()
        .sample_size(1000) // Test this many needles.
        .warm_up_time(std::time::Duration::from_secs(10)) // Let the CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(120)) // Actual measurement time.
}

fn bench_tfidf(c: &mut Criterion) {
    // Get the haystack path from the environment variable.
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let haystack_content = fs::read_to_string(&dataset_path).expect("Could not read haystack");

    // Tokenize the haystack content by white space.
    let needles: Vec<&str> = haystack_content.split_whitespace().collect();
    if needles.is_empty() {
        panic!("No tokens found in the haystack.");
    }

    let haystack = haystack_content.as_bytes();
    let haystack_length = haystack.len();

    // Benchmarks for forward search
    let mut g = c.benchmark_group("search-forward");
    g.throughput(Throughput::Bytes(haystack_length as u64));
    perform_forward_benchmarks(&mut g, &needles, haystack);
    g.finish();

    // Benchmarks for reverse search
    let mut g = c.benchmark_group("search-reverse");
    g.throughput(Throughput::Bytes(haystack_length as u64));
    perform_reverse_benchmarks(&mut g, &needles, haystack);
    g.finish();
}

...

criterion_group! {
    name = bench_tfidf_group;
    config = configure_bench();
    targets = bench_tfidf
}
criterion_main!(bench_tfidf_group);
