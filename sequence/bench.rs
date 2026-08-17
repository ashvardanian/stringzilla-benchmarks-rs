#![doc = r#"# StringWars: String Sequence Operations Benchmarks

This file benchmarks various libraries for processing string-identifiable collections. Including sorting arrays of
strings:

- StringZilla's `sz::argsort` (byte order and Unicode case-folded `uncased` order)
- The standard library's stable `sort_by_key` / `sort_by` (byte order, and `sz::utf8_uncased_order` folding comparator)
- Apache Arrow's `lexsort_to_indices`, and Polars `Series`/`DataFrame` sorts

Intersecting string collections, similar to "STRICT INNER JOIN" in SQL databases.

## Usage Example

The benchmarks use two environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.

To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:

```sh
RUSTFLAGS="-C target-cpu=native" \
    RAYON_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_sequence --bench bench_sequence
```
"#]

use std::hint::black_box;
use std::sync::Arc;

use stringtape::CharsCowsAuto;

use arrow::array::{ArrayRef, LargeStringArray};
use arrow::compute::{lexsort_to_indices, SortColumn};
use polars::prelude::*;
use stringzilla::sz;
use stringzilla::sz::ArgsortOptions;

#[path = "../utils.rs"]
mod utils;
use utils::{
    install_panic_hook, load_dataset, log_stringzilla_metadata, measure_throughput, reclaim_memory,
    should_run, BenchBudget, ReportAs, ResultExt, WorkUnits,
};

fn measure_argsort<Sort: FnMut(&[&str], &mut Vec<usize>)>(
    name: &str,
    budget: &BenchBudget,
    references: &[&str],
    comparisons_estimate: u64,
    total_bytes: u64,
    indices: &mut Vec<usize>,
    mut sort: Sort,
) {
    if !should_run(name) {
        return;
    }
    let count = references.len();
    measure_throughput(name, ReportAs::Comparisons, budget, || {
        indices.clear();
        indices.extend(0..count);
        sort(references, indices);
        black_box(&indices);
        WorkUnits::new(comparisons_estimate, total_bytes)
    });
}

fn bench_argsort(budget: &BenchBudget, unsorted: &CharsCowsAuto<'static>) {
    // For comparison-based sorting algorithms, we report throughput in terms of comparisons,
    // which is proportional to the number of elements in the array multiplied by the logarithm of
    // the number of elements. Each full sort accomplishes one batch of `comparisons_estimate`
    // comparisons; the secondary bytes/s metric uses the total UTF-8 size of the dataset.
    let count = unsorted.len();
    let comparisons_estimate = (count as f64 * (count as f64).log2()) as u64;
    let total_bytes: u64 = unsorted.iter().map(|token| token.len() as u64).sum();

    let mut reusable_indices = Vec::with_capacity(count);

    // StringZilla's sort is always stable, so every competitor is configured for a stable sort
    // too: Polars keeps `maintain_order: true` (its default leaves equal keys in arbitrary order).
    const POLARS_SORT_OPTIONS: SortOptions = SortOptions {
        descending: false,
        nulls_last: false,
        multithreaded: false,
        maintain_order: true,
        limit: None,
    };
    let polars_sort_multiple_options = SortMultipleOptions::default().with_maintain_order(true);

    const COLUMN_NAME: &str = "strings";

    // Collect StringTape into Vec<&str> for the four std/stringzilla argsort variants
    // (zero-copy, just references). The Arrow and Polars variants build their own
    // data structures and are handled separately below.
    let unsorted_references: Vec<&str> = unsorted.iter().collect();

    // Benchmark: StringZilla's argsort
    measure_argsort(
        "argsort/stringzilla::argsort",
        budget,
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
        &mut reusable_indices,
        |references, indices| {
            sz::argsort(references, indices, ArgsortOptions::default())
                .expect("StringZilla argsort failed");
        },
    );

    // Benchmark: StringZilla's case-insensitive (Unicode case-folding) argsort.
    // StringZilla orders by `sz_sequence_argsort_utf8_uncased` without materializing folded keys.
    measure_argsort(
        "argsort/stringzilla::argsort<uncased>",
        budget,
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
        &mut reusable_indices,
        |references, indices| {
            sz::argsort(references, indices, ArgsortOptions::default().uncased())
                .expect("StringZilla uncased argsort failed");
        },
    );

    // Benchmark: Standard library argsort using the stable `sort_by_key`. StringZilla's argsort
    // is always stable, so we compare against std's stable sort (not `sort_unstable_by_key`) to
    // keep the head-to-head honest — both preserve the input order of equal keys.
    measure_argsort(
        "argsort/std::sort_by_key",
        budget,
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
        &mut reusable_indices,
        |references, indices| {
            indices.sort_by_key(|&index| references[index]);
        },
    );

    // Benchmark: case-insensitive stable standard-library sort driven by StringZilla's own pairwise
    // Unicode case-folding comparator, `sz::utf8_uncased_order`. Holding the folding
    // implementation identical to `stringzilla::argsort_uncased` isolates the only remaining
    // variable — the sort algorithm (std's stable mergesort vs StringZilla's radix argsort).
    measure_argsort(
        "argsort/std::sort_by<uncased>",
        budget,
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
        &mut reusable_indices,
        |references, indices| {
            indices.sort_by(|&left, &right| {
                sz::utf8_uncased_order(references[left], references[right])
            });
        },
    );

    drop(unsorted_references);
    reclaim_memory();

    // Benchmark: Apache Arrow's `lexsort_to_indices`. Uses `LargeStringArray` because the
    // dataset's tape can exceed the 32-bit offset of the standard `StringArray` and would panic.
    let name = "argsort/arrow::lexsort_to_indices";
    if should_run(name) {
        let array = Arc::new(LargeStringArray::from_iter_values(unsorted.iter())) as ArrayRef;

        measure_throughput(name, ReportAs::Comparisons, budget, || {
            let column_to_sort = SortColumn {
                values: array.clone(),
                options: Some(arrow::compute::SortOptions {
                    descending: false,
                    nulls_first: true,
                }),
            };
            match lexsort_to_indices(&[column_to_sort], None) {
                Ok(indices) => black_box(indices),
                Err(error) => panic!("Arrow lexsort failed: {:?}", error),
            };
            WorkUnits::new(comparisons_estimate, total_bytes)
        });

        // Explicitly drop and reclaim memory (~4.7 GB)
        drop(array);
        reclaim_memory();
    }

    // Benchmark: Polars Series sort
    let name = "argsort/polars::Series::sort";
    if should_run(name) {
        // Polars can create Series from an iterator of &str
        let polars_series = Series::new(COLUMN_NAME.into(), unsorted.iter().collect::<Vec<&str>>());
        measure_throughput(name, ReportAs::Comparisons, budget, || {
            let sorted = polars_series.sort(POLARS_SORT_OPTIONS).unwrap();
            let _ = black_box(sorted);
            WorkUnits::new(comparisons_estimate, total_bytes)
        });
        drop(polars_series);
        reclaim_memory();
    }

    // Benchmark: Polars Series argsort (returning indices)
    let name = "argsort/polars::Series::arg_sort";
    if should_run(name) {
        let polars_series = Series::new(COLUMN_NAME.into(), unsorted.iter().collect::<Vec<&str>>());
        measure_throughput(name, ReportAs::Comparisons, budget, || {
            let indices = polars_series.arg_sort(POLARS_SORT_OPTIONS);
            black_box(indices);
            WorkUnits::new(comparisons_estimate, total_bytes)
        });
        drop(polars_series);
        reclaim_memory();
    }

    // Benchmark: Polars DataFrame sort
    let name = "argsort/polars::DataFrame::sort";
    if should_run(name) {
        // Lazy initialization: only create DataFrame when needed
        // No unnecessary clone - DataFrame takes ownership directly
        let polars_dataframe = DataFrame::new(
            unsorted.len(),
            vec![Series::new(COLUMN_NAME.into(), unsorted.iter().collect::<Vec<&str>>()).into()],
        )
        .unwrap();

        measure_throughput(name, ReportAs::Comparisons, budget, || {
            let sorted = polars_dataframe
                .sort([COLUMN_NAME], polars_sort_multiple_options.clone())
                .unwrap();
            black_box(sorted);
            WorkUnits::new(comparisons_estimate, total_bytes)
        });

        // Explicitly drop and reclaim memory (~4.7 GB)
        drop(polars_dataframe);
        reclaim_memory();
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables
    let tokens_bytes = load_dataset("lines", "0", "data/xlsum/xlsum.csv").unwrap_nice();
    // Leak BytesCowsAuto to get 'static lifetime, then cast to CharsCowsAuto for UTF-8 string benchmarks
    let tokens_bytes_static: &'static _ = Box::leak(Box::new(tokens_bytes));
    let tokens = tokens_bytes_static
        .as_chars()
        .expect("Dataset must be valid UTF-8");

    let budget = BenchBudget::from_env(5.0, 10.0);

    println!("# argsort");
    bench_argsort(&budget, &tokens);
}
