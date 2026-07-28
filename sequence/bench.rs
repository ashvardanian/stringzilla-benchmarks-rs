#![doc = r#"# StringWars: Sequence

Sorting benchmarks: argsort and full sorts over the token tape, reported in comparisons/s.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_sequence --bench bench_sequence
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

use stringwars::{
    finish, install_panic_hook, log_stringzilla_metadata, log_timing_overhead, measure,
    measure_with_setup, reclaim_memory, resolve_dataset, should_run, MeasureSpec, ResultExt, Unit,
    WorkUnits,
};

fn measure_argsort<Sort: FnMut(&[&str], &mut Vec<usize>)>(
    name: &str,
    references: &[&str],
    comparisons_estimate: u64,
    total_bytes: u64,
    mut sort: Sort,
) {
    let count = references.len();
    // Rebuilding the index vector is setup, not the kernel: sorting consumes its
    // input, so a second pass would sort already-sorted data. Excluding it here is
    // what the Python suite always did; Rust used to time it and the two argsort
    // rows were never comparable.
    measure_with_setup(
        name,
        MeasureSpec::new(
            Unit::Comparisons,
            WorkUnits::new(comparisons_estimate, total_bytes),
        ),
        || (0..count).collect::<Vec<usize>>(),
        |indices| {
            sort(references, indices);
            black_box(&indices);
        },
    );
}

fn bench_argsort(unsorted: &CharsCowsAuto<'static>) {
    // For comparison-based sorting algorithms, we report throughput in terms of comparisons,
    // which is proportional to the number of elements in the array multiplied by the logarithm of
    // the number of elements. Each full sort accomplishes one batch of `comparisons_estimate`
    // comparisons; the secondary bytes/s metric uses the total UTF-8 size of the dataset.
    let count = unsorted.len();
    let comparisons_estimate = (count as f64 * (count as f64).log2()) as u64;
    let total_bytes: u64 = unsorted.iter().map(|token| token.len() as u64).sum();

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

    measure_argsort(
        "argsort/stringzilla::argsort",
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
        |references, indices| {
            sz::argsort(references, indices, ArgsortOptions::default())
                .expect("StringZilla argsort failed");
        },
    );

    // Benchmark: StringZilla's case-insensitive (Unicode case-folding) argsort.
    // StringZilla orders by `sz_sequence_argsort_utf8_uncased` without materializing folded keys.
    measure_argsort(
        "argsort/stringzilla::argsort<uncased>",
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
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
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
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
        &unsorted_references,
        comparisons_estimate,
        total_bytes,
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
    if should_run("argsort/arrow::lexsort_to_indices") {
        let array = Arc::new(LargeStringArray::from_iter_values(unsorted.iter())) as ArrayRef;

        measure(
            "argsort/arrow::lexsort_to_indices",
            MeasureSpec::new(
                Unit::Comparisons,
                WorkUnits::new(comparisons_estimate, total_bytes),
            ),
            || {
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
            },
        );

        // Explicitly drop and reclaim memory (~4.7 GB)
        drop(array);
        reclaim_memory();
    }

    // Benchmark: Polars. All three rows read the same unsorted column — `sort` and `arg_sort`
    // take `&self` and the DataFrame takes an Arc-backed clone — so it is built once, and
    // straight from the tape rather than through a throwaway `Vec<&str>`.
    let run_series_sort = should_run("argsort/polars::Series::sort");
    let run_series_arg_sort = should_run("argsort/polars::Series::arg_sort");
    let run_dataframe_sort = should_run("argsort/polars::DataFrame::sort");
    if run_series_sort || run_series_arg_sort || run_dataframe_sort {
        let polars_series =
            StringChunked::from_iter_values(COLUMN_NAME.into(), unsorted.iter()).into_series();

        if run_series_sort {
            measure(
                "argsort/polars::Series::sort",
                MeasureSpec::new(
                    Unit::Comparisons,
                    WorkUnits::new(comparisons_estimate, total_bytes),
                ),
                || {
                    let sorted = polars_series.sort(POLARS_SORT_OPTIONS).unwrap();
                    let _ = black_box(sorted);
                },
            );
        }

        if run_series_arg_sort {
            measure(
                "argsort/polars::Series::arg_sort",
                MeasureSpec::new(
                    Unit::Comparisons,
                    WorkUnits::new(comparisons_estimate, total_bytes),
                ),
                || {
                    let indices = polars_series.arg_sort(POLARS_SORT_OPTIONS);
                    black_box(indices);
                },
            );
        }

        if run_dataframe_sort {
            let polars_dataframe =
                DataFrame::new(unsorted.len(), vec![polars_series.clone().into()]).unwrap();
            measure(
                "argsort/polars::DataFrame::sort",
                MeasureSpec::new(
                    Unit::Comparisons,
                    WorkUnits::new(comparisons_estimate, total_bytes),
                ),
                || {
                    let sorted = polars_dataframe
                        .sort([COLUMN_NAME], polars_sort_multiple_options.clone())
                        .unwrap();
                    black_box(sorted);
                },
            );
            drop(polars_dataframe);
        }

        // Explicitly drop and reclaim memory (~4.7 GB)
        drop(polars_series);
        reclaim_memory();
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    let tokens_bytes = resolve_dataset("sequence").unwrap_nice();
    log_timing_overhead();
    let tokens_bytes_static: &'static _ = Box::leak(Box::new(tokens_bytes));
    let tokens = tokens_bytes_static
        .as_chars()
        .expect("Dataset must be valid UTF-8");

    println!("# argsort");
    bench_argsort(&tokens);

    finish();
}
