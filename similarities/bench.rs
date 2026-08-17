#![doc = r#"# StringWars: String Similarity Benchmarks

This file benchmarks different libraries implementing string alignment and edit distance calculation, comparing
single-threaded, multi-threaded, and GPU-accelerated implementations across three gap cost models:

- **Uniform**: Classic Levenshtein distance with uniform substitution costs (match=0, mismatch=1)
- **Linear**: Needleman-Wunsch and Smith-Waterman with linear gap penalties (open_cost == extend_cost)
- **Affine**: Advanced alignment with different gap opening vs extension costs (open_cost != extend_cost)

The input file is tokenized into lines or words. The StringZilla engines evaluate a square `side x side`
cross-product: the first `side` tokens (queries) against the next `side` disjoint tokens (candidates), producing
a dense `side x side` similarity matrix. As most algorithms have quadratic complexity and use Dynamic Programming
techniques, their throughput is evaluated in the number of CUPS, or Cell Updates Per Second.

## Usage Examples

The benchmarks use environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.
  - `file`: Process the entire file as a single token.
- `STRINGWARS_DATASET_LIMIT`: Read at most this many bytes of the dataset (`0` reads all).
- `STRINGWARS_BATCH_PER_CORE`: Number of pairs processed per core (default: 256). A CPU core is one core and a GPU
  streaming multiprocessor (SM) is one core, so the actual batch is auto-derived as `STRINGWARS_BATCH_PER_CORE * cores`:
  `cores` is 1 for the single-core variant, the logical core count for the multi-core variant, and the device's SM count
  for the GPU variant. The square cross-product side is `round(sqrt(batch))`, so a `side x side` matrix holds about
  `STRINGWARS_BATCH_PER_CORE * cores` pairs.
- `STRINGWARS_TIME`: Wall-time budget per benchmark variant (seconds).
- `STRINGWARS_WARMUP`: Uncounted warm-up budget per variant (seconds).
- `STRINGWARS_FILTER`: Regex selecting which benchmark variants run.

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH_PER_CORE=128 \
    STRINGWARS_TOKENS=lines \
    cargo run --release --features bench_similarities --bin bench_similarities
```

To run on a GPU-capable machine, enable the CUDA feature; the GPU batch is auto-derived from the device's
streaming-multiprocessor count:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH_PER_CORE=128 \
    STRINGWARS_TOKENS=lines \
    STRINGWARS_FILTER=1gpu \
    cargo run --release --features "cuda bench_similarities" --bin bench_similarities
```
"#]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unnecessary_unwrap
)]
use core::convert::TryInto;

use forkunion as fu;
use stringtape::{BytesTape, BytesTapeView, CharsTapeView};

/// Logical core count for the multi-core device scope, probed once from a caller-owned topology.
///
/// ForkUnion spawns thread pools onto an immutable topology, so we construct it a single time and
/// thread the derived count through the benchmark helpers instead of hiding it behind a global.
/// `STRINGWARS_CPU_CORES` overrides the count so a specific socket width can be reproduced.
fn resolve_core_count(topology: &fu::Topology) -> usize {
    std::env::var("STRINGWARS_CPU_CORES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&cores| cores > 0)
        .unwrap_or_else(|| topology.logical_cores_count())
}

use bio::alignment::{distance as bio_distance, pairwise::Aligner};
use rapidfuzz::distance::levenshtein;
use stringzilla::szs::{
    AnyBytesTape, AnyCharsTape, LevenshteinDistances, LevenshteinDistancesUtf8,
    NeedlemanWunschScores, SmithWatermanScores, UnifiedAlloc, UnifiedMat,
};

#[path = "../utils.rs"]
mod utils;
use utils::{
    auto_batch_size, for_each_device_pass, gpu_multiprocessor_count, install_panic_hook,
    load_dataset, log_stringzilla_metadata, measure_throughput, report_skipped, BenchBudget,
    DeviceChoice, ReportAs, ResultExt, WorkUnits, COMPUTE_BOUND_SLICE,
};

/// Per-core batch size for similarity benchmarks. 256 is the measured GPU saturation knee
/// for short-word edit distance; `auto_batch_size` scales it by each variant's core count.
const DEFAULT_BATCH_PER_CORE: usize = 256;

/// Builds a substitution table for classic unary scoring: `match_cost` on the diagonal,
/// `mismatch_cost` everywhere else. Bytes are folded into 32 classes via `i % 32`, which keeps
/// the table compact; throughput (MCUPS) is invariant to the actual costs.
fn unary_class_costs(match_cost: i8, mismatch_cost: i8) -> ([u8; 256], [[i8; 32]; 32]) {
    let mut byte_to_class = [0u8; 256];
    for byte_value in 0..256 {
        byte_to_class[byte_value] = (byte_value % 32) as u8;
    }
    let mut class_costs = [[mismatch_cost; 32]; 32];
    for class_index in 0..32 {
        class_costs[class_index][class_index] = match_cost;
    }
    (byte_to_class, class_costs)
}

/// Computes the square cross-product side for a per-device pair `budget` and an available token
/// `tape_len`. A `side x side` cross-product holds ~`budget` pairs, clamped so queries `[0, side)`
/// and candidates `[side, 2*side)` are disjoint, i.e. `2 * side <= tape_len`.
fn crossproduct_side(budget: usize, tape_len: usize) -> usize {
    let target = ((budget as f64).sqrt().round() as usize).max(1);
    let max_side = tape_len / 2;
    target.min(max_side).max(1)
}

/// Runs one `measure_throughput` block for a bytes-tape cross-product engine that writes `usize`
/// results (e.g. `LevenshteinDistances`). The disjoint query slice `[0, side)` and candidate
/// slice `[side, 2*side)` are rebuilt from `full_view` each iteration (views are not `Clone`),
/// wrapped as `AnyBytesTape::View64`, and written into the pre-allocated `matrix` via `compute`.
fn measure_crossproduct_bytes_usize(
    name: &str,
    budget: &BenchBudget,
    full_view: &BytesTapeView<u64>,
    side: usize,
    total_cells: u64,
    total_bytes: u64,
    matrix: &mut UnifiedMat<usize>,
    mut compute: impl FnMut(AnyBytesTape<'_>, Option<AnyBytesTape<'_>>, &mut UnifiedMat<usize>),
) {
    measure_throughput(name, ReportAs::Cups, budget, || {
        let query_view = full_view
            .subview(0, side)
            .expect("Failed to create query subview");
        let candidate_view = full_view
            .subview(side, 2 * side)
            .expect("Failed to create candidate subview");
        compute(
            AnyBytesTape::View64(query_view),
            Some(AnyBytesTape::View64(candidate_view)),
            matrix,
        );
        std::hint::black_box(&matrix);
        WorkUnits::new(total_cells, total_bytes)
    });
}

/// Runs one `measure_throughput` block for a chars-tape cross-product engine that writes `usize`
/// results (e.g. `LevenshteinDistancesUtf8`). The disjoint query slice `[0, side)` and candidate
/// slice `[side, 2*side)` are rebuilt from `full_view` each iteration (views are not `Clone`),
/// wrapped as `AnyCharsTape::View64`, and written into the pre-allocated `matrix` via `compute`.
fn measure_crossproduct_chars_usize(
    name: &str,
    budget: &BenchBudget,
    full_view: &CharsTapeView<u64>,
    side: usize,
    total_cells: u64,
    total_bytes: u64,
    matrix: &mut UnifiedMat<usize>,
    mut compute: impl FnMut(AnyCharsTape<'_>, Option<AnyCharsTape<'_>>, &mut UnifiedMat<usize>),
) {
    measure_throughput(name, ReportAs::Cups, budget, || {
        let query_view = full_view
            .subview(0, side)
            .expect("Failed to create query subview");
        let candidate_view = full_view
            .subview(side, 2 * side)
            .expect("Failed to create candidate subview");
        compute(
            AnyCharsTape::View64(query_view),
            Some(AnyCharsTape::View64(candidate_view)),
            matrix,
        );
        std::hint::black_box(&matrix);
        WorkUnits::new(total_cells, total_bytes)
    });
}

/// Runs one `measure_throughput` block for a bytes-tape cross-product engine that writes `isize`
/// results (e.g. `NeedlemanWunschScores`, `SmithWatermanScores`). The disjoint query slice
/// `[0, side)` and candidate slice `[side, 2*side)` are rebuilt from `full_view` each iteration
/// (views are not `Clone`), wrapped as `AnyBytesTape::View64`, and written into the pre-allocated
/// `matrix` via `compute`.
fn measure_crossproduct_bytes_isize(
    name: &str,
    budget: &BenchBudget,
    full_view: &BytesTapeView<u64>,
    side: usize,
    total_cells: u64,
    total_bytes: u64,
    matrix: &mut UnifiedMat<isize>,
    mut compute: impl FnMut(AnyBytesTape<'_>, Option<AnyBytesTape<'_>>, &mut UnifiedMat<isize>),
) {
    measure_throughput(name, ReportAs::Cups, budget, || {
        let query_view = full_view
            .subview(0, side)
            .expect("Failed to create query subview");
        let candidate_view = full_view
            .subview(side, 2 * side)
            .expect("Failed to create candidate subview");
        compute(
            AnyBytesTape::View64(query_view),
            Some(AnyBytesTape::View64(candidate_view)),
            matrix,
        );
        std::hint::black_box(&matrix);
        WorkUnits::new(total_cells, total_bytes)
    });
}

/// Sums the byte lengths of a `[0, side)` query slice and `[side, 2*side)` candidate slice of a
/// bytes view, returning `(cross_product_cells, total_bytes)` where
/// `cells = sum_query_bytes * sum_candidate_bytes` and `total_bytes = sum_query_bytes + sum_candidate_bytes`.
fn crossproduct_metrics_bytes(full_view: &BytesTapeView<u64>, side: usize) -> (u64, u64) {
    let mut sum_query = 0u64;
    let mut sum_candidate = 0u64;
    for index in 0..side {
        sum_query += full_view[index].len() as u64;
        sum_candidate += full_view[side + index].len() as u64;
    }
    (sum_query * sum_candidate, sum_query + sum_candidate)
}

/// Sums the character lengths of a `[0, side)` query slice and `[side, 2*side)` candidate slice of
/// a chars view, returning `(cross_product_cells, total_bytes)` where
/// `cells = sum_query_chars * sum_candidate_chars`. The byte total mirrors the bytes view (UTF-8
/// byte length) for the secondary GB/s metric.
fn crossproduct_metrics_chars(full_view: &CharsTapeView<u64>, side: usize) -> (u64, u64) {
    let mut sum_query_chars = 0u64;
    let mut sum_candidate_chars = 0u64;
    let mut sum_query_bytes = 0u64;
    let mut sum_candidate_bytes = 0u64;
    for index in 0..side {
        let query = &full_view[index];
        let candidate = &full_view[side + index];
        sum_query_chars += query.chars().count() as u64;
        sum_candidate_chars += candidate.chars().count() as u64;
        sum_query_bytes += query.len() as u64;
        sum_candidate_bytes += candidate.len() as u64;
    }
    (
        sum_query_chars * sum_candidate_chars,
        sum_query_bytes + sum_candidate_bytes,
    )
}

/// Collects the `[0, side)` query slice of a bytes view into an owned `Vec<&[u8]>`.
fn bytes_query_vec<'a>(full_view: &'a BytesTapeView<u64>, side: usize) -> Vec<&'a [u8]> {
    (0..side).map(|index| &full_view[index]).collect()
}

/// Collects the `[side, 2*side)` candidate slice of a bytes view into an owned `Vec<&[u8]>`.
fn bytes_candidate_vec<'a>(full_view: &'a BytesTapeView<u64>, side: usize) -> Vec<&'a [u8]> {
    (0..side).map(|index| &full_view[side + index]).collect()
}

/// Collects the `[0, side)` query slice of a chars view into an owned `Vec<&str>`.
fn chars_query_vec<'a>(full_view: &'a CharsTapeView<u64>, side: usize) -> Vec<&'a str> {
    (0..side).map(|index| &full_view[index]).collect()
}

/// Collects the `[side, 2*side)` candidate slice of a chars view into an owned `Vec<&str>`.
fn chars_candidate_vec<'a>(full_view: &'a CharsTapeView<u64>, side: usize) -> Vec<&'a str> {
    (0..side).map(|index| &full_view[side + index]).collect()
}

fn bench_similarities(budget: &BenchBudget) {
    // Load dataset using unified loader
    let tape_bytes =
        load_dataset("words", COMPUTE_BOUND_SLICE, "data/acgt/acgt_1k.txt").unwrap_nice();
    let tape = tape_bytes
        .as_chars()
        .expect("Dataset must be valid UTF-8 for similarities");

    if tape.len() < 2 {
        panic!("Dataset must contain at least two items for comparisons.");
    }

    // Core-aware batch sizing: each variant scales `STRINGWARS_BATCH_PER_CORE` by its own core count.
    // A CPU core is one core; a GPU streaming multiprocessor (SM) is one core.
    let topology = fu::Topology::new().expect("Failed to probe CPU topology");
    let num_cores = resolve_core_count(&topology);
    let batch_single_cpu = auto_batch_size(1, DEFAULT_BATCH_PER_CORE);
    let batch_multi_cpu = auto_batch_size(num_cores, DEFAULT_BATCH_PER_CORE);
    let batch_gpu = auto_batch_size(
        gpu_multiprocessor_count(0).unwrap_or(64),
        DEFAULT_BATCH_PER_CORE,
    );

    // Create BytesTape and populate it with all tokens (the read is already bounded by STRINGWARS_DATASET_LIMIT)
    let mut units_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    units_tape
        .extend(tape.iter().map(|string| string.as_bytes()))
        .expect("Failed to extend BytesTape");

    // Full zero-copy bytes view over every token (the cross-product slices disjoint halves of it).
    let tape_bytes_view = units_tape
        .view()
        .try_into()
        .expect("Failed to create bytes view");
    // Full zero-copy chars view over every token (UTF-8 variant of the same tape).
    let chars_view: CharsTapeView<u64> = units_tape
        .view()
        .try_into()
        .expect("Failed to convert to CharsTapeView");

    // Each pass derives its own cross-product side from its core count; these three are the same
    // arithmetic, reported up front so the configuration block still names what every pass will run.
    let tape_len = units_tape.len();
    let side_single_cpu = crossproduct_side(batch_single_cpu, tape_len);
    let side_multi_cpu = crossproduct_side(batch_multi_cpu, tape_len);
    let side_gpu = crossproduct_side(batch_gpu, tape_len);

    // Log benchmark-specific configuration
    println!("Benchmark configuration:");
    println!(
        "- Single-core batch: {} ({}x{} cross-product)",
        batch_single_cpu, side_single_cpu, side_single_cpu
    );
    println!(
        "- {}-core batch: {} ({}x{} cross-product)",
        num_cores, batch_multi_cpu, side_multi_cpu, side_multi_cpu
    );
    println!(
        "- GPU batch: {} ({}x{} cross-product)",
        batch_gpu, side_gpu, side_gpu
    );
    println!("- Tokens available: {}", tape_len);
    println!();

    // Uniform cost benchmarks (classic Levenshtein: match=0, mismatch=1, open=1, extend=1)
    println!("# uniform");
    perform_uniform_benchmarks(budget, &tape_bytes_view, &chars_view, num_cores, tape_len);

    // Linear gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-2, extend=-2)
    println!("# linear");
    perform_score_benchmarks(
        "linear",
        budget,
        &tape_bytes_view,
        num_cores,
        tape_len,
        -2,
        -2,
    );

    // Affine gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-5, extend=-1)
    println!("# affine");
    perform_score_benchmarks(
        "affine",
        budget,
        &tape_bytes_view,
        num_cores,
        tape_len,
        -5,
        -1,
    );
}

/// Uniform cost benchmarks: Classic Levenshtein distance (match=0, mismatch=1, open=1, extend=1)
fn perform_uniform_benchmarks(
    budget: &BenchBudget,
    tape_bytes_view: &BytesTapeView<u64>,
    chars_view: &CharsTapeView<u64>,
    num_cores: usize,
    tape_len: usize,
) {
    for_each_device_pass(num_cores, |pass, scope_name, device, cores| {
        let side = crossproduct_side(auto_batch_size(cores, DEFAULT_BATCH_PER_CORE), tape_len);

        // Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if matches!(pass, DeviceChoice::OneCore) {
            uniform_baselines(budget, tape_bytes_view, chars_view, side);
        }

        // StringZilla byte-level Levenshtein distance (uniform costs: 0,1,1,1)
        let name = format!("uniform/stringzillas::LevenshteinDistances<{scope_name}>");
        match LevenshteinDistances::new(device, 0, 1, 1, 1) {
            Ok(engine) => {
                let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side);
                let queries = bytes_query_vec(tape_bytes_view, side);
                let candidates = bytes_candidate_vec(tape_bytes_view, side);
                match engine.compute(device, &queries, &candidates) {
                    Ok(mut matrix) => measure_crossproduct_bytes_usize(
                        &name,
                        budget,
                        tape_bytes_view,
                        side,
                        cells,
                        bytes,
                        &mut matrix,
                        |queries, candidates, matrix| {
                            engine
                                .compute_into(device, queries, candidates, matrix)
                                .unwrap_or_else(|error| {
                                    panic!("{name}: {error}");
                                });
                        },
                    ),
                    Err(error) => report_skipped(&name, error),
                }
            }
            Err(error) => report_skipped(&name, error),
        }

        // StringZilla UTF-8 Levenshtein distance. The engine may decline inputs beyond its supported
        // length, returning an error we skip on rather than aborting the suite.
        let name = format!("uniform/stringzillas::LevenshteinDistancesUtf8<{scope_name}>");
        match LevenshteinDistancesUtf8::new(device, 0, 1, 1, 1) {
            Ok(engine) => {
                let (cells, bytes) = crossproduct_metrics_chars(chars_view, side);
                let queries = chars_query_vec(chars_view, side);
                let candidates = chars_candidate_vec(chars_view, side);
                match engine.compute(device, &queries, &candidates) {
                    Ok(mut matrix) => measure_crossproduct_chars_usize(
                        &name,
                        budget,
                        chars_view,
                        side,
                        cells,
                        bytes,
                        &mut matrix,
                        |queries, candidates, matrix| {
                            engine
                                .compute_into(device, queries, candidates, matrix)
                                .unwrap_or_else(|error| {
                                    panic!("{name}: {error}");
                                });
                        },
                    ),
                    Err(error) => report_skipped(&name, error),
                }
            }
            Err(error) => report_skipped(&name, error),
        }
    });
}

/// The single-threaded uniform-cost rivals, run only in the single-core pass. They take no batch, so
/// each call scans one pair across the query/candidate diagonal of that pass's cross-product.
fn uniform_baselines(
    budget: &BenchBudget,
    tape_bytes_view: &BytesTapeView<u64>,
    chars_view: &CharsTapeView<u64>,
    baseline_side: usize,
) {
    {
        let mut pair_index = 0;
        measure_throughput(
            "uniform/rapidfuzz::levenshtein<Bytes,1cpu>",
            ReportAs::Cups,
            budget,
            || {
                let a_bytes = &tape_bytes_view[pair_index % baseline_side];
                let b_bytes = &tape_bytes_view[baseline_side + (pair_index % baseline_side)];
                let cells = (a_bytes.len() * b_bytes.len()) as u64;
                let bytes = (a_bytes.len() + b_bytes.len()) as u64;
                pair_index = (pair_index + 1) % baseline_side;
                std::hint::black_box(levenshtein::distance(
                    a_bytes.iter().copied(),
                    b_bytes.iter().copied(),
                ));
                WorkUnits::new(cells, bytes)
            },
        );
    }

    {
        let mut pair_index = 0;
        measure_throughput(
            "uniform/rapidfuzz::levenshtein<Chars,1cpu>",
            ReportAs::Cups,
            budget,
            || {
                let a_str = &chars_view[pair_index % baseline_side];
                let b_str = &chars_view[baseline_side + (pair_index % baseline_side)];
                let cells = (a_str.chars().count() * b_str.chars().count()) as u64;
                let bytes = (a_str.len() + b_str.len()) as u64;
                pair_index = (pair_index + 1) % baseline_side;
                std::hint::black_box(levenshtein::distance(a_str.chars(), b_str.chars()));
                WorkUnits::new(cells, bytes)
            },
        );
    }

    {
        let mut pair_index = 0;
        measure_throughput(
            "uniform/bio::levenshtein<1cpu>",
            ReportAs::Cups,
            budget,
            || {
                let a_bytes = &tape_bytes_view[pair_index % baseline_side];
                let b_bytes = &tape_bytes_view[baseline_side + (pair_index % baseline_side)];
                let cells = (a_bytes.len() * b_bytes.len()) as u64;
                let bytes = (a_bytes.len() + b_bytes.len()) as u64;
                pair_index = (pair_index + 1) % baseline_side;
                std::hint::black_box(bio_distance::levenshtein(a_bytes, b_bytes));
                WorkUnits::new(cells, bytes)
            },
        );
    }
}

/// Largest token byte length across the query/candidate slices of one pass, used to pre-size the
/// `bio` aligner capacity.
fn max_token_len(tape_bytes_view: &BytesTapeView<u64>, side: usize) -> usize {
    let mut max_len = 0usize;
    for index in 0..(2 * side) {
        let token_len = tape_bytes_view[index].len();
        if token_len > max_len {
            max_len = token_len;
        }
    }
    std::cmp::max(1, max_len)
}

/// NW/SW score benchmarks for one gap-cost group. The bio baselines and the StringZilla variants are
/// identical between the `linear` and `affine` groups; only the gap penalties and the label differ.
fn perform_score_benchmarks(
    group_name: &str,
    budget: &BenchBudget,
    tape_bytes_view: &BytesTapeView<u64>,
    num_cores: usize,
    tape_len: usize,
    open_cost: i8,
    extend_cost: i8,
) {
    // Unary scoring (match=2, mismatch=-1) folded into the 32-class table.
    let (byte_to_class, class_costs) = unary_class_costs(2, -1);

    for_each_device_pass(num_cores, |pass, scope_name, device, cores| {
        let side = crossproduct_side(auto_batch_size(cores, DEFAULT_BATCH_PER_CORE), tape_len);

        // Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if matches!(pass, DeviceChoice::OneCore) {
            bio_align_baselines(
                group_name,
                budget,
                tape_bytes_view,
                side,
                max_token_len(tape_bytes_view, side),
                open_cost,
                extend_cost,
            );
        }

        // Needleman-Wunsch (Global alignment)
        let name = format!("{group_name}/stringzillas::NeedlemanWunschScores<{scope_name}>");
        match NeedlemanWunschScores::new(
            device,
            &byte_to_class,
            &class_costs,
            open_cost,
            extend_cost,
        ) {
            Ok(engine) => {
                let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side);
                let queries = bytes_query_vec(tape_bytes_view, side);
                let candidates = bytes_candidate_vec(tape_bytes_view, side);
                match engine.compute(device, &queries, &candidates) {
                    Ok(mut matrix) => measure_crossproduct_bytes_isize(
                        &name,
                        budget,
                        tape_bytes_view,
                        side,
                        cells,
                        bytes,
                        &mut matrix,
                        |queries, candidates, matrix| {
                            engine
                                .compute_into(device, queries, candidates, matrix)
                                .unwrap_or_else(|error| {
                                    panic!("{name}: {error}");
                                });
                        },
                    ),
                    Err(error) => report_skipped(&name, error),
                }
            }
            Err(error) => report_skipped(&name, error),
        }

        // Smith-Waterman (Local alignment)
        let name = format!("{group_name}/stringzillas::SmithWatermanScores<{scope_name}>");
        match SmithWatermanScores::new(device, &byte_to_class, &class_costs, open_cost, extend_cost)
        {
            Ok(engine) => {
                let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side);
                let queries = bytes_query_vec(tape_bytes_view, side);
                let candidates = bytes_candidate_vec(tape_bytes_view, side);
                match engine.compute(device, &queries, &candidates) {
                    Ok(mut matrix) => measure_crossproduct_bytes_isize(
                        &name,
                        budget,
                        tape_bytes_view,
                        side,
                        cells,
                        bytes,
                        &mut matrix,
                        |queries, candidates, matrix| {
                            engine
                                .compute_into(device, queries, candidates, matrix)
                                .unwrap_or_else(|error| {
                                    panic!("{name}: {error}");
                                });
                        },
                    ),
                    Err(error) => report_skipped(&name, error),
                }
            }
            Err(error) => report_skipped(&name, error),
        }
    });
}

/// The single-threaded `bio` alignment rivals, run only in the single-core pass.
fn bio_align_baselines(
    group_name: &str,
    budget: &BenchBudget,
    tape_bytes_view: &BytesTapeView<u64>,
    baseline_side: usize,
    max_len: usize,
    open_cost: i8,
    extend_cost: i8,
) {
    {
        let mut aligner = Aligner::with_capacity(
            max_len,
            max_len,
            open_cost as i32,
            extend_cost as i32,
            |a: u8, b: u8| {
                if a == b {
                    2
                } else {
                    -1
                }
            },
        );
        let mut pair_index = 0;
        measure_throughput(
            &format!("{group_name}/bio::pairwise::global<1cpu>"),
            ReportAs::Cups,
            budget,
            || {
                let a_bytes = &tape_bytes_view[pair_index % baseline_side];
                let b_bytes = &tape_bytes_view[baseline_side + (pair_index % baseline_side)];
                let cells = (a_bytes.len() * b_bytes.len()) as u64;
                let bytes = (a_bytes.len() + b_bytes.len()) as u64;
                pair_index = (pair_index + 1) % baseline_side;
                std::hint::black_box(aligner.global(a_bytes, b_bytes).score);
                WorkUnits::new(cells, bytes)
            },
        );
    }

    {
        let mut aligner = Aligner::with_capacity(
            max_len,
            max_len,
            open_cost as i32,
            extend_cost as i32,
            |a: u8, b: u8| {
                if a == b {
                    2
                } else {
                    -1
                }
            },
        );
        let mut pair_index = 0;
        measure_throughput(
            &format!("{group_name}/bio::pairwise::local<1cpu>"),
            ReportAs::Cups,
            budget,
            || {
                let a_bytes = &tape_bytes_view[pair_index % baseline_side];
                let b_bytes = &tape_bytes_view[baseline_side + (pair_index % baseline_side)];
                let cells = (a_bytes.len() * b_bytes.len()) as u64;
                let bytes = (a_bytes.len() + b_bytes.len()) as u64;
                pair_index = (pair_index + 1) % baseline_side;
                std::hint::black_box(aligner.local(a_bytes, b_bytes).score);
                WorkUnits::new(cells, bytes)
            },
        );
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    let budget = BenchBudget::from_env(5.0, 30.0);
    bench_similarities(&budget);
}
