#![doc = r#"# StringWars: Similarities

String-similarity benchmarks in CUPS: Levenshtein, Needleman-Wunsch, Smith-Waterman.

The engines evaluate a square `side x side` cross-product: the first `side` tokens
against the next `side` disjoint ones. The per-device pair budget is
`STRINGWARS_BATCH_PER_CORE * cores` and `side = round(sqrt(budget))`, so the matrix
holds about that many pairs.

- `STRINGWARS_CPU_CORES` overrides the multi-core scope width.

```sh
STRINGWARS_DATASET=README.md cargo bench --features bench_similarities --bench bench_similarities
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

use bio::alignment::{distance as bio_distance, pairwise::Aligner};
use rapidfuzz::distance::levenshtein;
use stringzilla::szs::{
    AnyBytesTape, AnyCharsTape, DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8,
    NeedlemanWunschScores, SmithWatermanScores, UnifiedAlloc, UnifiedMat,
};

use stringwars::{
    auto_batch_size, finish, gpu_multiprocessor_count, install_panic_hook,
    log_stringzilla_metadata, log_timing_overhead, measure, resolve_core_count, resolve_dataset,
    MeasureSpec, ResultExt, Unit, WorkUnits,
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
    full_view: &BytesTapeView<u64>,
    side: usize,
    total_cells: u64,
    total_bytes: u64,
    matrix: &mut UnifiedMat<usize>,
    mut compute: impl FnMut(AnyBytesTape<'_>, Option<AnyBytesTape<'_>>, &mut UnifiedMat<usize>),
) {
    measure(
        name,
        MeasureSpec::new(Unit::Cups, WorkUnits::new(total_cells, total_bytes)),
        || {
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
        },
    );
}

/// Runs one `measure_throughput` block for a chars-tape cross-product engine that writes `usize`
/// results (e.g. `LevenshteinDistancesUtf8`). The disjoint query slice `[0, side)` and candidate
/// slice `[side, 2*side)` are rebuilt from `full_view` each iteration (views are not `Clone`),
/// wrapped as `AnyCharsTape::View64`, and written into the pre-allocated `matrix` via `compute`.
fn measure_crossproduct_chars_usize(
    name: &str,
    full_view: &CharsTapeView<u64>,
    side: usize,
    total_cells: u64,
    total_bytes: u64,
    matrix: &mut UnifiedMat<usize>,
    mut compute: impl FnMut(AnyCharsTape<'_>, Option<AnyCharsTape<'_>>, &mut UnifiedMat<usize>),
) {
    measure(
        name,
        MeasureSpec::new(Unit::Cups, WorkUnits::new(total_cells, total_bytes)),
        || {
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
        },
    );
}

/// Runs one `measure_throughput` block for a bytes-tape cross-product engine that writes `isize`
/// results (e.g. `NeedlemanWunschScores`, `SmithWatermanScores`). The disjoint query slice
/// `[0, side)` and candidate slice `[side, 2*side)` are rebuilt from `full_view` each iteration
/// (views are not `Clone`), wrapped as `AnyBytesTape::View64`, and written into the pre-allocated
/// `matrix` via `compute`.
fn measure_crossproduct_bytes_isize(
    name: &str,
    full_view: &BytesTapeView<u64>,
    side: usize,
    total_cells: u64,
    total_bytes: u64,
    matrix: &mut UnifiedMat<isize>,
    mut compute: impl FnMut(AnyBytesTape<'_>, Option<AnyBytesTape<'_>>, &mut UnifiedMat<isize>),
) {
    measure(
        name,
        MeasureSpec::new(Unit::Cups, WorkUnits::new(total_cells, total_bytes)),
        || {
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
        },
    );
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

fn bench_similarities() {
    // Load dataset using unified loader
    let tape_bytes = resolve_dataset("similarities").unwrap_nice();
    log_timing_overhead();
    let tape = tape_bytes
        .as_chars()
        .expect("Dataset must be valid UTF-8 for similarities");

    if tape.len() < 2 {
        panic!("Dataset must contain at least two items for comparisons.");
    }

    // Core-aware batch sizing: each variant scales `STRINGWARS_BATCH_PER_CORE` by its own core count.
    // A CPU core is one core; a GPU streaming multiprocessor (SM) is one core.
    let topology = fu::Topology::new().expect("Failed to probe CPU topology");
    let num_cores = resolve_core_count(topology.logical_cores_count());
    let batch_single_cpu = auto_batch_size(1, DEFAULT_BATCH_PER_CORE);
    let batch_multi_cpu = auto_batch_size(num_cores, DEFAULT_BATCH_PER_CORE);
    let batch_gpu = auto_batch_size(
        gpu_multiprocessor_count(0).unwrap_or(64),
        DEFAULT_BATCH_PER_CORE,
    );

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
    perform_uniform_benchmarks(
        &tape_bytes_view,
        &chars_view,
        num_cores,
        side_single_cpu,
        side_multi_cpu,
        side_gpu,
    );

    // Linear gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-2, extend=-2)
    println!("# linear");
    perform_linear_benchmarks(
        &tape_bytes_view,
        num_cores,
        side_single_cpu,
        side_multi_cpu,
        side_gpu,
    );

    // Affine gap cost benchmarks (NW/SW: match=2, mismatch=-1, open=-5, extend=-1)
    println!("# affine");
    perform_affine_benchmarks(
        &tape_bytes_view,
        num_cores,
        side_single_cpu,
        side_multi_cpu,
        side_gpu,
    );
}

/// Uniform cost benchmarks: Classic Levenshtein distance (match=0, mismatch=1, open=1, extend=1)
fn perform_uniform_benchmarks(
    tape_bytes_view: &BytesTapeView<u64>,
    chars_view: &CharsTapeView<u64>,
    num_cores: usize,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
) {
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(num_cores).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    let lev_single = LevenshteinDistances::new(&cpu_single, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistances single");
    let lev_parallel = LevenshteinDistances::new(&cpu_parallel, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistances parallel");
    let lev_utf8_single = LevenshteinDistancesUtf8::new(&cpu_single, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistancesUtf8 single");
    let lev_utf8_parallel = LevenshteinDistancesUtf8::new(&cpu_parallel, 0, 1, 1, 1)
        .expect("Failed to create LevenshteinDistancesUtf8 parallel");
    let maybe_lev_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| LevenshteinDistances::new(gpu, 0, 1, 1, 1).ok());
    // GPU UTF-8 Levenshtein: the engine may decline inputs beyond its supported length, returning an error we skip
    // on rather than aborting the suite.
    let maybe_lev_utf8_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| LevenshteinDistancesUtf8::new(gpu, 0, 1, 1, 1).ok());

    // RapidFuzz baselines (no batching; scan one-by-one). One pair per call across the
    // query/candidate diagonal of the single-core cross-product.
    let baseline_side = side_single_cpu;
    // One pass scores every baseline pair, so the pair mixture is identical in every
    // sample rather than depending on how far the deadline happened to reach.
    let baseline_chars_work = (0..baseline_side).fold(WorkUnits::new(0, 0), |acc, index| {
        let a = &chars_view[index];
        let b = &chars_view[baseline_side + index];
        WorkUnits::new(
            acc.elements + (a.chars().count() * b.chars().count()) as u64,
            acc.bytes + (a.len() + b.len()) as u64,
        )
    });
    let baseline_pass_work = (0..baseline_side).fold(WorkUnits::new(0, 0), |acc, index| {
        let a = &tape_bytes_view[index];
        let b = &tape_bytes_view[baseline_side + index];
        WorkUnits::new(
            acc.elements + (a.len() * b.len()) as u64,
            acc.bytes + (a.len() + b.len()) as u64,
        )
    });
    {
        measure(
            "uniform/rapidfuzz::levenshtein<Bytes,1cpu>",
            MeasureSpec::new(Unit::Cups, baseline_pass_work),
            || {
                for index in 0..baseline_side {
                    let a_bytes = &tape_bytes_view[index];
                    let b_bytes = &tape_bytes_view[baseline_side + index];
                    std::hint::black_box(levenshtein::distance(
                        a_bytes.iter().copied(),
                        b_bytes.iter().copied(),
                    ));
                }
            },
        );
    }

    {
        measure(
            "uniform/rapidfuzz::levenshtein<Chars,1cpu>",
            MeasureSpec::new(Unit::Cups, baseline_chars_work),
            || {
                for index in 0..baseline_side {
                    let a_str = &chars_view[index];
                    let b_str = &chars_view[baseline_side + index];
                    std::hint::black_box(levenshtein::distance(a_str.chars(), b_str.chars()));
                }
            },
        );
    }

    {
        measure(
            "uniform/bio::levenshtein<1cpu>",
            MeasureSpec::new(Unit::Cups, baseline_pass_work),
            || {
                for index in 0..baseline_side {
                    let a_bytes = &tape_bytes_view[index];
                    let b_bytes = &tape_bytes_view[baseline_side + index];
                    std::hint::black_box(bio_distance::levenshtein(a_bytes, b_bytes));
                }
            },
        );
    }

    // StringZilla byte-level Levenshtein distance (uniform costs: 0,1,1,1)
    {
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_single_cpu);
        let queries = bytes_query_vec(tape_bytes_view, side_single_cpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_single_cpu);
        let mut matrix = lev_single
            .compute(&cpu_single, &queries, &candidates)
            .expect("Failed to allocate LevenshteinDistances matrix (single)");
        measure_crossproduct_bytes_usize(
            "uniform/stringzillas::LevenshteinDistances<1cpu>",
            tape_bytes_view,
            side_single_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                lev_single
                    .compute_into(&cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistances on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_multi_cpu);
        let queries = bytes_query_vec(tape_bytes_view, side_multi_cpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_multi_cpu);
        let mut matrix = lev_parallel
            .compute(&cpu_parallel, &queries, &candidates)
            .expect("Failed to allocate LevenshteinDistances matrix (parallel)");
        measure_crossproduct_bytes_usize(
            &format!(
                "uniform/stringzillas::LevenshteinDistances<{}cpu>",
                num_cores
            ),
            tape_bytes_view,
            side_multi_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                lev_parallel
                    .compute_into(&cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistances on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    // StringZilla UTF-8 Levenshtein Distance (uniform costs: 0,1,1,1)
    {
        let (cells, bytes) = crossproduct_metrics_chars(chars_view, side_single_cpu);
        let queries = chars_query_vec(chars_view, side_single_cpu);
        let candidates = chars_candidate_vec(chars_view, side_single_cpu);
        let mut matrix = lev_utf8_single
            .compute(&cpu_single, &queries, &candidates)
            .expect("Failed to allocate LevenshteinDistancesUtf8 matrix (single)");
        measure_crossproduct_chars_usize(
            "uniform/stringzillas::LevenshteinDistancesUtf8<1cpu>",
            chars_view,
            side_single_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                lev_utf8_single
                    .compute_into(&cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistancesUtf8 on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let (cells, bytes) = crossproduct_metrics_chars(chars_view, side_multi_cpu);
        let queries = chars_query_vec(chars_view, side_multi_cpu);
        let candidates = chars_candidate_vec(chars_view, side_multi_cpu);
        let mut matrix = lev_utf8_parallel
            .compute(&cpu_parallel, &queries, &candidates)
            .expect("Failed to allocate LevenshteinDistancesUtf8 matrix (parallel)");
        measure_crossproduct_chars_usize(
            &format!(
                "uniform/stringzillas::LevenshteinDistancesUtf8<{}cpu>",
                num_cores
            ),
            chars_view,
            side_multi_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                lev_utf8_parallel
                    .compute_into(&cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute LevenshteinDistancesUtf8 on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    if maybe_gpu.is_ok() && maybe_lev_gpu.is_some() {
        let gpu = maybe_gpu.as_ref().ok().unwrap();
        let engine = maybe_lev_gpu.as_ref().unwrap();
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_gpu);
        let queries = bytes_query_vec(tape_bytes_view, side_gpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_gpu);
        let mut matrix = engine
            .compute(gpu, &queries, &candidates)
            .expect("Failed to allocate LevenshteinDistances matrix (GPU)");
        measure_crossproduct_bytes_usize(
            "uniform/stringzillas::LevenshteinDistances<1gpu>",
            tape_bytes_view,
            side_gpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                engine
                    .compute_into(gpu, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
            },
        );
    }

    // GPU UTF-8 Levenshtein: the engine may decline inputs beyond its supported length, returning an error we skip on.
    if maybe_gpu.is_ok() && maybe_lev_utf8_gpu.is_some() {
        let gpu = maybe_gpu.as_ref().ok().unwrap();
        let engine = maybe_lev_utf8_gpu.as_ref().unwrap();
        let (cells, bytes) = crossproduct_metrics_chars(chars_view, side_gpu);
        let queries = chars_query_vec(chars_view, side_gpu);
        let candidates = chars_candidate_vec(chars_view, side_gpu);
        match engine.compute(gpu, &queries, &candidates) {
            Ok(mut matrix) => measure_crossproduct_chars_usize(
                "uniform/stringzillas::LevenshteinDistancesUtf8<1gpu>",
                chars_view,
                side_gpu,
                cells,
                bytes,
                &mut matrix,
                |queries, candidates, matrix| {
                    engine
                        .compute_into(gpu, queries, candidates, matrix)
                        .unwrap_or_else(|error| {
                            panic!("Failed to compute UTF-8 Levenshtein on GPU: {}", error)
                        });
                },
            ),
            Err(error) => eprintln!(
                "uniform/stringzillas::LevenshteinDistancesUtf8<1gpu>: SKIPPED ({})",
                error
            ),
        }
    }
}

/// Linear gap cost benchmarks: NW/SW with linear penalties (match=2, mismatch=-1, open=-2, extend=-2)
fn perform_linear_benchmarks(
    tape_bytes_view: &BytesTapeView<u64>,
    num_cores: usize,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
) {
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(num_cores).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // Unary scoring (match=2, mismatch=-1) folded into the 32-class table.
    let (byte_to_class, class_costs) = unary_class_costs(2, -1);

    let nw_single = NeedlemanWunschScores::new(&cpu_single, &byte_to_class, &class_costs, -2, -2)
        .expect("Failed to create NW single");
    let nw_parallel =
        NeedlemanWunschScores::new(&cpu_parallel, &byte_to_class, &class_costs, -2, -2)
            .expect("Failed to create NW parallel");
    let sw_single = SmithWatermanScores::new(&cpu_single, &byte_to_class, &class_costs, -2, -2)
        .expect("Failed to create SW single");
    let sw_parallel = SmithWatermanScores::new(&cpu_parallel, &byte_to_class, &class_costs, -2, -2)
        .expect("Failed to create SW parallel");
    let maybe_nw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| NeedlemanWunschScores::new(gpu, &byte_to_class, &class_costs, -2, -2).ok());
    let maybe_sw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| SmithWatermanScores::new(gpu, &byte_to_class, &class_costs, -2, -2).ok());

    let max_len = max_token_len(tape_bytes_view, side_single_cpu, side_multi_cpu, side_gpu);

    align_score_benchmarks(
        "linear",
        tape_bytes_view,
        &cpu_single,
        &cpu_parallel,
        &maybe_gpu,
        &nw_single,
        &nw_parallel,
        &maybe_nw_gpu,
        &sw_single,
        &sw_parallel,
        &maybe_sw_gpu,
        max_len,
        -2,
        -2,
        side_single_cpu,
        side_multi_cpu,
        side_gpu,
        num_cores,
    );
}

/// Largest token byte length across the union of the query/candidate slices of every variant,
/// used to pre-size the `bio` aligner capacity.
fn max_token_len(
    tape_bytes_view: &BytesTapeView<u64>,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
) -> usize {
    let widest_side = side_single_cpu.max(side_multi_cpu).max(side_gpu);
    let mut max_len = 0usize;
    for index in 0..(2 * widest_side) {
        let token_len = tape_bytes_view[index].len();
        if token_len > max_len {
            max_len = token_len;
        }
    }
    std::cmp::max(1, max_len)
}

/// Shared NW/SW score-benchmark body for the `linear` and `affine` gap-cost groups. The bio
/// baselines and the StringZilla CPU/GPU variants are identical between the two; only the gap
/// open/extend penalties and the group label differ.
fn align_score_benchmarks<GpuError>(
    group_name: &str,
    tape_bytes_view: &BytesTapeView<u64>,
    cpu_single: &DeviceScope,
    cpu_parallel: &DeviceScope,
    maybe_gpu: &Result<DeviceScope, GpuError>,
    nw_single: &NeedlemanWunschScores,
    nw_parallel: &NeedlemanWunschScores,
    maybe_nw_gpu: &Option<NeedlemanWunschScores>,
    sw_single: &SmithWatermanScores,
    sw_parallel: &SmithWatermanScores,
    maybe_sw_gpu: &Option<SmithWatermanScores>,
    max_len: usize,
    open_cost: i32,
    extend_cost: i32,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
    num_cores: usize,
) {
    let baseline_side = side_single_cpu;
    {
        let mut aligner =
            Aligner::with_capacity(max_len, max_len, open_cost, extend_cost, |a: u8, b: u8| {
                if a == b {
                    2
                } else {
                    -1
                }
            });
        // One pass scores every baseline pair, so the pair mixture is identical in
        // every sample instead of depending on how far the deadline reached.
        let (total_cells, total_bytes) = (0..baseline_side).fold((0u64, 0u64), |(c, b), index| {
            let a = &tape_bytes_view[index];
            let z = &tape_bytes_view[baseline_side + index];
            (
                c + (a.len() * z.len()) as u64,
                b + (a.len() + z.len()) as u64,
            )
        });
        measure(
            &format!("{group_name}/bio::pairwise::global<1cpu>"),
            MeasureSpec::new(Unit::Cups, WorkUnits::new(total_cells, total_bytes)),
            || {
                for index in 0..baseline_side {
                    let a_bytes = &tape_bytes_view[index];
                    let b_bytes = &tape_bytes_view[baseline_side + index];
                    std::hint::black_box(aligner.global(a_bytes, b_bytes).score);
                }
            },
        );
    }

    {
        let mut aligner =
            Aligner::with_capacity(max_len, max_len, open_cost, extend_cost, |a: u8, b: u8| {
                if a == b {
                    2
                } else {
                    -1
                }
            });
        // One pass scores every baseline pair, so the pair mixture is identical in
        // every sample instead of depending on how far the deadline reached.
        let (total_cells, total_bytes) = (0..baseline_side).fold((0u64, 0u64), |(c, b), index| {
            let a = &tape_bytes_view[index];
            let z = &tape_bytes_view[baseline_side + index];
            (
                c + (a.len() * z.len()) as u64,
                b + (a.len() + z.len()) as u64,
            )
        });
        measure(
            &format!("{group_name}/bio::pairwise::local<1cpu>"),
            MeasureSpec::new(Unit::Cups, WorkUnits::new(total_cells, total_bytes)),
            || {
                for index in 0..baseline_side {
                    let a_bytes = &tape_bytes_view[index];
                    let b_bytes = &tape_bytes_view[baseline_side + index];
                    std::hint::black_box(aligner.local(a_bytes, b_bytes).score);
                }
            },
        );
    }

    // Needleman-Wunsch (Global alignment)
    {
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_single_cpu);
        let queries = bytes_query_vec(tape_bytes_view, side_single_cpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_single_cpu);
        let mut matrix = nw_single
            .compute(cpu_single, &queries, &candidates)
            .expect("Failed to allocate NeedlemanWunschScores matrix (single)");
        measure_crossproduct_bytes_isize(
            "stringzillas::NeedlemanWunschScores<1cpu>",
            tape_bytes_view,
            side_single_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                nw_single
                    .compute_into(cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute NeedlemanWunschScores on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_multi_cpu);
        let queries = bytes_query_vec(tape_bytes_view, side_multi_cpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_multi_cpu);
        let mut matrix = nw_parallel
            .compute(cpu_parallel, &queries, &candidates)
            .expect("Failed to allocate NeedlemanWunschScores matrix (parallel)");
        measure_crossproduct_bytes_isize(
            &format!("stringzillas::NeedlemanWunschScores<{}cpu>", num_cores),
            tape_bytes_view,
            side_multi_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                nw_parallel
                    .compute_into(cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute NeedlemanWunschScores on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    if maybe_gpu.is_ok() && maybe_nw_gpu.is_some() {
        let gpu = maybe_gpu.as_ref().ok().unwrap();
        let engine = maybe_nw_gpu.as_ref().unwrap();
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_gpu);
        let queries = bytes_query_vec(tape_bytes_view, side_gpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_gpu);
        let mut matrix = engine
            .compute(gpu, &queries, &candidates)
            .expect("Failed to allocate NeedlemanWunschScores matrix (GPU)");
        measure_crossproduct_bytes_isize(
            "stringzillas::NeedlemanWunschScores<1gpu>",
            tape_bytes_view,
            side_gpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                engine
                    .compute_into(gpu, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
            },
        );
    }

    // Smith-Waterman (Local alignment)
    {
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_single_cpu);
        let queries = bytes_query_vec(tape_bytes_view, side_single_cpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_single_cpu);
        let mut matrix = sw_single
            .compute(cpu_single, &queries, &candidates)
            .expect("Failed to allocate SmithWatermanScores matrix (single)");
        measure_crossproduct_bytes_isize(
            "stringzillas::SmithWatermanScores<1cpu>",
            tape_bytes_view,
            side_single_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                sw_single
                    .compute_into(cpu_single, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute SmithWatermanScores on CPU (single-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    {
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_multi_cpu);
        let queries = bytes_query_vec(tape_bytes_view, side_multi_cpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_multi_cpu);
        let mut matrix = sw_parallel
            .compute(cpu_parallel, &queries, &candidates)
            .expect("Failed to allocate SmithWatermanScores matrix (parallel)");
        measure_crossproduct_bytes_isize(
            &format!("stringzillas::SmithWatermanScores<{}cpu>", num_cores),
            tape_bytes_view,
            side_multi_cpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                sw_parallel
                    .compute_into(cpu_parallel, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute SmithWatermanScores on CPU (multi-threaded): {}",
                            error
                        );
                    });
            },
        );
    }

    if maybe_gpu.is_ok() && maybe_sw_gpu.is_some() {
        let gpu = maybe_gpu.as_ref().ok().unwrap();
        let engine = maybe_sw_gpu.as_ref().unwrap();
        let (cells, bytes) = crossproduct_metrics_bytes(tape_bytes_view, side_gpu);
        let queries = bytes_query_vec(tape_bytes_view, side_gpu);
        let candidates = bytes_candidate_vec(tape_bytes_view, side_gpu);
        let mut matrix = engine
            .compute(gpu, &queries, &candidates)
            .expect("Failed to allocate SmithWatermanScores matrix (GPU)");
        measure_crossproduct_bytes_isize(
            "stringzillas::SmithWatermanScores<1gpu>",
            tape_bytes_view,
            side_gpu,
            cells,
            bytes,
            &mut matrix,
            |queries, candidates, matrix| {
                engine
                    .compute_into(gpu, queries, candidates, matrix)
                    .unwrap_or_else(|error| {
                        panic!("Failed to compute on GPU: {}. This may indicate GPU memory allocation issues with BytesTapeView.", error);
                    });
            },
        );
    }
}

/// Affine gap cost benchmarks: NW/SW with affine penalties (match=2, mismatch=-1, open=-5, extend=-1)
fn perform_affine_benchmarks(
    tape_bytes_view: &BytesTapeView<u64>,
    num_cores: usize,
    side_single_cpu: usize,
    side_multi_cpu: usize,
    side_gpu: usize,
) {
    let cpu_single = DeviceScope::cpu_cores(1).expect("Failed to create single-core device scope");
    let cpu_parallel =
        DeviceScope::cpu_cores(num_cores).expect("Failed to create multi-core device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    // Unary scoring (match=2, mismatch=-1) folded into the 32-class table.
    let (byte_to_class, class_costs) = unary_class_costs(2, -1);

    let nw_single = NeedlemanWunschScores::new(&cpu_single, &byte_to_class, &class_costs, -5, -1)
        .expect("Failed to create NW single");
    let nw_parallel =
        NeedlemanWunschScores::new(&cpu_parallel, &byte_to_class, &class_costs, -5, -1)
            .expect("Failed to create NW parallel");
    let sw_single = SmithWatermanScores::new(&cpu_single, &byte_to_class, &class_costs, -5, -1)
        .expect("Failed to create SW single");
    let sw_parallel = SmithWatermanScores::new(&cpu_parallel, &byte_to_class, &class_costs, -5, -1)
        .expect("Failed to create SW parallel");
    let maybe_nw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| NeedlemanWunschScores::new(gpu, &byte_to_class, &class_costs, -5, -1).ok());
    let maybe_sw_gpu = maybe_gpu
        .as_ref()
        .ok()
        .and_then(|gpu| SmithWatermanScores::new(gpu, &byte_to_class, &class_costs, -5, -1).ok());

    let max_len = max_token_len(tape_bytes_view, side_single_cpu, side_multi_cpu, side_gpu);

    align_score_benchmarks(
        "affine",
        tape_bytes_view,
        &cpu_single,
        &cpu_parallel,
        &maybe_gpu,
        &nw_single,
        &nw_parallel,
        &maybe_nw_gpu,
        &sw_single,
        &sw_parallel,
        &maybe_sw_gpu,
        max_len,
        -5,
        -1,
        side_single_cpu,
        side_multi_cpu,
        side_gpu,
        num_cores,
    );
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    bench_similarities();

    finish();
}
