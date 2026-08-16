#![doc = r#"# StringWars: Text Fingerprinting Benchmarks

This benchmark compares fingerprinting implementations using multi-width byte n-grams across CPU (1x, Nx) and GPU
architectures.

It evaluates:
- StringZilla Fingerprints (`szs::Fingerprints`) with configurable window widths [5, 9, 17, 33] bytes
- Custom MinHash with byte n-grams (baseline comparison)

Both use NDIM hash functions distributed evenly across the 4 window widths. Work is reported in bytes/sec (average bytes
per line × batch size).

## Usage Examples

Environment variables control dataset and processing:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_BATCH_PER_CORE`: Number of lines processed per core (default: 128). A CPU core is one core and a GPU
  streaming multiprocessor (SM) is one core, so the actual batch is auto-derived as `STRINGWARS_BATCH_PER_CORE * cores`:
  `cores` is 1 for the single-core variant, the logical core count for the multi-core variant, and the device's SM count
  for the GPU variant.
- `STRINGWARS_DATASET_LIMIT`: Read at most this many bytes of the dataset (`0` reads all).
- `STRINGWARS_NDIM`: Total hash functions distributed across n-gram widths (default: 256).

N-gram configuration:
- Fixed window widths: [5, 9, 17, 33] bytes for multi-scale fingerprinting
- Hash distribution: NDIM/4 hash functions per window width
- Example: NDIM=256 → 64 hashes each for 5-byte, 9-byte, 17-byte, 33-byte n-grams

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH_PER_CORE=256 \
    STRINGWARS_NDIM=256 \
    cargo run --release --features bench_fingerprints --bin bench_fingerprints
```

To run on a GPU-capable machine, enable the CUDA feature; the GPU batch is auto-derived from the device's
streaming-multiprocessor count:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH_PER_CORE=128 \
    STRINGWARS_NDIM=256 \
    STRINGWARS_FILTER=1gpu \
    cargo run --release --features "cuda bench_fingerprints" --bin bench_fingerprints
```
"#]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::convert::TryInto;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use forkunion as fu;
use stringtape::{BytesTape, BytesTapeView, CharsTapeView};

/// Logical core count for the multi-core device scope, probed once from a caller-owned topology.
///
/// ForkUnion spawns thread pools onto an immutable topology, so we construct it a single time in
/// `bench_fingerprints` rather than hiding it behind a global.
/// `STRINGWARS_CPU_CORES` overrides the count so a specific socket width can be reproduced.
fn resolve_core_count(topology: &fu::Topology) -> usize {
    std::env::var("STRINGWARS_CPU_CORES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&cores| cores > 0)
        .unwrap_or_else(|| topology.logical_cores_count())
}

use probabilistic_collections::similarity::{ByteGrams, MinHash};
use stringzilla::szs::{AnyBytesTape, DeviceScope, Fingerprints, UnifiedAlloc, UnifiedVec};

#[path = "../utils.rs"]
mod utils;
use utils::{
    auto_batch_size, get_env, get_env_or_default, gpu_multiprocessor_count, install_panic_hook,
    load_dataset, log_stringzilla_metadata, measure_throughput, BenchBudget, ReportAs, ResultExt,
    WorkUnits, COMPUTE_BOUND_SLICE,
};

// Fixed n-gram widths for multi-scale fingerprinting
const NGRAM_WIDTHS: [usize; 4] = [5, 9, 17, 33];

/// Per-core batch size for fingerprint benchmarks. 128 already saturates fingerprinting;
/// `auto_batch_size` scales it by each variant's core count.
const DEFAULT_BATCH_PER_CORE: usize = 128;

/// Calculate bit entropy (how well distributed the bits are) - generic
fn bit_entropy<T>(hash_matrix: &[Vec<T>]) -> f64
where
    T: Copy + Into<u64>,
{
    if hash_matrix.is_empty() || hash_matrix[0].is_empty() {
        return 0.0;
    }

    // Determine bit width based on type
    let bits_per_hash = std::mem::size_of::<T>() * 8;
    let mut bit_ones_count = vec![0usize; bits_per_hash]; // Count of 1s for each bit position
    let total_hash_values = hash_matrix.len() * hash_matrix[0].len();

    for document_hashes in hash_matrix {
        for &hash_value in document_hashes {
            let hash_as_u64: u64 = hash_value.into();
            for bit_position in 0..bits_per_hash {
                if (hash_as_u64 >> bit_position) & 1 == 1 {
                    bit_ones_count[bit_position] += 1;
                }
            }
        }
    }

    // Calculate entropy
    let mut total_entropy = 0.0;
    for ones_count in bit_ones_count {
        let probability_of_one = ones_count as f64 / total_hash_values as f64;
        if probability_of_one > 0.0 && probability_of_one < 1.0 {
            total_entropy -= probability_of_one * probability_of_one.log2()
                + (1.0 - probability_of_one) * (1.0 - probability_of_one).log2();
        }
    }

    total_entropy / bits_per_hash as f64 // Normalize to [0, 1]
}

/// Calculate collision rate (duplicate hash values) - generic
fn collision_rate<T>(hash_matrix: &[Vec<T>]) -> f64
where
    T: Copy + std::hash::Hash + Eq,
{
    if hash_matrix.is_empty() || hash_matrix[0].is_empty() {
        return 0.0;
    }

    let mut unique_hash_values = HashSet::new();
    let mut total_hash_count = 0;

    for document_hashes in hash_matrix {
        for &hash_value in document_hashes {
            unique_hash_values.insert(hash_value);
            total_hash_count += 1;
        }
    }

    1.0 - (unique_hash_values.len() as f64 / total_hash_count as f64)
}

fn tokens_tape_slice<'a>(
    bytes_view: &'a BytesTapeView<u64>,
    chars_view: &'a CharsTapeView<u64>,
    start_index: &mut usize,
    batch_size: usize,
    tokens_count: usize,
) -> (BytesTapeView<'a, u64>, CharsTapeView<'a, u64>, usize) {
    let current_start = *start_index;
    let end = std::cmp::min(current_start + batch_size, tokens_count);
    let actual_batch_size = end - current_start;

    let batch_bytes_view = bytes_view
        .subview(current_start, end)
        .unwrap_or_else(|error| {
            panic!(
                "Failed to create BytesTape subview from {} to {} (total tokens: {}): {}",
                current_start, end, tokens_count, error
            )
        });

    let batch_chars_view = chars_view
        .subview(current_start, end)
        .unwrap_or_else(|error| {
            panic!(
                "Failed to create CharsTape subview from {} to {} (total tokens: {}): {}",
                current_start, end, tokens_count, error
            )
        });

    *start_index = (current_start + actual_batch_size) % tokens_count;
    (batch_bytes_view, batch_chars_view, actual_batch_size)
}

/// Runs one `measure_throughput` block for `Fingerprints::compute_into`. Allocates
/// `UnifiedVec<u32>` buffers for min-hashes and min-counts (each of length
/// `batch_size * dimensions`), cycles through `tokens_count` tokens, and calls
/// `compute` for each timing iteration. After filling the buffers, `quality_callback`
/// (if `Some`) is called with `(actual_batch_size, min_hashes_slice)` so the caller
/// can populate quality-analysis matrices without duplicating the buffer allocation.
fn measure_fingerprints(
    name: &str,
    budget: &BenchBudget,
    bytes_view: &BytesTapeView<u64>,
    chars_view: &CharsTapeView<u64>,
    batch_size: usize,
    dimensions: usize,
    tokens_count: usize,
    hash_ops_per_token: u64,
    avg_token_bytes: u64,
    mut compute: impl FnMut(BytesTapeView<'_, u64>, usize, &mut [u32], &mut [u32]),
    mut quality_callback: impl FnMut(usize, &[u32]),
) {
    let mut min_hashes = UnifiedVec::<u32>::with_capacity_in(batch_size * dimensions, UnifiedAlloc);
    min_hashes.resize(batch_size * dimensions, 0);
    let mut min_counts = UnifiedVec::<u32>::with_capacity_in(batch_size * dimensions, UnifiedAlloc);
    min_counts.resize(batch_size * dimensions, 0);
    let mut start_index = 0usize;
    measure_throughput(name, ReportAs::Hashes, budget, || {
        let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
            bytes_view,
            chars_view,
            &mut start_index,
            batch_size,
            tokens_count,
        );
        compute(
            batch_bytes_view,
            dimensions,
            &mut min_hashes[..actual * dimensions],
            &mut min_counts[..actual * dimensions],
        );
        quality_callback(actual, &min_hashes[..actual * dimensions]);
        std::hint::black_box((
            &min_hashes[..actual * dimensions],
            &min_counts[..actual * dimensions],
        ));
        WorkUnits::new(
            actual as u64 * hash_ops_per_token,
            actual as u64 * avg_token_bytes,
        )
    });
}

fn bench_fingerprints(budget: &BenchBudget) {
    // Load dataset using unified loader
    let tape_bytes =
        load_dataset("lines", COMPUTE_BOUND_SLICE, "data/xlsum/xlsum.csv").unwrap_nice();
    let tape = tape_bytes
        .as_chars()
        .expect("Dataset must be valid UTF-8 for fingerprinting");

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

    // STRINGWARS_NDIM forces a single scale; otherwise sweep STRINGWARS_NDIM_SCALES.
    let scales: Vec<usize> = match get_env("STRINGWARS_NDIM") {
        Some(single) => vec![single
            .parse()
            .expect("STRINGWARS_NDIM must be a positive integer")],
        None => get_env_or_default("STRINGWARS_NDIM_SCALES", "64,128,256,512")
            .split(',')
            .map(|piece| {
                piece
                    .trim()
                    .parse()
                    .expect("STRINGWARS_NDIM_SCALES must be comma-separated positive integers")
            })
            .collect(),
    };

    // Create single BytesTape and populate it with all units
    let mut units_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    units_tape
        .extend(tape.iter().map(|token| token.as_bytes()))
        .unwrap_or_else(|error| panic!("Failed to extend BytesTape for fingerprinting: {}", error));

    // Create both byte and char views from single tape (zero-copy casting)
    let bytes_view = units_tape.view();
    let chars_view: CharsTapeView<u64> = units_tape.view().try_into().unwrap_or_else(|error| {
        panic!(
            "Failed to convert BytesTapeView to CharsTapeView: {}",
            error
        )
    });

    // Calculate average bytes per token for throughput reporting
    let total_documents = units_tape.len();
    let total_bytes: usize = tape.iter().map(|token| token.len()).sum();
    let avg_token_bytes: u64 = total_bytes as u64 / total_documents as u64;

    for dimensions in scales {
        if dimensions == 0 {
            panic!("Fingerprint dimensions must be greater than zero.");
        }
        println!("\nBenchmark configuration:");
        println!(
            "- Single-core batch size (for StringZilla): {}",
            batch_single_cpu
        );
        println!(
            "- {}-core batch size (for StringZilla): {}",
            num_cores, batch_multi_cpu
        );
        println!("- GPU batch size (for StringZilla): {}", batch_gpu);
        println!("- Fingerprint dimensions: {}", dimensions);
        println!("- N-gram widths: {:?} bytes", NGRAM_WIDTHS);
        println!(
            "- Hashes per width: {} (total NDIM: {})",
            dimensions / NGRAM_WIDTHS.len(),
            dimensions
        );

        // Hash operations per processed token: NDIM hash functions over the token's n-grams,
        // approximated as NDIM times the average token length. The harness reports hashes/s as
        // the primary metric and bytes/s as the secondary one, with no global ratio plumbing.
        let hash_ops_per_token: u64 = (dimensions as u64).saturating_mul(avg_token_bytes);

        // Pre-allocated matrices for quality analysis: N_docs × N_dims (algorithm-specific types)
        let mut serial_matrix = vec![vec![0u64; dimensions]; total_documents]; // u64 for serial MinHash
        let mut pc_matrix = vec![vec![0u64; dimensions]; total_documents]; // u64 for probabilistic_collections
        let mut sz_matrix = vec![vec![0u32; dimensions]; total_documents]; // u32 for StringZilla fingerprints

        // Track which documents have been processed for each implementation
        let mut serial_document_index = 0usize;
        let mut pc_document_index = 0usize;
        let mut sz_document_index = 0usize;

        println!("# minhash/ndim_{}", dimensions);

        // StringZilla engines and device scopes (1cpu, Ncpu, GPU?)
        let cpu_single = DeviceScope::cpu_cores(1).unwrap_or_else(|error| {
            panic!(
                "Failed to create single-core CPU device scope for fingerprinting: {}",
                error
            )
        });
        let cpu_parallel = DeviceScope::cpu_cores(num_cores).unwrap_or_else(|error| {
            panic!(
                "Failed to create {}-core CPU device scope for fingerprinting: {}",
                num_cores, error
            )
        });
        let maybe_gpu = DeviceScope::gpu_device(0);

        let sz_single = Fingerprints::builder()
            .ascii()
            .window_widths(&NGRAM_WIDTHS)
            .dimensions(dimensions)
            .build(&cpu_single)
            .unwrap_or_else(|error| {
                panic!(
                    "Failed to create single-core StringZilla fingerprinting engine: {}",
                    error
                )
            });
        let sz_parallel = Fingerprints::builder()
            .ascii()
            .window_widths(&NGRAM_WIDTHS)
            .dimensions(dimensions)
            .build(&cpu_parallel)
            .unwrap_or_else(|error| {
                panic!(
                    "Failed to create {}-core StringZilla fingerprinting engine: {}",
                    num_cores, error
                )
            });
        let maybe_sz_gpu = maybe_gpu.as_ref().ok().and_then(|gpu| {
            Fingerprints::builder()
                .ascii()
                .window_widths(&NGRAM_WIDTHS)
                .dimensions(dimensions)
                .build(gpu)
                .ok()
        });

        let tokens_count = total_documents;

        // StringZilla: 1x CPU. Result buffers sized to this variant's single-core batch.
        measure_fingerprints(
            "minhash/stringzillas::Fingerprints<1cpu>",
            budget,
            &bytes_view,
            &chars_view,
            batch_single_cpu,
            dimensions,
            tokens_count,
            hash_ops_per_token,
            avg_token_bytes,
            |batch_bytes_view, dimensions, min_hashes_slice, min_counts_slice| {
                sz_single
                    .compute_into(
                        &cpu_single,
                        AnyBytesTape::View64(batch_bytes_view),
                        dimensions,
                        min_hashes_slice,
                        min_counts_slice,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute StringZilla fingerprints on single CPU core: {}",
                            error
                        )
                    });
            },
            |_actual, _min_hashes_slice| {},
        );

        // StringZilla: Nx CPU. Result buffers sized to this variant's multi-core batch.
        measure_fingerprints(
            &format!("minhash/stringzillas::Fingerprints<{}cpu>", num_cores),
            budget,
            &bytes_view,
            &chars_view,
            batch_multi_cpu,
            dimensions,
            tokens_count,
            hash_ops_per_token,
            avg_token_bytes,
            |batch_bytes_view, dimensions, min_hashes_slice, min_counts_slice| {
                sz_parallel
                    .compute_into(
                        &cpu_parallel,
                        AnyBytesTape::View64(batch_bytes_view),
                        dimensions,
                        min_hashes_slice,
                        min_counts_slice,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "Failed to compute StringZilla fingerprints on {} CPU cores: {}",
                            num_cores, error
                        )
                    });
            },
            |actual, min_hashes_slice| {
                // Fill quality matrix - direct copy for StringZilla (u32 values)
                for document_index in 0..actual {
                    if sz_document_index < total_documents {
                        let start = document_index * dimensions;
                        let end = start + dimensions;
                        for (dimension_index, &hash_value) in
                            min_hashes_slice[start..end].iter().enumerate()
                        {
                            sz_matrix[sz_document_index][dimension_index] = hash_value;
                        }
                        sz_document_index += 1;
                    }
                }
            },
        );

        // StringZilla: 1x GPU (if available). Result buffers sized to the GPU batch.
        if let (Ok(gpu), Some(engine)) = (maybe_gpu.as_ref(), maybe_sz_gpu.as_ref()) {
            measure_fingerprints(
                "minhash/stringzillas::Fingerprints<1gpu>",
                budget,
                &bytes_view,
                &chars_view,
                batch_gpu,
                dimensions,
                tokens_count,
                hash_ops_per_token,
                avg_token_bytes,
                |batch_bytes_view, dimensions, min_hashes_slice, min_counts_slice| {
                    engine
                        .compute_into(
                            gpu,
                            AnyBytesTape::View64(batch_bytes_view),
                            dimensions,
                            min_hashes_slice,
                            min_counts_slice,
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "Failed to compute StringZilla fingerprints on GPU: {}",
                                error
                            )
                        });
                },
                |_actual, _min_hashes_slice| {},
            );
        }

        // Create separate MinHash instances for each n-gram width
        let hashes_per_width = dimensions / NGRAM_WIDTHS.len();
        // One MinHash per n-gram width, on the stack (no heap Vec).
        let minhashers: [MinHash<ByteGrams, _>; NGRAM_WIDTHS.len()] =
            core::array::from_fn(|_| MinHash::new(hashes_per_width));

        // pc::MinHash baseline (single-threaded scalar; uses the single-core batch).
        {
            // Pre-allocate output buffers outside the loop (reused across iterations)
            let mut out = Vec::with_capacity(batch_single_cpu);
            let mut combined_signature = Vec::with_capacity(dimensions);
            let mut start_index = 0usize;
            measure_throughput(
                "minhash/pc::MinHash<ByteGrams>",
                ReportAs::Hashes,
                budget,
                || {
                    let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
                        &bytes_view,
                        &chars_view,
                        &mut start_index,
                        batch_single_cpu,
                        tokens_count,
                    );

                    // Reuse output buffer (clear and reserve)
                    out.clear();
                    out.reserve(actual);

                    // Process each line with separate MinHash per width
                    for line_index in 0..batch_bytes_view.len() {
                        let line_bytes: &[u8] = &batch_bytes_view[line_index];
                        if !line_bytes.is_empty() {
                            // Clear and reuse combined signature buffer
                            combined_signature.clear();
                            combined_signature.reserve(dimensions);

                            // Process each n-gram width separately and concatenate results
                            for (width_index, &width) in NGRAM_WIDTHS.iter().enumerate() {
                                let iter = ByteGrams::new(line_bytes, width);
                                let partial_sig = minhashers[width_index].get_min_hashes(iter);
                                combined_signature.extend(partial_sig);
                            }

                            out.push(combined_signature.clone());

                            // Fill quality matrix - direct memcpy
                            if pc_document_index < total_documents
                                && combined_signature.len() == dimensions
                            {
                                pc_matrix[pc_document_index].copy_from_slice(&combined_signature);
                                pc_document_index += 1;
                            }
                        }
                    }

                    std::hint::black_box(&out);
                    WorkUnits::new(
                        actual as u64 * hash_ops_per_token,
                        actual as u64 * avg_token_bytes,
                    )
                },
            );
        }

        // Serial MinHash baseline with independent universal hash functions per dimension.
        {
            // Pre-construct hash parameters for independent universal hash functions
            // Each hash function uses: hash_i(x) = (a_i * hash(x) + b_i) mod mersenne_prime
            const MERSENNE_PRIME: u64 = (1u64 << 61) - 1; // Large prime for universal hashing

            // Generate independent hash function parameters
            let hash_params: Vec<(u64, u64)> = (0..dimensions)
                .map(|dimension_index| {
                    let multiplier = 2 * dimension_index as u64 + 1; // Odd for universal hashing
                    let offset = dimension_index as u64;
                    (multiplier, offset)
                })
                .collect();

            let mut start_index = 0usize;
            measure_throughput(
                "minhash/serial::MinHash<ByteGrams>",
                ReportAs::Hashes,
                budget,
                || {
                    let (batch_bytes_view, _batch_chars_view, actual) = tokens_tape_slice(
                        &bytes_view,
                        &chars_view,
                        &mut start_index,
                        batch_single_cpu,
                        tokens_count,
                    );

                    let mut out = Vec::with_capacity(actual);

                    // Process each line with serial MinHash
                    for line_index in 0..batch_bytes_view.len() {
                        let line_bytes: &[u8] = &batch_bytes_view[line_index];
                        if !line_bytes.is_empty() {
                            // Initialize minimum hash values for each hash function
                            let mut min_hashes = vec![u64::MAX; dimensions];

                            // Process all n-gram widths
                            for &width in &NGRAM_WIDTHS {
                                if line_bytes.len() >= width {
                                    // Generate n-grams of this width
                                    for window in line_bytes.windows(width) {
                                        // Compute base hash of the n-gram
                                        let mut hasher = DefaultHasher::new();
                                        window.hash(&mut hasher);
                                        let base_hash = hasher.finish();

                                        // Apply each independent hash function
                                        for (hash_index, &(multiplier, offset)) in
                                            hash_params.iter().enumerate()
                                        {
                                            let independent_hash = (multiplier
                                                .wrapping_mul(base_hash)
                                                .wrapping_add(offset))
                                                % MERSENNE_PRIME;
                                            min_hashes[hash_index] =
                                                min_hashes[hash_index].min(independent_hash);
                                        }
                                    }
                                }
                            }

                            out.push(min_hashes.clone());

                            // Fill quality matrix - direct memcpy
                            if serial_document_index < total_documents
                                && min_hashes.len() == dimensions
                            {
                                serial_matrix[serial_document_index].copy_from_slice(&min_hashes);
                                serial_document_index += 1;
                            }
                        }
                    }

                    std::hint::black_box(&out);
                    WorkUnits::new(
                        actual as u64 * hash_ops_per_token,
                        actual as u64 * avg_token_bytes,
                    )
                },
            );
        }

        // Compute and display quality metrics
        println!("\nHash Quality Analysis");
        println!(
            "Documents processed: PC={}, Serial={}, StringZilla={}",
            pc_document_index, serial_document_index, sz_document_index
        );

        if pc_document_index > 0 {
            let pc_slice = &pc_matrix[..pc_document_index];
            println!("\npc::MinHash<ByteGrams>:");
            println!("  Bit Entropy:  {:.4}", bit_entropy(pc_slice));
            println!("  Collision:    {:.4}%", collision_rate(pc_slice) * 100.0);
        }

        if serial_document_index > 0 {
            let serial_slice = &serial_matrix[..serial_document_index];
            println!("\nserial::MinHash<ByteGrams>:");
            println!("  Bit Entropy:  {:.4}", bit_entropy(serial_slice));
            println!(
                "  Collision:    {:.4}%",
                collision_rate(serial_slice) * 100.0
            );
        }

        if sz_document_index > 0 {
            let sz_slice = &sz_matrix[..sz_document_index];
            println!("\nszs::Fingerprints:");
            println!("  Bit Entropy:  {:.4}", bit_entropy(sz_slice));
            println!("  Collision:    {:.4}%", collision_rate(sz_slice) * 100.0);
        }
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    println!("Text Fingerprinting Benchmarks");
    println!("- szs::Fingerprints: CPU/GPU fingerprints with multi-width byte n-grams");
    println!("- pc::MinHash<ByteGrams>: MinHash with ByteGrams iterator");

    let budget = BenchBudget::from_env(1.0, 30.0);
    bench_fingerprints(&budget);
}
