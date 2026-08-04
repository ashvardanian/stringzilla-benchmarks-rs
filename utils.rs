use std::borrow::Cow;
use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fs;
use std::hint::black_box;
use std::panic;
use std::path::Path;
use std::str::FromStr;
use std::time::{Duration, Instant};
use stringtape::BytesCowsAuto;

/// Get an optional environment variable, returning None if not set.
#[allow(dead_code)]
pub fn get_env(name: &str) -> Option<String> {
    env::var(name).ok()
}

/// Get an environment variable with a default value.
#[allow(dead_code)]
pub fn get_env_or_default(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

/// Get an environment variable parsed to a type, with a default value.
/// Returns the default if the variable is not set or cannot be parsed.
#[allow(dead_code)]
pub fn get_env_parsed<T: FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Get an optional environment variable parsed to a type.
/// Returns None if the variable is not set or cannot be parsed.
#[allow(dead_code)]
pub fn get_env_parsed_opt<T: FromStr>(name: &str) -> Option<T> {
    env::var(name).ok().and_then(|value| value.parse().ok())
}

/// Get a boolean environment variable.
/// Accepts "1", "true", or "yes" (case-insensitive) as true values.
/// Returns false if not set or set to any other value.
#[allow(dead_code)]
pub fn get_env_bool(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Installs a custom panic hook that formats errors cleanly for CLI usage.
/// Call this at the start of main() before any potential panics.
#[allow(dead_code)]
pub fn install_panic_hook() {
    panic::set_hook(Box::new(|info| {
        let message = if let Some(payload_text) = info.payload().downcast_ref::<&str>() {
            payload_text.to_string()
        } else if let Some(payload_text) = info.payload().downcast_ref::<String>() {
            payload_text.clone()
        } else {
            "Unknown error".to_string()
        };

        eprintln!("\nError: {}", message);

        // Location only in debug/RUST_BACKTRACE mode, not for CLI users.
        if cfg!(debug_assertions) || get_env("RUST_BACKTRACE").is_some() {
            if let Some(location) = info.location() {
                eprintln!("  at {}:{}", location.file(), location.line());
            }
        }
    }));
}

/// Prints the StringZilla version, dispatch mode, and detected SIMD capabilities.
#[allow(dead_code)]
pub fn log_stringzilla_metadata() {
    let version = stringzilla::sz::version();
    println!(
        "StringZilla v{}.{}.{}",
        version.major, version.minor, version.patch
    );
    println!(
        "- uses dynamic dispatch: {}",
        stringzilla::sz::dynamic_dispatch()
    );
    println!(
        "- capabilities: {}",
        stringzilla::sz::capabilities().as_str()
    );
}

/// Extension trait for Result that provides clean panic-on-error semantics.
/// Uses Display formatting for errors (not Debug), which works well with
/// the custom panic hook for user-friendly CLI error messages.
#[allow(dead_code)]
pub trait ResultExt<T> {
    /// Unwrap the result or panic with the Display-formatted error.
    /// Equivalent to `.unwrap_or_else(|e| panic!("{}", e))` but cleaner.
    fn unwrap_nice(self) -> T;

    /// Unwrap the result or panic with a custom message and the error.
    /// Equivalent to `.unwrap_or_else(|e| panic!("{}: {}", msg, e))`.
    fn expect_nice(self, msg: &str) -> T;
}

impl<T, E: fmt::Display> ResultExt<T> for Result<T, E> {
    #[track_caller]
    fn unwrap_nice(self) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{}", error),
        }
    }

    #[track_caller]
    fn expect_nice(self, msg: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{}: {}", msg, error),
        }
    }
}

/// Extension trait for Option that provides clean panic-on-none semantics.
#[allow(dead_code)]
pub trait OptionExt<T> {
    /// Unwrap the option or panic with a custom message.
    fn expect_nice(self, msg: &str) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    #[track_caller]
    fn expect_nice(self, msg: &str) -> T {
        match self {
            Some(value) => value,
            None => panic!("{}", msg),
        }
    }
}

/// Errors that can occur when loading a dataset.
#[derive(Debug)]
pub enum DatasetError {
    /// The STRINGWARS_DATASET environment variable is not set.
    EnvVarNotSet,
    /// The dataset file does not exist.
    FileNotFound { path: String },
    /// Failed to read the dataset file.
    ReadError {
        path: String,
        source: std::io::Error,
    },
    /// The dataset file is empty.
    EmptyFile { path: String },
    /// No tokens were extracted from the dataset.
    NoTokens { path: String, mode: String },
    /// Unknown tokenization mode.
    UnknownMode { mode: String },
    /// Failed to create the token tape.
    TapeCreationFailed { path: String },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::EnvVarNotSet => {
                write!(
                    f,
                    "STRINGWARS_DATASET environment variable not set.\n\n\
                     Usage: STRINGWARS_DATASET=<file> STRINGWARS_TOKENS=<mode> cargo bench --features <bench> --bench <bench>\n\n\
                     Examples:\n  \
                       STRINGWARS_DATASET=README.md STRINGWARS_TOKENS=lines cargo bench --features bench_hash --bench bench_hash\n  \
                       STRINGWARS_DATASET=README.md STRINGWARS_TOKENS=words cargo bench --features bench_find --bench bench_find"
                )
            }
            DatasetError::FileNotFound { path } => {
                write!(
                    f,
                    "Dataset file not found: {}\n\n\
                     Please ensure the file exists. For Leipzig corpora, download with:\n  \
                       curl -fL https://downloads.wortschatz-leipzig.de/corpora/<corpus>.tar.gz \\\n    \
                         | tar --wildcards -xzf - --to-stdout '*-sentences.txt' | cut -f2 > {}",
                    path, path
                )
            }
            DatasetError::ReadError { path, source } => {
                write!(f, "Failed to read dataset '{}': {}", path, source)
            }
            DatasetError::EmptyFile { path } => {
                write!(
                    f,
                    "Dataset file is empty: {}\n\n\
                     Please provide a non-empty file.",
                    path
                )
            }
            DatasetError::NoTokens { path, mode } => {
                write!(
                    f,
                    "No tokens found in dataset '{}' with mode '{}'.\n\n\
                     The file exists but contains no valid tokens for this mode.\n\
                     Try a different STRINGWARS_TOKENS mode (lines, words, or file).",
                    path, mode
                )
            }
            DatasetError::UnknownMode { mode } => {
                write!(
                    f,
                    "Unknown STRINGWARS_TOKENS mode: '{}'\n\n\
                     Valid modes: 'lines', 'words', 'file'",
                    mode
                )
            }
            DatasetError::TapeCreationFailed { path } => {
                write!(f, "Failed to create token tape from '{}'", path)
            }
        }
    }
}

impl std::error::Error for DatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DatasetError::ReadError { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Forces the allocator to release memory back to the OS.
/// This is particularly useful after dropping large allocations in benchmarks.
#[allow(dead_code)]
#[cfg(target_os = "linux")]
#[inline]
pub fn reclaim_memory() {
    unsafe {
        libc::malloc_trim(0);
    }
}

#[allow(dead_code)]
#[cfg(not(target_os = "linux"))]
#[inline]
pub fn reclaim_memory() {
    // No-op on non-Linux platforms
}

/// Loads binary data from the file specified by the `STRINGWARS_DATASET` environment variable.
/// Uses StringTape to avoid allocating separate byte vectors for each token.
/// Returns BytesCowsAuto for memory-efficient byte slice storage.
/// Can be cast to CharsCowsAuto for UTF-8 string benchmarks (StringTape 2.2+).
/// Supports `STRINGWARS_MAX_TOKENS` to limit the number of tokens loaded.
/// Logs dataset statistics to stderr.
///
/// # Errors
/// Returns `DatasetError` if:
/// - `STRINGWARS_DATASET` environment variable is not set
/// - The dataset file does not exist or cannot be read
/// - The dataset file is empty
/// - No tokens can be extracted from the dataset
/// - Unknown tokenization mode is specified
#[allow(dead_code)]
pub fn load_dataset() -> Result<BytesCowsAuto<'static>, DatasetError> {
    load_dataset_with_default_mode("lines")
}

/// Like [`load_dataset`], but with a caller-chosen default token mode used when `STRINGWARS_TOKENS`
/// is unset. Each benchmark passes the granularity its kernel measures (e.g. `words` for hashing
/// and similarity, `lines` for normalization and fingerprinting); the env variable still overrides.
#[allow(dead_code)]
pub fn load_dataset_with_default_mode(
    default_mode: &str,
) -> Result<BytesCowsAuto<'static>, DatasetError> {
    let dataset_path = get_env("STRINGWARS_DATASET").ok_or(DatasetError::EnvVarNotSet)?;
    let mode = get_env_or_default("STRINGWARS_TOKENS", default_mode);
    let max_tokens: Option<usize> = get_env_parsed_opt("STRINGWARS_MAX_TOKENS");
    let unique = get_env_bool("STRINGWARS_UNIQUE");

    if let Some(max) = max_tokens {
        eprintln!("STRINGWARS_MAX_TOKENS: limiting to {} tokens", max);
    }
    if unique {
        eprintln!("STRINGWARS_UNIQUE: deduplicating tokens");
    }

    // Check if file exists before attempting to read
    if !Path::new(&dataset_path).exists() {
        return Err(DatasetError::FileNotFound { path: dataset_path });
    }

    // Read the file content
    let content = fs::read(&dataset_path).map_err(|error| DatasetError::ReadError {
        path: dataset_path.clone(),
        source: error,
    })?;

    // Check for empty file
    if content.is_empty() {
        return Err(DatasetError::EmptyFile { path: dataset_path });
    }

    // Leak the content to get 'static lifetime
    let content_static: &'static [u8] = Box::leak(content.into_boxed_slice());
    let limit = max_tokens.unwrap_or(usize::MAX);

    // Build BytesCowsAuto directly from iterator - it will own references to the leaked bytes
    let tape = match mode.as_str() {
        "lines" => {
            let iter = content_static
                .split(|&byte| byte == b'\n')
                .filter(|slice| !slice.is_empty())
                .take(limit);
            if unique {
                let mut seen: HashSet<&'static [u8]> = HashSet::new();
                let unique_tokens: Vec<&'static [u8]> =
                    iter.filter(|token| seen.insert(*token)).collect();
                BytesCowsAuto::from_iter_and_data(unique_tokens, Cow::Borrowed(content_static))
            } else {
                BytesCowsAuto::from_iter_and_data(iter, Cow::Borrowed(content_static))
            }
        }
        "words" => {
            let iter = content_static
                .split(|&byte| byte == b' ' || byte == b'\n')
                .filter(|slice| !slice.is_empty())
                .take(limit);
            if unique {
                let mut seen: HashSet<&'static [u8]> = HashSet::new();
                let unique_tokens: Vec<&'static [u8]> =
                    iter.filter(|token| seen.insert(*token)).collect();
                BytesCowsAuto::from_iter_and_data(unique_tokens, Cow::Borrowed(content_static))
            } else {
                BytesCowsAuto::from_iter_and_data(iter, Cow::Borrowed(content_static))
            }
        }
        "file" => {
            let iter = std::iter::once(content_static);
            BytesCowsAuto::from_iter_and_data(iter, Cow::Borrowed(content_static))
        }
        other => {
            return Err(DatasetError::UnknownMode {
                mode: other.to_string(),
            });
        }
    };

    let tape = tape.map_err(|_error| DatasetError::TapeCreationFailed {
        path: dataset_path.clone(),
    })?;

    // Check if we got any tokens
    if tape.is_empty() {
        return Err(DatasetError::NoTokens {
            path: dataset_path,
            mode,
        });
    }

    // Streaming statistics with log-scale histogram (O(1) memory)
    let count = tape.len();
    let total_bytes: usize = tape.iter().map(|slice: &[u8]| slice.len()).sum();
    let mean_len = total_bytes as f64 / count as f64;

    // Log-scale buckets: 0, 1, 2-3, 4-7, 8-15, 16-31, ... 32K-64K, 64K+
    let mut buckets = [0u64; 18];
    let mut min_len = usize::MAX;
    let mut max_len = 0;
    let mut variance_sum = 0.0;

    for token in tape.iter() {
        let len: usize = token.len();
        min_len = min_len.min(len);
        max_len = max_len.max(len);

        // Variance calculation
        let diff = len as f64 - mean_len;
        variance_sum += diff * diff;

        // Log-scale bucketing (powers of 2)
        let bucket = if len == 0 {
            0
        } else if len == 1 {
            1
        } else {
            // For len >= 2: bucket = log2(len) + 1
            // E.g., len=2-3 -> bucket 2, len=4-7 -> bucket 3, etc.
            ((len.ilog2() + 1) as usize).min(17)
        };
        buckets[bucket] += 1;
    }

    let std_dev: f64 = (variance_sum / count as f64).sqrt();

    eprintln!(
        "Dataset: {} tokens, {} bytes ({:.2} GB)\n  \
         Length: min {}, max {}, mean {:.1}, std {:.1}",
        format_number(count as u64),
        format_number(total_bytes as u64),
        total_bytes as f64 / 1e9,
        min_len,
        max_len,
        mean_len,
        std_dev
    );

    // Show distribution (only non-empty buckets)
    eprintln!("  Distribution:");
    let bucket_ranges = [
        "0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512-1K",
        "1K-2K", "2K-4K", "4K-8K", "8K-16K", "16K-32K", "32K-64K", "64K+",
    ];
    for (index, &bucket_count) in buckets.iter().enumerate() {
        if bucket_count > 0 {
            let percent = (bucket_count as f64 / count as f64) * 100.0;
            let label = if index < bucket_ranges.len() {
                bucket_ranges[index]
            } else {
                "64K+"
            };
            eprintln!("    {:>10} bytes: {:>6.2}%", label, percent);
        }
    }

    Ok(tape)
}

/// Format large numbers with thousand separators for readability
#[allow(dead_code)]
fn format_number(n: u64) -> String {
    let digits = n.to_string();
    let mut result = String::new();
    for (index, digit) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, digit);
    }
    result
}

// Perf profiling utilities
#[cfg(target_os = "linux")]
use perf_event::{
    events::CacheOp, events::CacheResult, events::Hardware, events::WhichCache, Builder, Counter,
};

/// Filter helper function to check if a benchmark should run based on STRINGWARS_FILTER env var
#[allow(dead_code)]
pub fn should_run(name: &str) -> bool {
    use std::sync::Once;
    static FILTER_INIT: Once = Once::new();

    if let Some(filter) = get_env("STRINGWARS_FILTER") {
        FILTER_INIT.call_once(|| {
            eprintln!("STRINGWARS_FILTER active: '{}'", filter);
        });

        if let Ok(regex) = regex::Regex::new(&filter) {
            let matches = regex.is_match(name);
            if !matches {
                eprintln!("  Skipping: {}", name);
            }
            matches
        } else {
            // Fallback to substring match if regex is invalid
            eprintln!(
                "Warning: Invalid regex pattern '{}', falling back to substring match",
                filter
            );
            name.contains(&filter)
        }
    } else {
        true
    }
}

// Simple SI scaling helper: returns the scaled value and its metric prefix.
#[allow(dead_code)]
fn scale_si(mut v: f64) -> (f64, &'static str) {
    if v >= 1_000_000_000.0 {
        v /= 1_000_000_000.0;
        (v, "G")
    } else if v >= 1_000_000.0 {
        v /= 1_000_000.0;
        (v, "M")
    } else if v >= 1_000.0 {
        v /= 1_000.0;
        (v, "k")
    } else {
        (v, "")
    }
}

#[allow(dead_code)]
fn format_seconds(value: f64) -> String {
    // value is seconds
    if value < 1e-6 {
        format!("{:.2} ns", value * 1e9)
    } else if value < 1e-3 {
        format!("{:.2} µs", value * 1e6)
    } else if value < 1.0 {
        format!("{:.2} ms", value * 1e3)
    } else {
        format!("{:.2} s", value)
    }
}

// Time-budgeted benchmark loop.
//
// Instead of a fixed sample count over the whole dataset, each variant runs for a fixed
// wall-time budget while cycling bounded items,
// and reports `work / elapsed`. Because we own the loop we also read hardware cycle and
// instruction counters around the measured region, surfacing exact cycles-per-byte and IPC
// instead of a batch-wide wall-clock estimate.

/// What one routine call accomplished, for dual-metric reporting.
/// `elements` counts pairs / hashes / comparisons / tokens (0 when not applicable);
/// `bytes` counts the bytes touched.
#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
pub struct WorkUnits {
    pub elements: u64,
    pub bytes: u64,
}

#[allow(dead_code)]
impl WorkUnits {
    /// Byte-only work (whole-buffer scans, transforms): `elements` stays 0.
    pub fn bytes(bytes: u64) -> Self {
        Self { elements: 0, bytes }
    }

    /// Both an element count and the bytes it spanned.
    pub fn new(elements: u64, bytes: u64) -> Self {
        Self { elements, bytes }
    }
}

/// Which primary unit a benchmark reports; bytes/s is always shown as the secondary metric.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum ReportAs {
    /// Bytes per second is the primary (and only) rate.
    Bytes,
    /// Cell updates per second (Needleman-Wunsch / Smith-Waterman / Levenshtein).
    Cups,
    /// Hashes per second (fingerprinting, hashing).
    Hashes,
    /// Hash-digest bits per second (multi-hash generation).
    Bits,
    /// Comparisons per second (sorting / sequence operations).
    Comparisons,
}

/// Wall-time budget for one benchmarked variant, read from the environment so the CLI can
/// override it (`STRINGWARS_WARMUP`, `STRINGWARS_TIME`, both in fractional seconds).
#[allow(dead_code)]
pub struct BenchBudget {
    pub warm_up: Duration,
    pub measure: Duration,
}

#[allow(dead_code)]
impl BenchBudget {
    /// Build the budget from the environment, falling back to the given defaults (seconds).
    pub fn from_env(default_warm_up_seconds: f64, default_measure_seconds: f64) -> Self {
        let warm_up_seconds = get_env_parsed("STRINGWARS_WARMUP", default_warm_up_seconds);
        let measure_seconds = get_env_parsed("STRINGWARS_TIME", default_measure_seconds);
        Self {
            warm_up: Duration::from_secs_f64(warm_up_seconds.max(0.0)),
            measure: Duration::from_secs_f64(measure_seconds.max(0.0)),
        }
    }
}

// Adaptive deadline cadence: a stride that starts at 1 and doubles toward this cap while a
// stride runs faster than the target, so fine-grained kernels amortize the clock and counter
// reads while a single slow call cannot overshoot the deadline by more than one item. Mirrors
// the Python `paced_items` design.
const PACING_STRIDE_CAP: u64 = 1024;
const PACING_TARGET_BETWEEN_CHECKS: Duration = Duration::from_millis(1);

/// Hardware cycle and instruction counters around the measured region (Linux only).
#[cfg(target_os = "linux")]
struct HardwareCounters {
    cycles: Option<Counter>,
    instructions: Option<Counter>,
}

#[cfg(target_os = "linux")]
impl HardwareCounters {
    fn start() -> Self {
        let build = |kind: Hardware| -> Option<Counter> {
            let mut counter = Builder::new().kind(kind).build().ok()?;
            counter.enable().ok()?;
            Some(counter)
        };
        Self {
            cycles: build(Hardware::CPU_CYCLES),
            instructions: build(Hardware::INSTRUCTIONS),
        }
    }

    fn stop(mut self) -> (Option<u64>, Option<u64>) {
        let read = |counter: &mut Option<Counter>| -> Option<u64> {
            let handle = counter.as_mut()?;
            handle.disable().ok()?;
            handle.read().ok()
        };
        (read(&mut self.cycles), read(&mut self.instructions))
    }
}

/// Accurate per-operation accounting accumulated across a measured region.
#[allow(dead_code)]
#[derive(Default)]
pub struct BenchStats {
    pub elapsed: Duration,
    pub calls: u64,
    pub elements: u64,
    pub bytes: u64,
    pub cycles: Option<u64>,
    pub instructions: Option<u64>,
    /// Per-checkpoint nanoseconds-per-call samples, for the latency distribution.
    latencies_ns: Vec<f64>,
}

#[allow(dead_code)]
impl BenchStats {
    /// p-quantile (0.0..=1.0) of the recorded per-call latency samples, in nanoseconds.
    fn latency_quantile(&self, quantile: f64) -> Option<f64> {
        if self.latencies_ns.is_empty() {
            return None;
        }
        let mut sorted = self.latencies_ns.clone();
        sorted.sort_by(|left, right| left.total_cmp(right));
        let rank = (quantile * (sorted.len() as f64 - 1.0)).round() as usize;
        Some(sorted[rank.min(sorted.len() - 1)])
    }

    /// Print the single canonical, column-aligned line for this variant. Columns are joined by
    /// " | " in a fixed order; columns that cannot be computed (no perf counters, no element
    /// count) are omitted, never reformatted, so Rust and Python stay layout-compatible.
    pub fn report(&self, name: &str, report: ReportAs) {
        let seconds = self.elapsed.as_secs_f64().max(1e-12);
        let mut columns: Vec<String> = Vec::new();

        // Primary rate.
        let elements_per_second = self.elements as f64 / seconds;
        let bytes_per_second = self.bytes as f64 / seconds;
        columns.push(match report {
            ReportAs::Bytes => format_byte_rate(bytes_per_second),
            ReportAs::Cups => format_si_rate(elements_per_second, "CUPS", false),
            ReportAs::Hashes => format_si_rate(elements_per_second, "hashes/s", true),
            ReportAs::Bits => format_si_rate(elements_per_second, "bits/s", true),
            ReportAs::Comparisons => format_si_rate(elements_per_second, "cmp/s", true),
        });

        // Secondary bytes/s, unless the primary already is bytes/s.
        if !matches!(report, ReportAs::Bytes) && self.bytes > 0 {
            columns.push(format_byte_rate(bytes_per_second));
        }

        // Exact cycles-per-byte and IPC from hardware counters (Linux only).
        if let (Some(cycles), true) = (self.cycles, self.bytes > 0) {
            columns.push(format!("{:.2} cyc/B", cycles as f64 / self.bytes as f64));
        }
        if let (Some(cycles), Some(instructions)) = (self.cycles, self.instructions) {
            if cycles > 0 {
                columns.push(format!("IPC {:.2}", instructions as f64 / cycles as f64));
            }
        }

        // Latency distribution.
        if let (Some(p50), Some(p99)) = (self.latency_quantile(0.5), self.latency_quantile(0.99)) {
            columns.push(format!(
                "p50 {} p99 {}",
                format_seconds(p50 / 1e9),
                format_seconds(p99 / 1e9)
            ));
        }

        println!("{:<42} {}", name, columns.join(" | "));
    }
}

/// Render a bytes-per-second rate as `<value> <prefix>B/s` (decimal SI, 2 decimals).
#[allow(dead_code)]
fn format_byte_rate(bytes_per_second: f64) -> String {
    let (value, prefix) = scale_si(bytes_per_second);
    format!("{:.2} {}B/s", value, prefix)
}

/// Render an SI rate as `<value> <prefix><unit>` (e.g. `1.24 GCUPS`). When `space_before_unit`
/// is set, a space separates the prefix from a word unit (`1.24 G hashes/s`).
#[allow(dead_code)]
fn format_si_rate(rate: f64, unit: &str, space_before_unit: bool) -> String {
    let (value, prefix) = scale_si(rate);
    if prefix.is_empty() {
        format!("{:.2} {}", value, unit)
    } else if space_before_unit {
        format!("{:.2} {} {}", value, prefix, unit)
    } else {
        format!("{:.2} {}{}", value, prefix, unit)
    }
}

/// Run `routine` cyclically until the measurement deadline (after an uncounted warm-up),
/// summing the work it reports and recording statistics, then print the canonical line and
/// return the stats. Honors `STRINGWARS_FILTER` via `should_run`; a filtered-out variant does
/// no work and prints nothing.
#[allow(dead_code)]
pub fn measure_throughput<Routine: FnMut() -> WorkUnits>(
    name: &str,
    report: ReportAs,
    budget: &BenchBudget,
    mut routine: Routine,
) -> BenchStats {
    if !should_run(name) {
        return BenchStats::default();
    }

    // Warm-up: run the kernel without recording so caches and frequency settle.
    if !budget.warm_up.is_zero() {
        let warm_up_start = Instant::now();
        while warm_up_start.elapsed() < budget.warm_up {
            let _ = black_box(routine());
        }
    }

    let mut elements = 0u64;
    let mut bytes = 0u64;
    let mut calls = 0u64;
    let mut latencies_ns: Vec<f64> = Vec::new();

    #[cfg(target_os = "linux")]
    let counters = HardwareCounters::start();

    let start = Instant::now();
    let deadline = start + budget.measure;
    let mut stride = 1u64;
    let mut countdown = 1u64;
    let mut last_check = start;
    let mut calls_since_check = 0u64;

    loop {
        let work = black_box(routine());
        elements += work.elements;
        bytes += work.bytes;
        calls += 1;
        calls_since_check += 1;
        countdown -= 1;
        if countdown != 0 {
            continue;
        }

        let now = Instant::now();
        let block = now - last_check;
        if calls_since_check > 0 {
            latencies_ns.push(block.as_nanos() as f64 / calls_since_check as f64);
        }
        if now >= deadline {
            break;
        }
        if block < PACING_TARGET_BETWEEN_CHECKS && stride < PACING_STRIDE_CAP {
            stride = (stride * 2).min(PACING_STRIDE_CAP);
        }
        last_check = now;
        calls_since_check = 0;
        countdown = stride;
    }

    let elapsed = start.elapsed();

    #[cfg(target_os = "linux")]
    let (cycles, instructions) = counters.stop();
    #[cfg(not(target_os = "linux"))]
    let (cycles, instructions): (Option<u64>, Option<u64>) = (None, None);

    let stats = BenchStats {
        elapsed,
        calls,
        elements,
        bytes,
        cycles,
        instructions,
        latencies_ns,
    };
    stats.report(name, report);
    stats
}

/// Items processed per core — one CPU core, or on the GPU one streaming multiprocessor (SM).
/// "Core" here means an SM, not an individual warp or CUDA core. `default_base` is the bench's own
/// per-core default (the right value differs by kernel — short-string similarity saturates the GPU
/// at a different batch than document fingerprinting); `STRINGWARS_BATCH_PER_CORE` overrides it.
#[allow(dead_code)]
fn items_per_core(default_base: usize) -> usize {
    get_env_parsed("STRINGWARS_BATCH_PER_CORE", default_base).max(1)
}

/// Batch size for a backend with `cores` parallel cores, scaling the bench's `default_base` by the
/// hardware. A CPU core counts as one core and a GPU streaming multiprocessor counts as one core,
/// so the batch scales automatically instead of a fixed CPU/GPU multiplier: a 1-core scope gets
/// `items_per_core`, an N-core scope `N * items_per_core`, and a GPU `SMs * items_per_core`.
#[allow(dead_code)]
pub fn auto_batch_size(cores: usize, default_base: usize) -> usize {
    items_per_core(default_base)
        .saturating_mul(cores.max(1))
        .max(1)
}

/// Number of streaming multiprocessors on the given CUDA device, queried from the CUDA runtime.
/// Each SM is counted as one core for batch sizing (an SM, not a warp or an individual CUDA core).
/// Returns None when CUDA is unavailable or the query fails, so callers fall back to a default
/// core count. The attribute id 16 is `cudaDevAttrMultiProcessorCount`.
#[allow(dead_code)]
#[cfg(feature = "cuda")]
pub fn gpu_multiprocessor_count(device_index: i32) -> Option<usize> {
    extern "C" {
        fn cudaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> i32;
    }
    const MULTIPROCESSOR_COUNT_ATTRIBUTE: i32 = 16;
    let mut count: i32 = 0;
    let status =
        unsafe { cudaDeviceGetAttribute(&mut count, MULTIPROCESSOR_COUNT_ATTRIBUTE, device_index) };
    (status == 0 && count > 0).then_some(count as usize)
}

/// Without the CUDA feature there is no device to query, so callers use their fallback core count.
#[allow(dead_code)]
#[cfg(not(feature = "cuda"))]
pub fn gpu_multiprocessor_count(_device_index: i32) -> Option<usize> {
    None
}

/// RAII guard that profiles a code section using perf events on Linux.
/// On non-Linux platforms, this is a zero-cost abstraction.
#[allow(dead_code)]
#[cfg(target_os = "linux")]
pub struct PerfSection {
    name: &'static str,
    // Core performance
    cycles: Option<Counter>,
    instructions: Option<Counter>,
    // Stall analysis (ARM-specific)
    stall_frontend: Option<Counter>,
    stall_backend: Option<Counter>,
    stall_backend_mem: Option<Counter>,
    // Memory hierarchy
    l1d_cache_misses: Option<Counter>,
    l1d_cache_accesses: Option<Counter>,
    dtlb_misses: Option<Counter>,
    l2_cache_misses: Option<Counter>,
}

#[cfg(target_os = "linux")]
impl PerfSection {
    /// Creates a new perf section with comprehensive counter collection.
    /// Collects: cycles, instructions, stalls (frontend/backend), cache, and TLB metrics.
    #[allow(dead_code)]
    pub fn new(name: &'static str) -> Self {
        // Core performance counters
        let cycles = Self::build_counter(Hardware::CPU_CYCLES);
        let instructions = Self::build_counter(Hardware::INSTRUCTIONS);

        // Stall counters (standard hardware events)
        let stall_frontend = Self::build_counter(Hardware::STALLED_CYCLES_FRONTEND);
        let stall_backend = Self::build_counter(Hardware::STALLED_CYCLES_BACKEND);

        // L1 data cache
        let l1d_cache_misses =
            Self::build_cache_counter(WhichCache::L1D, CacheOp::READ, CacheResult::MISS);
        let l1d_cache_accesses =
            Self::build_cache_counter(WhichCache::L1D, CacheOp::READ, CacheResult::ACCESS);

        // DTLB (data TLB)
        let dtlb_misses =
            Self::build_cache_counter(WhichCache::DTLB, CacheOp::READ, CacheResult::MISS);

        // L2/LLC cache
        let l2_cache_misses =
            Self::build_cache_counter(WhichCache::LL, CacheOp::READ, CacheResult::MISS);

        Self {
            name,
            cycles,
            instructions,
            stall_frontend,
            stall_backend,
            stall_backend_mem: None, // Not available via standard events
            l1d_cache_misses,
            l1d_cache_accesses,
            dtlb_misses,
            l2_cache_misses,
        }
    }

    /// Creates a minimal perf section that only tracks core performance (cycles + instructions).
    /// Use this for lower overhead when detailed metrics aren't needed.
    #[allow(dead_code)]
    pub fn minimal(name: &'static str) -> Self {
        let cycles = Self::build_counter(Hardware::CPU_CYCLES);
        let instructions = Self::build_counter(Hardware::INSTRUCTIONS);

        Self {
            name,
            cycles,
            instructions,
            stall_frontend: None,
            stall_backend: None,
            stall_backend_mem: None,
            l1d_cache_misses: None,
            l1d_cache_accesses: None,
            dtlb_misses: None,
            l2_cache_misses: None,
        }
    }

    /// Helper to build and enable a hardware counter
    #[allow(dead_code)]
    fn build_counter(kind: Hardware) -> Option<Counter> {
        Builder::new().kind(kind).build().ok().map(|mut counter| {
            if counter.enable().is_err() {
                eprintln!("Warning: Failed to enable counter {:?}", kind);
            }
            counter
        })
    }

    /// Helper to build and enable a cache counter
    #[allow(dead_code)]
    fn build_cache_counter(cache: WhichCache, op: CacheOp, result: CacheResult) -> Option<Counter> {
        use perf_event::events::Cache;
        Builder::new()
            .kind(Cache {
                which: cache,
                operation: op,
                result,
            })
            .build()
            .ok()
            .and_then(|mut counter| {
                counter.enable().ok()?;
                Some(counter)
            })
    }

    /// Disable and read a counter
    fn read_counter(counter: &mut Option<Counter>) -> Option<u64> {
        counter.as_mut().and_then(|counter_handle| {
            counter_handle.disable().ok()?;
            counter_handle.read().ok()
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for PerfSection {
    fn drop(&mut self) {
        // Read all counters
        let cycles = Self::read_counter(&mut self.cycles);
        let instructions = Self::read_counter(&mut self.instructions);
        let stall_frontend = Self::read_counter(&mut self.stall_frontend);
        let stall_backend = Self::read_counter(&mut self.stall_backend);
        let l1d_cache_misses = Self::read_counter(&mut self.l1d_cache_misses);
        let l1d_cache_accesses = Self::read_counter(&mut self.l1d_cache_accesses);
        let dtlb_misses = Self::read_counter(&mut self.dtlb_misses);
        let l2_cache_misses = Self::read_counter(&mut self.l2_cache_misses);

        // Only print if we have any counters
        let has_counters = cycles.is_some()
            || instructions.is_some()
            || stall_frontend.is_some()
            || stall_backend.is_some()
            || l1d_cache_misses.is_some()
            || dtlb_misses.is_some()
            || l2_cache_misses.is_some();

        if !has_counters {
            return;
        }

        eprintln!("[perf] {}", self.name);

        // Core performance
        if let (Some(cycle_count), Some(instruction_count)) = (cycles, instructions) {
            let instructions_per_cycle = instruction_count as f64 / cycle_count as f64;
            eprintln!(
                "  cycles: {}, instructions: {}, IPC: {:.2}",
                format_perf_number(cycle_count),
                format_perf_number(instruction_count),
                instructions_per_cycle
            );
        } else if let Some(cycle_count) = cycles {
            eprintln!("  cycles: {}", format_perf_number(cycle_count));
        } else if let Some(instruction_count) = instructions {
            eprintln!("  instructions: {}", format_perf_number(instruction_count));
        }

        // Stalls
        if let (Some(stall_frontend_count), Some(cycle_count)) = (stall_frontend, cycles) {
            let percent = (stall_frontend_count as f64 / cycle_count as f64) * 100.0;
            eprintln!(
                "  frontend stalls: {} ({:.1}%)",
                format_perf_number(stall_frontend_count),
                percent
            );
        }
        if let (Some(stall_backend_count), Some(cycle_count)) = (stall_backend, cycles) {
            let percent = (stall_backend_count as f64 / cycle_count as f64) * 100.0;
            eprintln!(
                "  backend stalls: {} ({:.1}%)",
                format_perf_number(stall_backend_count),
                percent
            );
        }

        // Memory hierarchy
        if let Some(misses) = l1d_cache_misses {
            if let Some(accesses) = l1d_cache_accesses {
                let rate = (misses as f64 / accesses as f64) * 100.0;
                eprintln!(
                    "  L1D misses: {} ({:.1}%)",
                    format_perf_number(misses),
                    rate
                );
            } else {
                eprintln!("  L1D misses: {}", format_perf_number(misses));
            }
        }
        if let Some(misses) = l2_cache_misses {
            eprintln!("  L2 misses: {}", format_perf_number(misses));
        }
        if let Some(misses) = dtlb_misses {
            eprintln!("  DTLB misses: {}", format_perf_number(misses));
        }
    }
}

/// Format large numbers with thousand separators for readability
#[allow(dead_code)]
#[cfg(target_os = "linux")]
fn format_perf_number(n: u64) -> String {
    let digits = n.to_string();
    let mut result = String::new();
    for (index, digit) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, digit);
    }
    result
}

// No-op implementation for non-Linux platforms
#[allow(dead_code)]
#[cfg(not(target_os = "linux"))]
pub struct PerfSection;

#[cfg(not(target_os = "linux"))]
impl PerfSection {
    #[inline(always)]
    pub fn new(_name: &'static str) -> Self {
        Self
    }

    #[inline(always)]
    pub fn minimal(_name: &'static str) -> Self {
        Self
    }
}
