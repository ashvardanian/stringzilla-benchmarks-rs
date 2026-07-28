#![allow(dead_code)] // Ten benches each use a subset of this harness.
use std::borrow::Cow;
use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fs;
use std::hint::black_box;
use std::io::{Read, Write};
use std::panic;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use stringtape::BytesCowsAuto;

/// Get an optional environment variable, returning None if not set.
pub fn get_env(name: &str) -> Option<String> {
    env::var(name).ok()
}

/// Get an environment variable with a default value.
pub fn get_env_or_default(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

/// Get an environment variable parsed to a type, with a default value.
/// Returns the default if the variable is not set or cannot be parsed.
pub fn get_env_parsed<T: FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Get a boolean environment variable.
/// Accepts "1", "true", or "yes" (case-insensitive) as true values.
/// Returns false if not set or set to any other value.
pub fn get_env_bool(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Installs a custom panic hook that formats errors cleanly for CLI usage.
/// Call this at the start of main() before any potential panics.
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
pub trait ResultExt<T> {
    /// Unwrap the result or panic with the Display-formatted error.
    /// Equivalent to `.unwrap_or_else(|e| panic!("{}", e))` but cleaner.
    fn unwrap_nice(self) -> T;

    /// Unwrap the result or panic with a custom message and the error.
    /// Equivalent to `.unwrap_or_else(|e| panic!("{}: {}", msg, e))`.
    fn expect_or_exit(self, msg: &str) -> T;
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
    fn expect_or_exit(self, msg: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => panic!("{}: {}", msg, error),
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
                       STRINGWARS_DATASET=data.txt STRINGWARS_TOKENS=words cargo bench --features bench_find --bench bench_find"
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
#[cfg(target_os = "linux")]
#[inline]
pub fn reclaim_memory() {
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
#[inline]
pub fn reclaim_memory() {
    // No-op on non-Linux platforms
}

/// The ASCII whitespace set used by `words`, spelled out so it cannot drift from
/// the Python side. Python's bare `str.split()` also splits on Unicode spaces.
/// Width of the left-aligned row-name column, matching `utils.py`.
pub const REPORT_NAME_WIDTH: usize = 42;

const ASCII_WHITESPACE: &[u8] = b" \n\t\r\x0b\x0c";

/// Derived from `ASCII_WHITESPACE` so the normative set stays the single source of truth.
/// `u8::is_ascii_whitespace` is *not* a substitute: it excludes U+000B, which Python's
/// bare `bytes.split()` does split on.
static IS_ASCII_WHITESPACE: [bool; 256] = {
    let mut table = [false; 256];
    let mut index = 0;
    while index < ASCII_WHITESPACE.len() {
        table[ASCII_WHITESPACE[index] as usize] = true;
        index += 1;
    }
    table
};

/// Largest `end <= index` that does not sit inside a UTF-8 continuation byte.
fn char_boundary_floor(bytes: &[u8], mut end: usize) -> usize {
    while end > 0 && end < bytes.len() && (bytes[end] & 0b1100_0000) == 0b1000_0000 {
        end -= 1;
    }
    end
}

/// Global measurement limits and per-suite overrides, read from `stringwars.toml`.
/// Neither harness carries its own defaults, so the two cannot drift.
pub struct Limits {
    pub bytes: u64,
    pub min_sample_ms: f64,
    pub min_samples: u32,
    pub min_seconds: f64,
    pub max_seconds: f64,
    pub target_spread: f64,
    pub warmup_max_seconds: f64,
}

/// Parses a decimal-SI size string (`"256MB"` is 256_000_000), matching how
/// throughput is reported. The old Python side was 1024-based while reporting was
/// 1000-based, so a `128mb` budget read 134,217,728 bytes and printed "134.22 MB".
pub fn parse_size(text: impl AsRef<str>) -> Option<u64> {
    let text = text.as_ref();
    let lowered = text.trim().to_lowercase();
    let digits_end = lowered
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(lowered.len());
    let (number, unit) = lowered.split_at(digits_end);
    let value: f64 = number.trim().parse().ok()?;
    let scale: f64 = match unit.trim() {
        "" | "b" => 1.0,
        "kb" => 1e3,
        "mb" => 1e6,
        "gb" => 1e9,
        _ => return None,
    };
    Some((value * scale) as u64)
}

/// `stringwars.toml` — the single source of defaults shared with `utils.py`.
/// `utils.rs` is included via `#[path]` from each suite directory, so the manifest
/// is resolved relative to this source file rather than the working directory.
/// Parsed once: `limits()` runs per row, and re-reading the file there put an open and a
/// TOML parse between every pair of measurements.
pub fn manifest() -> &'static toml::Value {
    static MANIFEST: OnceLock<toml::Value> = OnceLock::new();
    MANIFEST.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        fs::read_to_string(root.join("stringwars.toml"))
            .or_else(|_| fs::read_to_string("stringwars.toml"))
            .expect_or_exit(
                "stringwars.toml not found; it is the shared manifest for both harnesses",
            )
            .parse::<toml::Value>()
            .expect_or_exit("stringwars.toml is not valid TOML")
    })
}

/// Global measurement limits, with `STRINGWARS_*` overrides applied. Deliberately not
/// cached, so an override set mid-run still applies.
pub fn limits() -> Limits {
    let table = &manifest()["limits"];
    let number = |key: &str| {
        table[key]
            .as_float()
            .unwrap_or_else(|| table[key].as_integer().unwrap_or(0) as f64)
    };
    Limits {
        bytes: get_env("STRINGWARS_BYTES")
            .as_deref()
            .or_else(|| table["bytes"].as_str())
            .and_then(parse_size)
            .expect("limits.bytes is not a valid size; use e.g. 256MB"),
        min_sample_ms: number("min_sample_ms"),
        min_samples: get_env_parsed("STRINGWARS_MIN_SAMPLES", number("min_samples") as u32),
        min_seconds: number("min_seconds"),
        max_seconds: get_env_parsed("STRINGWARS_MAX_SECONDS", number("max_seconds")),
        target_spread: get_env_parsed("STRINGWARS_TARGET_SPREAD", number("target_spread")),
        warmup_max_seconds: get_env_parsed(
            "STRINGWARS_WARMUP_MAX_SECONDS",
            number("warmup_max_seconds"),
        ),
    }
}

/// One suite's dataset, token mode, and working-set budget, resolved from the
/// manifest with `STRINGWARS_*` overrides applied.
pub struct SuiteSettings {
    pub dataset: Option<String>,
    pub tokens: String,
    pub budget_bytes: u64,
}

pub fn suite_settings(suite: &str) -> SuiteSettings {
    let manifest = manifest();
    let entry = manifest.get("suite").and_then(|s| s.get(suite));
    let field = |key: &str| {
        entry
            .and_then(|e| e.get(key))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    };
    let dataset = get_env("STRINGWARS_DATASET").or_else(|| field("dataset"));
    let mode = get_env("STRINGWARS_TOKENS")
        .or_else(|| field("tokens"))
        .unwrap_or_else(|| "lines".to_string());
    let bytes = get_env("STRINGWARS_BYTES")
        .or_else(|| field("bytes"))
        .and_then(parse_size)
        .unwrap_or_else(|| limits().bytes);
    SuiteSettings {
        dataset,
        tokens: mode,
        budget_bytes: bytes,
    }
}

/// Identity of a resolved working set: CRC32 over the tokens joined by a NUL.
/// Digesting the tokens rather than the file means capping the working set does
/// not require re-hashing gigabytes. Mirrored exactly in `utils.py`.
pub fn fingerprint_tokens<'a>(tokens: impl Iterator<Item = &'a [u8]>) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    let mut first = true;
    for token in tokens {
        if !first {
            hasher.update(&[0u8]);
        }
        hasher.update(token);
        first = false;
    }
    hasher.finalize()
}

/// Resolves a suite's working set from the manifest and registers its identity.
pub fn resolve_dataset(suite: &str) -> Result<BytesCowsAuto<'static>, DatasetError> {
    let tape = load_working_set(suite)?;
    let settings = suite_settings(suite);
    let (dataset, mode) = (settings.dataset, settings.tokens);
    let _ = RUN.set(RunIdentity {
        suite: suite.to_string(),
        dataset: dataset.unwrap_or_default(),
        mode,
        tokens: tape.len() as u64,
        bytes: tape.iter().map(|token| token.len() as u64).sum(),
        crc: fingerprint_tokens(tape.iter()),
    });
    Ok(tape)
}

/// Unwraps inside a measured region, so a failing kernel stops the run.
///
/// `let _ = black_box(fallible())` measures the throughput of *returning an error*.
/// `encryption` did exactly that: every OpenSSL row was silently erroring on a null
/// IV and would have published the speed of failing as a cipher benchmark.
#[inline(always)]
pub fn expect_ok<T, E: fmt::Debug>(result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("benchmarked call failed: {error:?}"),
    }
}

/// Every row the run reached, with what became of it.
static ROSTER: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

fn note_row(name: &str, status: &str) {
    if let Ok(mut roster) = ROSTER.lock() {
        roster.push((name.to_string(), status.to_string()));
    }
}

/// Records a contender that could not run at all — a missing optional dependency, a
/// gated backend. Prints a line so the absence is in the output rather than implied
/// by a gap in the table.
pub fn note_unavailable(name: &str, reason: &str) {
    println!(
        "{:<width$} SKIPPED: {reason}",
        name,
        width = REPORT_NAME_WIDTH
    );
    note_row(name, "skipped");
}

/// Prints the roster tally and exits non-zero if any row that was asked to run
/// failed to produce a number. Call at the end of `main`.
pub fn finish() {
    let roster = ROSTER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let count = |want: &str| roster.iter().filter(|(_, status)| status == want).count();
    let (ran, skipped, filtered) = (count("ok"), count("skipped"), count("filtered"));
    let (refused, too_slow) = (count("refused"), count("too_slow"));

    println!(
        "\nRoster: {ran} measured, {refused} refused, {too_slow} too slow, \
         {skipped} skipped, {filtered} filtered"
    );
    if refused > 0 {
        for (name, _) in roster.iter().filter(|(_, status)| status == "refused") {
            println!("  refused: {name}");
        }
        drop(roster);
        std::process::exit(1);
    }
}

/// Registers a working set built by a suite that needs its own mutable tape.
pub fn note_working_set(suite: &str, tokens: &[impl AsRef<[u8]>]) {
    let settings = suite_settings(suite);
    let (dataset, mode) = (settings.dataset, settings.tokens);
    let bytes: u64 = tokens.iter().map(|t| t.as_ref().len() as u64).sum();
    let crc = fingerprint_tokens(tokens.iter().map(|t| t.as_ref()));
    eprintln!(
        "Dataset: {} tokens, {} bytes ({:.2} GB)\n  Identity: mode {} crc 0x{:08x}",
        format_number(tokens.len() as u64),
        format_number(bytes),
        bytes as f64 / 1e9,
        mode,
        crc
    );
    let _ = RUN.set(RunIdentity {
        suite: suite.to_string(),
        dataset: dataset.unwrap_or_default(),
        mode,
        tokens: tokens.len() as u64,
        bytes,
        crc,
    });
}

/// Stamped onto every record; a static because `measure` sees only its own row.
struct RunIdentity {
    suite: String,
    dataset: String,
    mode: String,
    tokens: u64,
    bytes: u64,
    crc: u32,
}

static RUN: OnceLock<RunIdentity> = OnceLock::new();

/// Appends one NDJSON record per row when `STRINGWARS_RESULTS_DIR` is set.
fn record_outcome(outcome: &Outcome, spec: &MeasureSpec, bytes_per_second: f64) {
    let Ok(dir) = env::var("STRINGWARS_RESULTS_DIR") else {
        return;
    };
    let Some(run) = RUN.get() else { return };
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let status = match &outcome.status {
        Status::Converged => "converged".to_string(),
        Status::Unconverged => format!("unconverged:{:.4}", outcome.spread),
        Status::TooFewSamples => format!("too_few_samples:{}", outcome.samples),
        Status::TooSlow { estimated_rate, .. } => format!("too_slow:{estimated_rate:.3e}"),
        Status::NonStationary { mean_median_gap } => format!("non_stationary:{mean_median_gap:.4}"),
        Status::Filtered => return,
    };
    let unit = match spec.unit {
        Unit::Bytes => "bytes/s",
        Unit::Cups => "CUPS",
        Unit::Hashes => "hashes/s",
        Unit::Bits => "bits/s",
        Unit::Comparisons => "cmp/s",
    };
    let line = format!(
        "{{\"lang\":\"rust\",\"suite\":\"{}\",\"dataset\":\"{}\",\"mode\":\"{}\",\
         \"tokens\":{},\"token_bytes\":{},\"crc\":\"0x{:08x}\",\"row\":\"{}\",\
         \"unit\":\"{}\",\"rate\":{:.6e},\"bytes_per_second\":{:.6e},\"spread\":{:.6},\
         \"samples\":{},\"passes_per_sample\":{},\"concurrency\":{},\"status\":\"{}\"}}\n",
        run.suite,
        run.dataset,
        run.mode,
        run.tokens,
        run.bytes,
        run.crc,
        outcome.name.replace('"', "'"),
        unit,
        outcome.median_rate,
        bytes_per_second,
        outcome.spread,
        outcome.samples,
        outcome.passes_per_sample,
        spec.concurrency,
        status,
    );
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(&dir).join(format!("{}.ndjson", run.suite)))
    {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Splits a haystack into token ranges under the normative rule.
///
/// Returned as ranges rather than slices so a caller that needs `&mut` tokens can
/// walk them without the rule being written twice: `memory` used to carry its own
/// copy, and the copy had already drifted — it ignored `STRINGWARS_UNIQUE`, and its
/// `file` arm applied neither the byte budget nor the UTF-8 backoff, so
/// `STRINGWARS_TOKENS=file` there ran over the whole 5 GB corpus.
pub fn token_ranges(haystack: &[u8], mode: &str, budget: u64) -> Vec<std::ops::Range<usize>> {
    if mode == "file" {
        // The one mode where the budget truncates: a single token cannot be dropped
        // without emptying the working set. Back off to a UTF-8 boundary.
        return vec![0..char_boundary_floor(haystack, (budget as usize).min(haystack.len()))];
    }

    let lines = mode == "lines";
    let (mut ranges, mut used, mut offset) = (Vec::new(), 0u64, 0usize);
    for field in haystack.split(|&byte| {
        if lines {
            byte == b'\n'
        } else {
            IS_ASCII_WHITESPACE[byte as usize]
        }
    }) {
        let start = offset;
        offset += field.len() + 1; // `split` consumed exactly one separator
        if field.is_empty() {
            continue;
        }
        if used + field.len() as u64 > budget && !ranges.is_empty() {
            break;
        }
        used += field.len() as u64;
        ranges.push(start..start + field.len());
    }
    ranges
}

/// Reads only what the byte budget can consume, never the whole file.
///
/// The budget counts *token* bytes while a read returns *raw* bytes: `budget` bytes
/// of `xlsum.csv` yields only ~1.80 MB of words against a 2 MB budget, so the read
/// tops up until non-separator content reaches the budget.
pub fn read_within_budget(path: &str, budget: u64, mode: &str) -> std::io::Result<Vec<u8>> {
    let mut file = fs::File::open(path)?;

    // One token: the cap is the budget itself, with the UTF-8 backoff applied later.
    if mode == "file" {
        let mut buffer = Vec::new();
        Read::by_ref(&mut file)
            .take(budget)
            .read_to_end(&mut buffer)?;
        return Ok(buffer);
    }
    // Dedup runs before the cap, so "N unique tokens" genuinely needs the whole
    // corpus; no prefix can answer it. Pay the full read knowingly.
    if get_env_bool("STRINGWARS_UNIQUE") {
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        return Ok(buffer);
    }

    let lines = mode == "lines";
    let is_separator = |byte: u8| {
        if lines {
            byte == b'\n'
        } else {
            IS_ASCII_WHITESPACE[byte as usize]
        }
    };

    let mut buffer = Vec::new();
    let mut want = budget;
    let mut token_bytes: u64 = 0;
    loop {
        let before = buffer.len();
        Read::by_ref(&mut file)
            .take(want)
            .read_to_end(&mut buffer)?;
        let hit_eof = (buffer.len() - before) < want as usize;
        // Count only what was just read. Rescanning the whole buffer per top-up made this
        // quadratic in the budget; `utils.py` already counts incrementally.
        token_bytes += buffer[before..]
            .iter()
            .filter(|&&byte| !is_separator(byte))
            .count() as u64;
        if hit_eof || token_bytes >= budget {
            break;
        }
        want = budget - token_bytes;
    }

    // Complete the straddling token: one bounded block, then cut at the separator.
    let mut tail = Vec::new();
    Read::by_ref(&mut file)
        .take(1 << 20)
        .read_to_end(&mut tail)?;
    let keep = tail
        .iter()
        .position(|&byte| is_separator(byte))
        .unwrap_or(tail.len());
    buffer.extend_from_slice(&tail[..keep]);
    Ok(buffer)
}

/// Resolves the working set for one suite: its dataset, token mode and byte budget
/// come from `stringwars.toml`, each overridable by the matching `STRINGWARS_*` var.
fn load_working_set(suite: &str) -> Result<BytesCowsAuto<'static>, DatasetError> {
    let settings = suite_settings(suite);
    let (manifest_dataset, mode, budget) =
        (settings.dataset, settings.tokens, settings.budget_bytes);
    let dataset_path = manifest_dataset.ok_or(DatasetError::EnvVarNotSet)?;
    let unique = get_env_bool("STRINGWARS_UNIQUE");

    if unique {
        eprintln!("STRINGWARS_UNIQUE: deduplicating tokens");
    }

    // Check if file exists before attempting to read
    if !Path::new(&dataset_path).exists() {
        return Err(DatasetError::FileNotFound { path: dataset_path });
    }

    // Read only what the budget can consume, never the whole file.
    let content = read_within_budget(&dataset_path, budget, &mode).map_err(|error| {
        DatasetError::ReadError {
            path: dataset_path.clone(),
            source: error,
        }
    })?;

    // Check for empty file
    if content.is_empty() {
        return Err(DatasetError::EmptyFile { path: dataset_path });
    }

    // Leak the content to get 'static lifetime
    let content_static: &'static [u8] = Box::leak(content.into_boxed_slice());

    // Normative resolution, mirrored in `utils.py`: tokenize, then accumulate whole
    // tokens and stop *before* the first that would exceed the byte budget. Never a
    // torn token. The budget counts token bytes — the exact denominator of every
    // bytes/s figure — so both languages share a denominator regardless of how much
    // separator overhead a mode discards.
    let take_within_budget = |iter: &mut dyn Iterator<Item = &'static [u8]>| -> Vec<&'static [u8]> {
        let mut kept: Vec<&'static [u8]> = Vec::new();
        let mut used: u64 = 0;
        for token in iter {
            let size = token.len() as u64;
            if used + size > budget && !kept.is_empty() {
                break;
            }
            kept.push(token);
            used += size;
        }
        kept
    };

    let mut seen: HashSet<&'static [u8]> = HashSet::new();

    let tape = match mode.as_str() {
        "lines" => {
            // Deduplicate before capping; the reverse order yields "up to N tokens,
            // then deduplicated", which is fewer than N unique tokens.
            let all = content_static
                .split(|&byte| byte == b'\n')
                .filter(|slice| !slice.is_empty());
            let mut deduped = all.filter(|token| !unique || seen.insert(*token));
            let kept = take_within_budget(&mut deduped);
            BytesCowsAuto::from_iter_and_data(kept, Cow::Borrowed(content_static))
        }
        "words" => {
            let all = content_static
                .split(|&byte| IS_ASCII_WHITESPACE[byte as usize])
                .filter(|slice| !slice.is_empty());
            let mut deduped = all.filter(|token| !unique || seen.insert(*token));
            let kept = take_within_budget(&mut deduped);
            BytesCowsAuto::from_iter_and_data(kept, Cow::Borrowed(content_static))
        }
        "file" => {
            // The one mode where the budget must truncate: a single token cannot be
            // dropped without emptying the working set. Back off to a UTF-8 boundary
            // so the two harnesses cap at the same byte and agree on the fingerprint.
            let end =
                char_boundary_floor(content_static, (budget as usize).min(content_static.len()));
            let capped = &content_static[..end];
            let iter = std::iter::once(capped);
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

    // The identity line is what makes the two harnesses checkable against each
    // other: same mode, count, bytes and CRC means they resolved the same working
    // set. They silently diverged for months (Rust split words on ASCII bytes,
    // Python's `str.split()` also split U+3000 in the CJK corpora), and nothing in
    // the output would have shown it.
    eprintln!(
        "Dataset: {} tokens, {} bytes ({:.2} GB)\n  \
         Identity: mode {} crc 0x{:08x}\n  \
         Length: min {}, max {}, mean {:.1}, std {:.1}",
        format_number(count as u64),
        format_number(total_bytes as u64),
        total_bytes as f64 / 1e9,
        mode,
        fingerprint_tokens(tape.iter()),
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

/// Format large numbers with thousand separators for readability.
fn format_number(n: u64) -> String {
    let digits = n.to_string();
    let head = match digits.len() % 3 {
        0 => 3,
        rest => rest,
    };
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    out.push_str(&digits[..head]);
    for start in (head..digits.len()).step_by(3) {
        out.push(',');
        out.push_str(&digits[start..start + 3]);
    }
    out
}

#[cfg(target_os = "linux")]
use perf_event::{events::Hardware, Builder, Counter};

/// Cycles and instructions across the measured span, when `STRINGWARS_COUNTERS=1`.
///
/// Read twice per row, at the same points as the clock, so the overhead bound is
/// the one already argued for timing. Linux only; elsewhere this is a no-op and the
/// columns simply do not appear.
struct HardwareCounters {
    #[cfg(target_os = "linux")]
    inner: Option<(Option<Counter>, Option<Counter>)>,
}

impl HardwareCounters {
    #[cfg(target_os = "linux")]
    fn start_if_enabled() -> Self {
        if !get_env_bool("STRINGWARS_COUNTERS") {
            return Self { inner: None };
        }
        let build = |kind: Hardware| -> Option<Counter> {
            let mut counter = Builder::new().kind(kind).build().ok()?;
            counter.enable().ok()?;
            Some(counter)
        };
        Self {
            inner: Some((build(Hardware::CPU_CYCLES), build(Hardware::INSTRUCTIONS))),
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn start_if_enabled() -> Self {
        Self {}
    }

    /// Cycles and instructions over the span, or `None` when disabled.
    #[cfg(target_os = "linux")]
    fn stop(self) -> Option<(u64, u64)> {
        let (mut cycles, mut instructions) = self.inner?;
        let read = |counter: &mut Option<Counter>| -> Option<u64> {
            let handle = counter.as_mut()?;
            handle.disable().ok()?;
            handle.read().ok()
        };
        Some((read(&mut cycles)?, read(&mut instructions)?))
    }

    #[cfg(not(target_os = "linux"))]
    fn stop(self) -> Option<(u64, u64)> {
        None
    }
}

/// Whether a row passes `STRINGWARS_FILTER`, recording the skip as it decides.
///
/// The roster note lives here rather than in `measure` because several suites
/// consult the filter themselves and return early; those rows used to vanish from
/// the tally entirely instead of being counted as filtered.
pub fn should_run(name: &str) -> bool {
    static FILTER: OnceLock<Option<(String, Option<regex::Regex>)>> = OnceLock::new();
    let compiled = FILTER.get_or_init(|| {
        get_env("STRINGWARS_FILTER").map(|filter| {
            eprintln!("STRINGWARS_FILTER active: '{}'", filter);
            let regex = regex::Regex::new(&filter).ok();
            if regex.is_none() {
                eprintln!("Warning: invalid regex '{}', matching as substring", filter);
            }
            (filter, regex)
        })
    });

    if let Some((filter, regex)) = compiled {
        if let Some(regex) = regex {
            let matches = regex.is_match(name);
            if !matches {
                eprintln!("  Skipping: {}", name);
                note_row(name, "filtered");
            }
            matches
        } else {
            let matches = name.contains(filter.as_str());
            if !matches {
                note_row(name, "filtered");
            }
            matches
        }
    } else {
        true
    }
}

// Simple SI scaling helper: returns the scaled value and its metric prefix.
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

/// What one routine call accomplished, for dual-metric reporting.
/// `elements` counts pairs / hashes / comparisons / tokens (0 when not applicable);
/// `bytes` counts the bytes touched.
#[derive(Clone, Copy, Default)]
pub struct WorkUnits {
    pub elements: u64,
    pub bytes: u64,
}

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
#[derive(Clone, Copy)]
pub enum Unit {
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

/// Render a bytes-per-second rate as `<value> <prefix>B/s` (decimal SI, 2 decimals).
fn format_byte_rate(bytes_per_second: f64) -> String {
    let (value, prefix) = scale_si(bytes_per_second);
    format!("{:.2} {}B/s", value, prefix)
}

/// Render an SI rate as `<value> <prefix><unit>` (e.g. `1.24 GCUPS`). When `space_before_unit`
/// is set, a space separates the prefix from a word unit (`1.24 G hashes/s`).
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

/// What one pass over the pinned working set costs, declared up front.
///
/// The work is declared rather than accumulated because accumulating it costs two
/// `+=` per call inside the hot loop. At `hash`'s ~15 ns/call that bookkeeping was
/// a measurable fraction of the kernel it was supposed to be measuring.
pub struct MeasureSpec {
    pub unit: Unit,
    /// Work performed by a single pass.
    pub work: WorkUnits,
    /// Threads the variant is expected to use; 1 for single-threaded rows.
    pub concurrency: u32,
}

impl MeasureSpec {
    pub fn new(unit: Unit, work: WorkUnits) -> Self {
        Self {
            unit,
            work,
            concurrency: 1,
        }
    }
    pub fn with_concurrency(mut self, threads: u32) -> Self {
        self.concurrency = threads;
        self
    }
}

/// Why a row printed no number. Reporting something plausible-looking is worse
/// than reporting nothing, so each of these is a hard stop rather than a caveat.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Converged,
    /// Hit `max_seconds` before the spread came inside `target_spread`.
    Unconverged,
    /// Fewer samples than `min_samples` fit inside the cap.
    TooFewSamples,
    /// One pass alone rules out the three samples a dispersion estimate needs, so the
    /// row was never run at full scale. Carries a one-figure ceiling, not a
    /// measurement: `-` in a table, an estimate in the console and the record.
    TooSlow {
        estimated_rate: f64,
        projected_pass_seconds: f64,
    },
    /// Per-sample rates are not stationary — thermal drift, a competing process,
    /// or a leak. A free contamination detector: a stationary distribution has
    /// mean and median within a couple of spreads of each other.
    NonStationary {
        mean_median_gap: f64,
    },
    Filtered,
}

pub struct Outcome {
    pub name: String,
    pub median_rate: f64,
    pub spread: f64,
    pub samples: usize,
    pub passes_per_sample: u64,
    pub status: Status,
}

/// Cost of one `Instant::now()`, measured. The harness reads the clock exactly
/// twice per sample, so this over `min_sample_ms` is the entire timing overhead.
pub fn clock_overhead_nanoseconds() -> f64 {
    let rounds = 10_000u32;
    let start = Instant::now();
    for _ in 0..rounds {
        black_box(Instant::now());
    }
    start.elapsed().as_nanos() as f64 / rounds as f64
}

/// Prints the measured timing overhead. This is the proof obligation behind
/// "the harness does not perturb the measurement" — an assertion otherwise.
pub fn log_timing_overhead() {
    let per_clock = clock_overhead_nanoseconds();
    let floor_nanos = limits().min_sample_ms * 1e6;
    println!(
        "Timing: {:.0} ns/clock, {:.1} ms sample floor -> harness overhead <= {:.4}%",
        per_clock,
        limits().min_sample_ms,
        200.0 * per_clock / floor_nanos
    );
}

/// Selects a sample by rank. Half-up is written out, not delegated: `f64::round`
/// and Python's `round` break ties differently and picked different medians.
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (fraction * (sorted.len() as f64 - 1.0) + 0.5).floor() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Rounds to the number of significant figures the measured dispersion justifies.
/// Printing `239,350 MCUPS` for a number good to +-35% is lying with typography.
fn round_to_earned_digits(value: f64, relative_halfwidth: f64) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    let mut digits = 1;
    for candidate in 1..=4 {
        if 10f64.powi(-(candidate - 1)) >= relative_halfwidth {
            digits = candidate;
        }
    }
    let magnitude = value.abs().log10().floor() as i32;
    let scale = 10f64.powi(digits - 1 - magnitude);
    // Half-up, spelled out, for the same reason as `quantile`: rates are positive,
    // so this is the tie rule both harnesses can state rather than inherit.
    (value * scale + 0.5).floor() / scale
}

/// Times `pass` over the pinned working set and reports one row.
///
/// A **sample** is `passes_per_sample` complete traversals of the working set,
/// bracketed by exactly two clock reads. `passes_per_sample` is chosen during
/// warm-up so a sample lasts at least `min_sample_ms`, which is what bounds the
/// timing overhead — there is no per-call clock and therefore no adaptive stride.
/// The loop between the two clock reads contains no harness instructions at all.
pub fn measure(name: &str, spec: MeasureSpec, mut pass: impl FnMut()) -> Outcome {
    measure_core(
        name,
        spec,
        &mut |passes| {
            let started = Instant::now();
            for _ in 0..passes {
                pass();
            }
            started.elapsed()
        },
        None,
    )
}

/// Times a kernel that consumes its input, rebuilding it outside the clock.
///
/// One pass per sample is forced: a rebuild between passes would land inside the
/// timed span. Sound only when a single pass clears `min_sample_ms`, as a sort does.
pub fn measure_with_setup<State>(
    name: &str,
    spec: MeasureSpec,
    mut setup: impl FnMut() -> State,
    mut body: impl FnMut(&mut State),
) -> Outcome {
    measure_core(
        name,
        spec,
        &mut |_passes| {
            let mut state = setup();
            let started = Instant::now();
            body(&mut state);
            started.elapsed()
        },
        Some(1),
    )
}

/// Shared statistics core behind `measure` and `measure_with_setup`, so convergence,
/// refusal and reporting cannot drift between them.
fn measure_core(
    name: &str,
    spec: MeasureSpec,
    run_sample: &mut dyn FnMut(u64) -> Duration,
    fixed_passes: Option<u64>,
) -> Outcome {
    // A refusal must be as visible as a result. Printing nothing is how a row goes
    // missing for weeks without anyone noticing.
    let refuse = |status: Status, samples: usize| {
        match &status {
            Status::Filtered => {}
            Status::TooFewSamples => println!(
                "{:<width$} REFUSED: {samples} samples in {:.0}s cap (need 3)",
                name,
                limits().max_seconds,
                width = REPORT_NAME_WIDTH
            ),
            Status::TooSlow {
                estimated_rate,
                projected_pass_seconds,
            } => println!(
                "{:<width$} TOO SLOW: ~{} , one pass ~{:.0}s against a {:.0}s cap",
                name,
                format_byte_rate(round_to_earned_digits(*estimated_rate, 1.0)),
                projected_pass_seconds,
                limits().max_seconds,
                width = REPORT_NAME_WIDTH
            ),
            other => println!(
                "{:<width$} REFUSED: {other:?}",
                name,
                width = REPORT_NAME_WIDTH
            ),
        }
        // `should_run` already noted a filtered row. A too-slow row is an expected
        // outcome, not a failure, so it gets its own bucket and does not force a
        // non-zero exit.
        match status {
            Status::Filtered => {}
            Status::TooSlow { .. } => note_row(name, "too_slow"),
            _ => note_row(name, "refused"),
        }
        Outcome {
            name: name.to_string(),
            median_rate: 0.0,
            spread: f64::NAN,
            samples,
            passes_per_sample: 0,
            status,
        }
    };
    if !should_run(name) {
        return refuse(Status::Filtered, 0);
    }

    let lim = limits();
    let sample_floor = Duration::from_secs_f64(lim.min_sample_ms / 1000.0);

    // Choose how many passes make one sample. Scaling by the observed shortfall
    // converges in a couple of steps even when a pass is nanoseconds long. A kernel
    // that consumes its input cannot be repeated inside a sample, so `fixed_passes`
    // pins it at one and skips calibration entirely.
    let mut passes: u64 = fixed_passes.unwrap_or(1);
    let mut pass_cost;
    if fixed_passes.is_none() {
        loop {
            let took = run_sample(passes);
            pass_cost = took / passes.max(1) as u32;
            if took >= sample_floor || passes >= 1 << 40 {
                break;
            }
            let shortfall = sample_floor.as_secs_f64() / took.as_secs_f64().max(1e-9);
            passes = passes
                .saturating_mul((shortfall.ceil() as u64).max(2))
                .min(1 << 40);
        }
    } else {
        // A consuming kernel cannot batch passes, but one sample still prices one. Leaving
        // `pass_cost` at zero made the refusal below test `0.0 * 3.0 > max_seconds`, so
        // `sequence` - the one suite whose work is superlinear - could never be refused.
        pass_cost = run_sample(passes) / passes.max(1) as u32;
    }

    if pass_cost.as_secs_f64() * 3.0 > lim.max_seconds {
        let seconds = pass_cost.as_secs_f64();
        return refuse(
            Status::TooSlow {
                estimated_rate: spec.work.bytes as f64 / seconds.max(1e-9),
                projected_pass_seconds: seconds,
            },
            1,
        );
    }

    // Warm-up: discard samples until three consecutive ones agree. Two cannot
    // distinguish agreement from coincidence. This is also what consumes the
    // cold-traversal climb that made `hash` look like it never converged.
    let warmup_deadline = Instant::now() + Duration::from_secs_f64(lim.warmup_max_seconds);
    let mut recent: Vec<f64> = Vec::new();
    while recent.len() < 3 || Instant::now() < warmup_deadline {
        let seconds = run_sample(passes).as_secs_f64();
        if recent.len() == 3 {
            recent.rotate_left(1);
            recent[2] = seconds;
        } else {
            recent.push(seconds);
        }
        if recent.len() == 3 {
            let (low, high) = recent
                .iter()
                .fold((f64::MAX, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
            if (high - low) / high.max(1e-12) <= lim.target_spread {
                break;
            }
        }
    }

    // Measure. Counters bracket the whole measured span rather than each sample, so
    // they cost two reads per row and cannot perturb the per-sample timing.
    let counters = HardwareCounters::start_if_enabled();
    let measure_start = Instant::now();
    let cap = Duration::from_secs_f64(lim.max_seconds);
    let mut seconds_per_sample: Vec<f64> = Vec::new();
    let mut scratch: Vec<f64> = Vec::new();
    loop {
        seconds_per_sample.push(run_sample(passes).as_secs_f64());

        let elapsed = measure_start.elapsed();
        let enough = seconds_per_sample.len() >= lim.min_samples as usize
            && elapsed.as_secs_f64() >= lim.min_seconds;
        if enough {
            scratch.clear();
            scratch.extend_from_slice(&seconds_per_sample);
            let sorted = &mut scratch;
            sorted.sort_unstable_by(f64::total_cmp);
            let median = quantile(&sorted, 0.5);
            let halfwidth =
                (quantile(&sorted, 0.9) - quantile(&sorted, 0.1)) / (2.0 * median.max(1e-12));
            if halfwidth <= lim.target_spread {
                break;
            }
        }
        if elapsed >= cap {
            break;
        }
    }

    if seconds_per_sample.len() < 3 {
        let collected = seconds_per_sample.len();
        return refuse(Status::TooFewSamples, collected);
    }

    scratch.clear();
    scratch.extend_from_slice(&seconds_per_sample);
    let sorted = &mut scratch;
    sorted.sort_unstable_by(f64::total_cmp);
    let median_seconds = quantile(&sorted, 0.5);
    let spread = (quantile(&sorted, 0.9) - quantile(&sorted, 0.1)) / median_seconds.max(1e-12);
    let mean_seconds = seconds_per_sample.iter().sum::<f64>() / seconds_per_sample.len() as f64;
    let gap = (mean_seconds - median_seconds).abs() / median_seconds.max(1e-12);

    let status = if gap > 2.0 * lim.target_spread {
        Status::NonStationary {
            mean_median_gap: gap,
        }
    } else if spread / 2.0 > lim.target_spread {
        Status::Unconverged
    } else {
        Status::Converged
    };

    let elements = spec.work.elements.saturating_mul(passes) as f64;
    let bytes = spec.work.bytes.saturating_mul(passes) as f64;
    let primary = match spec.unit {
        Unit::Bytes => bytes,
        _ => elements,
    } / median_seconds;

    let outcome = Outcome {
        name: name.to_string(),
        median_rate: primary,
        spread,
        samples: seconds_per_sample.len(),
        passes_per_sample: passes,
        status,
    };
    report_outcome(&outcome, &spec, bytes / median_seconds, counters.stop());
    record_outcome(&outcome, &spec, bytes / median_seconds);
    note_row(name, "ok");
    outcome
}

fn report_outcome(
    outcome: &Outcome,
    spec: &MeasureSpec,
    bytes_per_second: f64,
    counters: Option<(u64, u64)>,
) {
    let mut columns: Vec<String> = Vec::new();
    let shown = round_to_earned_digits(outcome.median_rate, outcome.spread / 2.0);
    columns.push(match spec.unit {
        Unit::Bytes => format_byte_rate(shown),
        Unit::Cups => format_si_rate(shown, "CUPS", false),
        Unit::Hashes => format_si_rate(shown, "hashes/s", true),
        Unit::Bits => format_si_rate(shown, "bits/s", true),
        Unit::Comparisons => format_si_rate(shown, "cmp/s", true),
    });
    if !matches!(spec.unit, Unit::Bytes) && spec.work.bytes > 0 {
        columns.push(format_byte_rate(round_to_earned_digits(
            bytes_per_second,
            outcome.spread / 2.0,
        )));
    }
    columns.push(format!(
        "+-{:.1}% n={}",
        100.0 * outcome.spread / 2.0,
        outcome.samples
    ));
    if let Some((cycles, instructions)) = counters {
        let bytes = (spec.work.bytes as f64)
            * (outcome.passes_per_sample as f64)
            * (outcome.samples as f64);
        if bytes > 0.0 {
            columns.push(format!("{:.2} cyc/B", cycles as f64 / bytes));
        }
        if cycles > 0 {
            columns.push(format!("{:.2} IPC", instructions as f64 / cycles as f64));
        }
    }
    match &outcome.status {
        Status::Converged => {}
        Status::Unconverged => columns.push(format!("UNCONVERGED {:.0}%", 100.0 * outcome.spread)),
        Status::NonStationary { mean_median_gap } => {
            columns.push(format!("NON-STATIONARY {:.0}%", 100.0 * mean_median_gap))
        }
        Status::TooFewSamples => columns.push(format!("REFUSED {} samples", outcome.samples)),
        // Reported by `refuse`, which never reaches this row-printing path.
        Status::TooSlow { .. } | Status::Filtered => return,
    }
    println!(
        "{:<width$} {}",
        outcome.name,
        columns.join(" | "),
        width = REPORT_NAME_WIDTH
    );
}

/// Logical cores for a multi-core scope, overridable with `STRINGWARS_CPU_CORES`.
pub fn resolve_core_count(available: usize) -> usize {
    get_env("STRINGWARS_CPU_CORES")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&cores| cores > 0)
        .unwrap_or(available)
        .max(1)
}

/// Items processed per core — one CPU core, or on the GPU one streaming multiprocessor (SM).
/// "Core" here means an SM, not an individual warp or CUDA core. `default_base` is the bench's own
/// per-core default (the right value differs by kernel — short-string similarity saturates the GPU
/// at a different batch than document fingerprinting); `STRINGWARS_BATCH_PER_CORE` overrides it.
fn items_per_core(default_base: usize) -> usize {
    get_env_parsed("STRINGWARS_BATCH_PER_CORE", default_base).max(1)
}

/// Batch size for a backend with `cores` parallel cores, scaling the bench's `default_base` by the
/// hardware. A CPU core counts as one core and a GPU streaming multiprocessor counts as one core,
/// so the batch scales automatically instead of a fixed CPU/GPU multiplier: a 1-core scope gets
/// `items_per_core`, an N-core scope `N * items_per_core`, and a GPU `SMs * items_per_core`.
pub fn auto_batch_size(cores: usize, default_base: usize) -> usize {
    items_per_core(default_base)
        .saturating_mul(cores.max(1))
        .max(1)
}

/// Number of streaming multiprocessors on the given CUDA device, queried from the CUDA runtime.
/// Each SM is counted as one core for batch sizing (an SM, not a warp or an individual CUDA core).
/// Returns None when CUDA is unavailable or the query fails, so callers fall back to a default
/// core count. The attribute id 16 is `cudaDevAttrMultiProcessorCount`.
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
#[cfg(not(feature = "cuda"))]
pub fn gpu_multiprocessor_count(_device_index: i32) -> Option<usize> {
    None
}

/// RAII guard that profiles a code section using perf events on Linux.
/// On non-Linux platforms, this is a zero-cost abstraction.
#[cfg(test)]
mod tests {
    use super::*;

    /// The rank rule is written out rather than delegated to `f64::round`, which
    /// breaks ties away from zero while Python's `round` breaks them to even. At
    /// n = 10 — the modal terminating count — that put the two harnesses on
    /// different samples, and every median and spread is derived from this.
    #[test]
    fn quantile_rank_is_half_up() {
        let expected = |n: usize, fraction: f64| -> usize {
            ((fraction * (n as f64 - 1.0) + 0.5).floor() as usize).min(n - 1)
        };
        for n in [3usize, 6, 10, 12, 14, 18, 25, 100] {
            let samples: Vec<f64> = (0..n).map(|index| index as f64).collect();
            for fraction in [0.1, 0.5, 0.9] {
                assert_eq!(
                    quantile(&samples, fraction),
                    expected(n, fraction) as f64,
                    "n={n} fraction={fraction}"
                );
            }
        }
        // The case that actually diverged: 10 samples, median.
        let ten: Vec<f64> = (0..10).map(|index| index as f64).collect();
        assert_eq!(quantile(&ten, 0.5), 5.0);
    }

    #[test]
    fn quantile_handles_degenerate_input() {
        assert_eq!(quantile(&[], 0.5), 0.0);
        assert_eq!(quantile(&[7.0], 0.9), 7.0);
    }

    /// Digits are earned from dispersion: `239,350 MCUPS` for a number good to
    /// +-35% is lying with typography.
    #[test]
    fn significant_figures_track_dispersion() {
        // digits = the largest d with 10^-(d-1) >= halfwidth, so +-35% earns one
        // figure and +-2% earns two.
        assert_eq!(round_to_earned_digits(1234.0, 0.35), 1000.0);
        assert_eq!(round_to_earned_digits(1234.0, 0.02), 1200.0);
        assert_eq!(round_to_earned_digits(1234.0, 0.002), 1230.0);
        assert_eq!(round_to_earned_digits(0.0, 0.02), 0.0);
        assert!(round_to_earned_digits(f64::NAN, 0.02).is_nan());
    }

    #[test]
    fn sizes_parse_as_decimal_si() {
        // Decimal, matching how throughput is reported: the old 1024-based parse
        // read `128mb` as 134,217,728 and then printed "134.22 MB".
        assert_eq!(parse_size("256MB"), Some(256_000_000));
        assert_eq!(parse_size("1GB"), Some(1_000_000_000));
        assert_eq!(parse_size(" 16mb "), Some(16_000_000));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size("16MBB"), None);
    }

    /// Both harnesses must resolve byte-identical working sets, so the tokenizer
    /// is the one rule that may never drift.
    #[test]
    fn fingerprint_distinguishes_token_boundaries() {
        let joined = fingerprint_tokens([&b"ab"[..], &b"c"[..]].into_iter());
        let single = fingerprint_tokens([&b"abc"[..]].into_iter());
        assert_ne!(joined, single, "NUL separator must survive the digest");
        assert_eq!(fingerprint_tokens([].into_iter()), 0);
    }

    /// The rule `memory` used to fork. Its copy ignored the budget in `file` mode
    /// and split `words` on a narrower set, so the two disagreed silently.
    #[test]
    fn token_ranges_match_the_reference_state_machine() {
        // The pre-`split` implementation, kept only as a fuzzing oracle: this is the
        // normative tokenizer, so a silent divergence would corrupt every working set.
        fn reference(haystack: &[u8], mode: &str, budget: u64) -> Vec<std::ops::Range<usize>> {
            let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
            let mut used: u64 = 0;
            if mode == "file" {
                let mut end = (budget as usize).min(haystack.len());
                while end > 0
                    && end < haystack.len()
                    && (haystack[end] & 0b1100_0000) == 0b1000_0000
                {
                    end -= 1;
                }
                return vec![0..end];
            }
            let is_separator = |byte: u8| {
                if mode == "lines" {
                    byte == b'\n'
                } else {
                    ASCII_WHITESPACE.contains(&byte)
                }
            };
            let mut start = None;
            for (index, &byte) in haystack.iter().enumerate() {
                if is_separator(byte) {
                    if let Some(begin) = start.take() {
                        let size = (index - begin) as u64;
                        if used + size > budget && !ranges.is_empty() {
                            return ranges;
                        }
                        ranges.push(begin..index);
                        used += size;
                    }
                } else if start.is_none() {
                    start = Some(index);
                }
            }
            if let Some(begin) = start {
                let size = (haystack.len() - begin) as u64;
                if used + size <= budget || ranges.is_empty() {
                    ranges.push(begin..haystack.len());
                }
            }
            ranges
        }

        // Deterministic xorshift, so a failure reproduces exactly.
        let alphabet = [
            b"a".as_slice(),
            b"bb",
            b" ",
            b"\n",
            b"\t",
            b"\r",
            b"\x0b",
            b"\x0c",
            "é".as_bytes(),
            "\u{4e2d}".as_bytes(),
        ];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let mut haystack = Vec::new();
            for _ in 0..(next() % 24) {
                haystack.extend_from_slice(alphabet[(next() % alphabet.len() as u64) as usize]);
            }
            let budget = next() % 70;
            for mode in ["words", "lines", "file"] {
                assert_eq!(
                    token_ranges(&haystack, mode, budget),
                    reference(&haystack, mode, budget),
                    "mode={mode} budget={budget} haystack={haystack:?}",
                );
            }
        }
    }

    #[test]
    fn token_ranges_follow_the_normative_rule() {
        let text = b"alpha beta\n\ngamma";
        let words = token_ranges(text, "words", 1 << 20);
        assert_eq!(words.len(), 3);
        assert_eq!(&text[words[0].clone()], b"alpha");
        assert_eq!(&text[words[2].clone()], b"gamma");

        // Empty tokens are dropped, so the blank line yields nothing.
        let lines = token_ranges(text, "lines", 1 << 20);
        assert_eq!(lines.len(), 2);

        // The budget stops *before* the first token that would exceed it.
        let capped = token_ranges(text, "words", 5);
        assert_eq!(capped.len(), 1);

        // A budget smaller than the first token still yields that token, or the
        // working set would be empty.
        assert_eq!(token_ranges(text, "words", 1).len(), 1);

        // `file` truncates and backs off to a UTF-8 boundary.
        let utf8 = "aé".as_bytes();
        assert_eq!(token_ranges(utf8, "file", 2), vec![0..1]);
    }

    #[test]
    fn ascii_whitespace_matches_the_documented_set() {
        assert_eq!(ASCII_WHITESPACE, b" \n\t\r\x0b\x0c");
        // Unicode spaces are deliberately excluded: Python's bare `str.split()`
        // also splits U+3000, which Rust never did, and the CJK corpora contain it.
        assert!(!ASCII_WHITESPACE.contains(&0xA0));
        // The lookup table the tokenizer actually consults must agree with the slice
        // it is derived from, for every byte.
        for byte in 0..=u8::MAX {
            assert_eq!(
                IS_ASCII_WHITESPACE[byte as usize],
                ASCII_WHITESPACE.contains(&byte),
                "byte {byte:#04x}",
            );
        }
    }
}
