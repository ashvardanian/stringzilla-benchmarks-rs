//! Repeated bounded Levenshtein search over an immutable dictionary.
//!
//! The dictionary is built once and reused for every query. Use `STRINGWARS_FILTER` to select
//! `levenshtein/fst` or `levenshtein/symspell`.

#[allow(dead_code)]
#[path = "../utils.rs"]
mod utils;

use fst::automaton::Levenshtein;
use fst::{IntoStreamer, Map, Streamer};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;
use symspell_rs::{SymSpell, Verbosity};
use utils::should_run;

type AnyError = Box<dyn std::error::Error>;

#[derive(Clone, Copy)]
struct Match {
    id: u32,
    distance: u8,
}

struct Settings {
    repeats: usize,
    min_distance: usize,
    max_distance: usize,
    dump_prefix: Option<String>,
    cache_evict_bytes: usize,
}

fn env_usize(name: &str, default: usize) -> Result<usize, AnyError> {
    env::var(name).map_or(Ok(default), |value| Ok(value.parse()?))
}

fn load_lines(path: &str, limit: usize) -> Result<Vec<String>, AnyError> {
    let contents = fs::read_to_string(path)?;
    Ok(contents
        .lines()
        .take(if limit == 0 { usize::MAX } else { limit })
        .map(|line| line.trim_end_matches('\r').to_owned())
        .collect())
}

fn checksum_id(checksum: u64, id: u64) -> u64 {
    checksum
        .wrapping_mul(0x9E37_79B1_85EB_CA87)
        .wrapping_add(id)
}

fn evict_cache(buffer: &mut [u8], checksum: &mut u64) {
    for value in buffer.iter_mut().step_by(64) {
        *value = value.wrapping_add(1);
        *checksum = checksum.wrapping_add(*value as u64);
    }
    std::hint::black_box(*checksum);
}

fn levenshtein_within(first: &str, second: &str, bound: usize) -> Option<usize> {
    let first: Vec<char> = first.chars().collect();
    let second: Vec<char> = second.chars().collect();
    if first.len().abs_diff(second.len()) > bound {
        return None;
    }

    let mut previous: Vec<usize> = (0..=first.len()).collect();
    let mut current = vec![0; first.len() + 1];
    for (row, &second_char) in second.iter().enumerate() {
        current[0] = row + 1;
        let mut row_min = current[0];
        for (column, &first_char) in first.iter().enumerate() {
            current[column + 1] = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(previous[column] + usize::from(first_char != second_char));
            row_min = row_min.min(current[column + 1]);
        }
        if row_min > bound {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[first.len()] <= bound).then_some(previous[first.len()])
}

fn symspell_search(
    symspell: &SymSpell,
    ids: &HashMap<String, u32>,
    query: &str,
    bound: usize,
) -> Vec<Match> {
    symspell
        .lookup(query, Verbosity::All, bound, &None, None, false)
        .into_iter()
        .filter_map(|suggestion| {
            levenshtein_within(query, &suggestion.term, bound).map(|distance| Match {
                id: ids[&suggestion.term],
                distance: distance as u8,
            })
        })
        .collect()
}

fn dump_symspell_matches(
    symspell: &SymSpell,
    ids: &HashMap<String, u32>,
    dictionary_size: usize,
    queries: &[String],
    bound: usize,
    path: &str,
) -> Result<(), AnyError> {
    let mut output = BufWriter::new(File::create(path)?);
    output.write_all(b"SZLEV001")?;
    output.write_all(&(dictionary_size as u64).to_ne_bytes())?;
    output.write_all(&(queries.len() as u64).to_ne_bytes())?;
    output.write_all(&[bound as u8])?;
    for query in queries {
        let mut matches = symspell_search(symspell, ids, query, bound);
        matches.sort_unstable_by_key(|found| (found.id, found.distance));
        output.write_all(&(matches.len() as u64).to_ne_bytes())?;
        for found in matches {
            output.write_all(&found.id.to_ne_bytes())?;
            output.write_all(&[found.distance])?;
        }
    }
    Ok(())
}

fn run_symspell(
    dictionary: &[String],
    queries: &[String],
    settings: &Settings,
) -> Result<(), AnyError> {
    if dictionary
        .iter()
        .chain(queries)
        .any(|text| text.to_lowercase() != *text)
    {
        return Err("SymSpell requires lowercase input for case-sensitive comparison".into());
    }

    let mut ids = HashMap::with_capacity(dictionary.len());
    for (id, word) in dictionary.iter().enumerate() {
        if ids.insert(word.clone(), id as u32).is_some() {
            return Err("SymSpell cannot preserve duplicate dictionary entries".into());
        }
    }

    println!("# levenshtein/symspell");
    println!(
        "dictionary={} queries={} semantics=utf8-codepoints+exact-levenshtein-filter output=id+distance",
        dictionary.len(),
        queries.len()
    );
    let mut cache = vec![0u8; settings.cache_evict_bytes];
    let mut cache_checksum = 0u64;

    for bound in settings.min_distance..=settings.max_distance {
        let build_start = Instant::now();
        let mut symspell = SymSpell::new(bound, None, 7, 1);
        for word in dictionary {
            symspell.create_dictionary_entry(word, 1);
        }
        println!(
            "k={bound} build={:.6}s indexed_words={}",
            build_start.elapsed().as_secs_f64(),
            symspell.get_dictionary_size()
        );

        for repeat in 0..settings.repeats {
            evict_cache(&mut cache, &mut cache_checksum);
            let start = Instant::now();
            let mut matches_count = 0usize;
            let mut checksum = 0u64;
            for query in queries {
                for found in symspell_search(&symspell, &ids, query, bound) {
                    matches_count += 1;
                    checksum = checksum.wrapping_add(
                        ((found.id as u64) << 8 | found.distance as u64)
                            .wrapping_mul(0x9E37_79B1_85EB_CA87),
                    );
                }
            }
            println!(
                "k={bound} repeat={repeat} query={:.6}s matches={matches_count} checksum={checksum:016x} cache_evict_bytes={}",
                start.elapsed().as_secs_f64(),
                cache.len()
            );
        }

        if let Some(prefix) = &settings.dump_prefix {
            dump_symspell_matches(
                &symspell,
                &ids,
                dictionary.len(),
                queries,
                bound,
                &format!("{prefix}.symspell.k{bound}.bin"),
            )?;
        }
    }
    Ok(())
}

fn verify_fst_unicode_contract() -> Result<(), AnyError> {
    let mut keys = vec!["é", "Ѐ", "А"];
    keys.sort_unstable();
    let map = Map::from_iter(keys.iter().enumerate().map(|(id, key)| (*key, id as u64)))?;
    let mut matches = 0usize;
    for query in &keys {
        let automaton = Levenshtein::new(query, 1)?;
        let mut stream = map.search(&automaton).into_stream();
        while stream.next().is_some() {
            matches += 1;
        }
    }
    if matches != 9 {
        return Err(format!(
            "fst Unicode smoke test failed: expected 9 one-character matches, observed {matches}"
        )
        .into());
    }
    Ok(())
}

fn run_fst(dictionary: &[String], queries: &[String], settings: &Settings) -> Result<(), AnyError> {
    let allow_unicode = env::var_os("STRINGWARS_ALLOW_FST_UNICODE").is_some();
    if allow_unicode {
        verify_fst_unicode_contract()?;
    } else if dictionary
        .iter()
        .chain(queries)
        .any(|text| !text.is_ascii())
    {
        return Err("fst requires ASCII input for byte-for-byte semantic parity".into());
    }

    let mut keyed_words: Vec<(&str, u64)> = dictionary
        .iter()
        .enumerate()
        .map(|(id, word)| (word.as_str(), id as u64))
        .collect();
    keyed_words.sort_unstable_by_key(|&(word, _)| word);
    if keyed_words.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err("fst cannot preserve duplicate dictionary entries".into());
    }

    let build_start = Instant::now();
    let map = Map::from_iter(keyed_words)?;
    println!("# levenshtein/fst");
    println!(
        "dictionary={} queries={} semantics={} build={:.6}s fst_bytes={} output=id-only",
        dictionary.len(),
        queries.len(),
        if allow_unicode {
            "unicode-codepoints"
        } else {
            "ascii-byte-parity"
        },
        build_start.elapsed().as_secs_f64(),
        map.as_fst().as_bytes().len()
    );

    let state_limit = env::var("STRINGWARS_FST_STATE_LIMIT")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let mut cache = vec![0u8; settings.cache_evict_bytes];
    let mut cache_checksum = 0u64;
    for bound in settings.min_distance..=settings.max_distance {
        for repeat in 0..settings.repeats {
            evict_cache(&mut cache, &mut cache_checksum);
            let start = Instant::now();
            let mut matches_count = 0usize;
            let mut checksum = 0u64;
            let mut failed = None;
            for query in queries {
                let automaton = match state_limit {
                    Some(limit) => Levenshtein::new_with_limit(query, bound as u32, limit),
                    None => Levenshtein::new(query, bound as u32),
                };
                let automaton = match automaton {
                    Ok(automaton) => automaton,
                    Err(error) => {
                        failed = Some(error);
                        break;
                    }
                };
                let mut stream = map.search(&automaton).into_stream();
                while let Some((_, id)) = stream.next() {
                    matches_count += 1;
                    checksum = checksum_id(checksum, id);
                }
            }
            if let Some(error) = failed {
                println!("k={bound} skipped={error}");
                break;
            }
            println!(
                "k={bound} repeat={repeat} query={:.6}s matches={matches_count} checksum={checksum:016x} cache_evict_bytes={}",
                start.elapsed().as_secs_f64(),
                cache.len()
            );
        }
    }
    Ok(())
}

fn main() -> Result<(), AnyError> {
    let args: Vec<String> = env::args()
        .filter(|argument| argument != "--bench")
        .collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: bench_levenshtein DICTIONARY QUERIES [QUERY_LIMIT]");
        std::process::exit(2);
    }

    let query_limit = args.get(3).map_or(Ok(0), |value| value.parse::<usize>())?;
    let dictionary = load_lines(&args[1], 0)?;
    let queries = load_lines(&args[2], query_limit)?;
    let cache_evict_mb = env_usize("STRINGWARS_CACHE_EVICT_MB", 0)?;
    let settings = Settings {
        repeats: env_usize("STRINGWARS_REPEATS", 3)?,
        min_distance: env_usize("STRINGWARS_MIN_DISTANCE", 1)?,
        max_distance: env_usize("STRINGWARS_MAX_DISTANCE", 2)?,
        dump_prefix: env::var("STRINGWARS_DUMP_PREFIX").ok(),
        cache_evict_bytes: cache_evict_mb
            .checked_mul(1024 * 1024)
            .ok_or("STRINGWARS_CACHE_EVICT_MB is too large")?,
    };
    if settings.repeats == 0
        || settings.min_distance == 0
        || settings.min_distance > settings.max_distance
        || settings.max_distance > 4
    {
        return Err(
            "expected positive repeats and 1 <= minimum distance <= maximum distance <= 4".into(),
        );
    }

    if should_run("levenshtein/fst") {
        run_fst(&dictionary, &queries, &settings)?;
    }
    if should_run("levenshtein/symspell") {
        run_symspell(&dictionary, &queries, &settings)?;
    }
    Ok(())
}
