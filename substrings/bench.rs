#![doc = r#"# StringWars: Multi-Pattern Search Benchmarks

Matches a whole dictionary of needles against a whole collection of haystacks in one pass - the workhorse of
log scanning, protocol dispatch, content filtering, and signature matching. Every engine here compiles its
needle set once and reuses it, so the dictionary is paid for outside the measured loop.

It evaluates:
- StringZilla `szs::Substrings` on one core, every core, and one GPU
- `aho-corasick`, the reference Rust implementation, as a dense DFA and as a contiguous NFA
- `daachorse`, a double-array Aho-Corasick, the direct analog of our cold tier
- `regex` over an alternation of literals, the baseline most people actually write
- `bm25`, an index-then-query scorer, for the scoring group only

Two framing rules, or the table lies. First, __the head-to-head row is `<1cpu>`__: every rival is
single-threaded and CPU-only, so `<Ncpu>` and `<1gpu>` are what the engine adds rather than what it beats
them by. Second, __scoring is scan versus index__: the `bm25` crate tokenizes a corpus and builds an
inverted index once, then answers many queries, while `szs::Substrings` scans raw bytes against a fixed
dictionary and builds nothing per query. Build and query are reported separately.

Uncased matching is not benchmarked here. `aho-corasick`'s `ascii_case_insensitive` folds ASCII only, while
`szs::Substrings` folds the whole Unicode table, so the two solve different problems.

The dictionary comes from the corpus itself, deterministically: whitespace-cut words of 3 to 32 bytes, with
every term occurring once and the most frequent one percent dropped. Each sweep cell draws one slice from
the frequency-ordered remainder, matching `bench/substrings.cpp` in the StringZilla tree so the numbers sit
beside `include/stringzillas/substrings/README.md`.

## Usage Examples

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_DATASET_LIMIT`: Read at most this many bytes; the default is a compute-bound slice.
- `STRINGWARS_TOKENS`: Tokenization model shaping the haystacks; `lines` is the default here.
- `STRINGWARS_CPU_CORES`: Cores for the multi-core scope, else the logical core count.
- `STRINGWARS_FILTER`: Regular expression over benchmark names.

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=data/xlsum/xlsum.csv \
    STRINGWARS_DATASET_LIMIT=64mb \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_substrings --bench bench_substrings
```

On a GPU-capable machine, enable the CUDA feature; the `<1gpu>` variants stay silent without one:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=data/xlsum/xlsum.csv \
    cargo bench --features "cuda bench_substrings" --bench bench_substrings
```
"#]

use std::collections::HashMap;
use std::hint::black_box;

use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use daachorse::{
    DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind as DoubleArrayMatchKind,
};
use forkunion as fu;
use regex::bytes::RegexBuilder;
use stringtape::BytesTape;
use stringzilla::szs::{
    AnyBytesTape, Bm25Params, CaseSensitivity, OverlapPolicy, Substrings, SubstringsMatch,
    UnifiedAlloc,
};

#[path = "../utils.rs"]
mod utils;
use utils::{
    for_each_device_pass, install_panic_hook, load_dataset, log_stringzilla_metadata,
    measure_throughput, report_skipped, BenchBudget, DeviceChoice, ReportAs, ResultExt, WorkUnits,
    COMPUTE_BOUND_SLICE,
};

// region: Vocabulary

/// Shortest word admitted into the dictionary: anything under three bytes matches at nearly every
/// position, so the benchmark would measure match materialization rather than the automaton walk.
const VOCABULARY_MIN_WORD_BYTES: usize = 3;

/// Longest word admitted. Longer whitespace-cut tokens are unsegmented CJK runs and URLs rather than
/// words, and the longest needle sets the warm-up every GPU chunk re-walks.
const VOCABULARY_MAX_WORD_BYTES: usize = 32;

/// Which end of the frequency-ordered vocabulary a dictionary is drawn from, and how much of it.
#[derive(Clone, Copy)]
enum VocabularySlice {
    MostFrequentOnePercent,
    MostFrequentTenPercent,
    LeastFrequentOnePercent,
    LeastFrequentTenPercent,
    Entire,
}

impl VocabularySlice {
    fn name(self) -> &'static str {
        match self {
            Self::MostFrequentOnePercent => "most_frequent_1%",
            Self::MostFrequentTenPercent => "most_frequent_10%",
            Self::LeastFrequentOnePercent => "least_frequent_1%",
            Self::LeastFrequentTenPercent => "least_frequent_10%",
            Self::Entire => "entire",
        }
    }
}

const VOCABULARY_SLICES: [VocabularySlice; 5] = [
    VocabularySlice::MostFrequentOnePercent,
    VocabularySlice::MostFrequentTenPercent,
    VocabularySlice::LeastFrequentOnePercent,
    VocabularySlice::LeastFrequentTenPercent,
    VocabularySlice::Entire,
];

/// Counts word frequencies over the whole corpus and drops both noisy ends, returning the survivors
/// ordered by descending frequency with ties broken by content so the ranking is reproducible.
///
/// One hashed counting pass, then a sort over only the distinct survivors: a corpus holds hundreds of
/// millions of words but only a few million distinct ones.
fn build_vocabulary(corpus: &'static [u8]) -> Vec<&'static [u8]> {
    let mut frequencies: HashMap<&'static [u8], usize> = HashMap::new();
    for word in corpus.split(|byte| byte.is_ascii_whitespace()) {
        if word.len() < VOCABULARY_MIN_WORD_BYTES || word.len() > VOCABULARY_MAX_WORD_BYTES {
            continue;
        }
        *frequencies.entry(word).or_insert(0) += 1;
    }

    // The hapax filter runs before the sort, so the ranking pass touches only the few percent that
    // survive. The top cutoff keeps its base: a fraction of ALL distinct terms, hapax included.
    let distinct_count = frequencies.len();
    let mut ranked: Vec<(&'static [u8], usize)> = frequencies
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .collect();
    ranked.sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));

    let dropped_from_top = (distinct_count / 100).min(ranked.len());
    ranked.drain(..dropped_from_top);
    ranked.into_iter().map(|(term, _)| term).collect()
}

/// One contiguous slice of the frequency-ordered vocabulary.
fn needle_slice_of(vocabulary: &[&'static [u8]], slice: VocabularySlice) -> Vec<&'static [u8]> {
    let available = vocabulary.len();
    let (first, wanted) = match slice {
        VocabularySlice::MostFrequentOnePercent => (0, available / 100),
        VocabularySlice::MostFrequentTenPercent => (0, available / 10),
        VocabularySlice::LeastFrequentOnePercent => (available - available / 100, available / 100),
        VocabularySlice::LeastFrequentTenPercent => (available - available / 10, available / 10),
        VocabularySlice::Entire => (0, available),
    };
    vocabulary[first..first + wanted.max(1).min(available)].to_vec()
}

// endregion: Vocabulary

// region: Harness

/// Logical core count for the multi-core scope, probed once from a caller-owned topology.
fn resolve_core_count(topology: &fu::Topology) -> usize {
    std::env::var("STRINGWARS_CPU_CORES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&cores| cores > 0)
        .unwrap_or_else(|| topology.logical_cores_count())
}

/// Every engine walks the same corpus, so one pass over it is the work unit every cell reports.
struct Corpus {
    tape: BytesTape<u64, UnifiedAlloc>,
    documents: Vec<&'static [u8]>,
    bytes: u64,
}

impl Corpus {
    fn haystacks(&self) -> AnyBytesTape<'_> {
        AnyBytesTape::View64(self.tape.view())
    }
}

/// One benchmarked engine's answer to "how many matches", so a cell reports work it actually did rather
/// than a count the compiler could fold away.
fn report(matches: usize, corpus_bytes: u64) -> WorkUnits {
    WorkUnits::new(matches as u64, corpus_bytes)
}

// endregion: Harness

// region: Counting

/// Counts every overlapping match of every needle, the cheapest walk each engine offers.
fn bench_counting(
    budget: &BenchBudget,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    core_count: usize,
) {
    let haystacks = corpus.haystacks();
    let mut counts = vec![0usize; corpus.documents.len()];

    for_each_device_pass(core_count, |pass, scope_name, device, _cores| {
        let name = format!("count-{slice}/stringzillas::Substrings<{scope_name}>");
        match Substrings::new(device, needles, CaseSensitivity::Cased) {
            Err(error) => report_skipped(&name, error),
            Ok(engine) => {
                measure_throughput(&name, ReportAs::Bytes, budget, || {
                    let total = engine
                        .count_into(device, &haystacks, OverlapPolicy::Overlapping, &mut counts)
                        .expect("counting failed");
                    report(total, corpus.bytes)
                });
            }
        }

        // Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if !matches!(pass, DeviceChoice::OneCore) {
            return;
        }

        for (kind_name, kind) in [
            ("DFA", AhoCorasickKind::DFA),
            ("ContiguousNFA", AhoCorasickKind::ContiguousNFA),
        ] {
            let Ok(automaton) = AhoCorasick::builder()
                .match_kind(MatchKind::Standard)
                .kind(Some(kind))
                .build(needles)
            else {
                continue;
            };
            measure_throughput(
                &format!("count-{slice}/aho_corasick::find_overlapping_iter<{kind_name}>"),
                ReportAs::Bytes,
                budget,
                || {
                    let total: usize = corpus
                        .documents
                        .iter()
                        .map(|document| automaton.find_overlapping_iter(document).count())
                        .sum();
                    report(total, corpus.bytes)
                },
            );
        }

        if let Ok(automaton) = DoubleArrayAhoCorasick::<u32>::new(needles) {
            measure_throughput(
                &format!("count-{slice}/daachorse::find_overlapping_iter"),
                ReportAs::Bytes,
                budget,
                || {
                    let total: usize = corpus
                        .documents
                        .iter()
                        .map(|document| automaton.find_overlapping_iter(document).count())
                        .sum();
                    report(total, corpus.bytes)
                },
            );
        }
    });
}

/// Counts a leftmost cover, the walk a rewrite needs and the one every engine spells differently.
fn bench_cover(
    budget: &BenchBudget,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    core_count: usize,
) {
    let haystacks = corpus.haystacks();
    let mut counts = vec![0usize; corpus.documents.len()];

    for_each_device_pass(core_count, |pass, scope_name, device, _cores| {
        let name = format!("cover-{slice}/stringzillas::Substrings<{scope_name}>");
        match Substrings::new(device, needles, CaseSensitivity::Cased) {
            Err(error) => report_skipped(&name, error),
            Ok(engine) => {
                measure_throughput(&name, ReportAs::Bytes, budget, || {
                    let total = engine
                        .count_into(
                            device,
                            &haystacks,
                            OverlapPolicy::LeftmostLongest,
                            &mut counts,
                        )
                        .expect("counting failed");
                    report(total, corpus.bytes)
                });
            }
        }

        // Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if !matches!(pass, DeviceChoice::OneCore) {
            return;
        }

        bench_cover_rivals(budget, corpus, needles, slice);
    });
}

/// The single-threaded leftmost-cover rivals, run only in the single-core pass.
fn bench_cover_rivals(budget: &BenchBudget, corpus: &Corpus, needles: &[&[u8]], slice: &str) {
    if let Ok(automaton) = AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostLongest)
        .build(needles)
    {
        measure_throughput(
            &format!("cover-{slice}/aho_corasick::find_iter<LeftmostLongest>"),
            ReportAs::Bytes,
            budget,
            || {
                let total: usize = corpus
                    .documents
                    .iter()
                    .map(|document| automaton.find_iter(document).count())
                    .sum();
                report(total, corpus.bytes)
            },
        );
    }

    if let Ok(automaton) = DoubleArrayAhoCorasickBuilder::new()
        .match_kind(DoubleArrayMatchKind::LeftmostLongest)
        .build(needles)
    {
        let automaton: DoubleArrayAhoCorasick<u32> = automaton;
        measure_throughput(
            &format!("cover-{slice}/daachorse::leftmost_find_iter"),
            ReportAs::Bytes,
            budget,
            || {
                let total: usize = corpus
                    .documents
                    .iter()
                    .map(|document| automaton.leftmost_find_iter(document).count())
                    .sum();
                report(total, corpus.bytes)
            },
        );
    }

    // An alternation of escaped literals is what a caller reaches for before learning about automata.
    // It resolves leftmost-first, so it belongs beside the covers rather than the overlapping walk.
    let alternation: String = needles
        .iter()
        .map(|needle| regex::escape(&String::from_utf8_lossy(needle)))
        .collect::<Vec<_>>()
        .join("|");
    if let Ok(expression) = RegexBuilder::new(&alternation).size_limit(1 << 30).build() {
        measure_throughput(
            &format!("cover-{slice}/regex::find_iter"),
            ReportAs::Bytes,
            budget,
            || {
                let total: usize = corpus
                    .documents
                    .iter()
                    .map(|document| expression.find_iter(document).count())
                    .sum();
                report(total, corpus.bytes)
            },
        );
    }
}

// endregion: Counting

// region: Locating

/// Materializes every match rather than tallying it, which is where the output bandwidth starts to show.
fn bench_locating(
    budget: &BenchBudget,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    core_count: usize,
) {
    let haystacks = corpus.haystacks();
    let mut counts = vec![0usize; corpus.documents.len()];

    // Size the output once, outside every measured loop, exactly as a pipeline would. The probe runs on
    // a single-core scope, which spawns no pool of its own, and is dropped before the passes begin.
    let matches_total = {
        let Some(probe_device) = DeviceChoice::OneCore.open_scope(core_count) else {
            return;
        };
        let Ok(probe) = Substrings::new(&probe_device, needles, CaseSensitivity::Cased) else {
            return;
        };
        let Ok(total) = probe.count_into(
            &probe_device,
            &haystacks,
            OverlapPolicy::Overlapping,
            &mut counts,
        ) else {
            return;
        };
        total
    };
    let mut matches = vec![SubstringsMatch::default(); matches_total];

    for_each_device_pass(core_count, |pass, scope_name, device, _cores| {
        let name = format!("find-{slice}/stringzillas::Substrings<{scope_name}>");
        match Substrings::new(device, needles, CaseSensitivity::Cased) {
            Err(error) => report_skipped(&name, error),
            Ok(engine) => {
                measure_throughput(&name, ReportAs::Bytes, budget, || {
                    let found = engine
                        .find_into(device, &haystacks, OverlapPolicy::Overlapping, &mut matches)
                        .expect("locating failed");
                    report(found, corpus.bytes)
                });
            }
        }

        // Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if !matches!(pass, DeviceChoice::OneCore) {
            return;
        }

        if let Ok(automaton) = AhoCorasick::builder()
            .match_kind(MatchKind::Standard)
            .build(needles)
        {
            let mut oracle_matches: Vec<(usize, usize, usize)> = Vec::with_capacity(matches_total);
            measure_throughput(
                &format!("find-{slice}/aho_corasick::find_overlapping_iter"),
                ReportAs::Bytes,
                budget,
                || {
                    oracle_matches.clear();
                    for (document_index, document) in corpus.documents.iter().enumerate() {
                        for one_match in automaton.find_overlapping_iter(document) {
                            oracle_matches.push((
                                document_index,
                                one_match.start(),
                                one_match.len(),
                            ));
                        }
                    }
                    report(black_box(oracle_matches.len()), corpus.bytes)
                },
            );
        }
    });
}

// endregion: Locating

// region: Rewriting

/// Substitutes every match under a cover, the one verb whose output is another corpus.
fn bench_rewriting(
    budget: &BenchBudget,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    core_count: usize,
) {
    let haystacks = corpus.haystacks();
    // Replacing every needle with itself keeps the output the size of the input, so the cell measures
    // the rewrite rather than an expansion factor, and its result is checkable by eye.
    let replacements: Vec<&[u8]> = needles.to_vec();

    // The output bound is probed on a single-core scope, which spawns no pool, and dropped before the
    // passes begin.
    let bound = {
        let Some(probe_device) = DeviceChoice::OneCore.open_scope(core_count) else {
            return;
        };
        let Ok(probe) = Substrings::new(&probe_device, needles, CaseSensitivity::Cased) else {
            return;
        };
        let Ok(bound) = probe.replace_bound(&replacements, corpus.bytes as usize) else {
            return;
        };
        bound
    };
    let mut output_data = vec![0u8; bound];
    let mut output_offsets = vec![0u64; corpus.documents.len() + 1];

    for_each_device_pass(core_count, |pass, scope_name, device, _cores| {
        let name = format!("replace-{slice}/stringzillas::Substrings<{scope_name}>");
        match Substrings::new(device, needles, CaseSensitivity::Cased) {
            Err(error) => report_skipped(&name, error),
            Ok(engine) => {
                measure_throughput(&name, ReportAs::Bytes, budget, || {
                    let written = engine
                        .replace_into(
                            device,
                            &haystacks,
                            OverlapPolicy::LeftmostLongest,
                            &replacements,
                            &mut output_data,
                            &mut output_offsets,
                        )
                        .expect("rewriting failed");
                    WorkUnits::new(written as u64, corpus.bytes)
                });
            }
        }

        // Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if !matches!(pass, DeviceChoice::OneCore) {
            return;
        }

        if let Ok(automaton) = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(needles)
        {
            measure_throughput(
                &format!("replace-{slice}/aho_corasick::replace_all_bytes"),
                ReportAs::Bytes,
                budget,
                || {
                    let written: usize = corpus
                        .documents
                        .iter()
                        .map(|document| automaton.replace_all_bytes(document, &replacements).len())
                        .sum();
                    WorkUnits::new(black_box(written) as u64, corpus.bytes)
                },
            );
        }
    });
}

// endregion: Rewriting

// region: Scoring

/// Scores every document against the dictionary, which is the query.
///
/// The `bm25` crate answers the same question a different way: it tokenizes the corpus and builds an
/// inverted index once, then every query is a lookup. Build and query are reported as separate cells,
/// since one amortizes over many queries and the other does not exist here at all.
fn bench_scoring(
    budget: &BenchBudget,
    corpus: &Corpus,
    needles: &[&[u8]],
    slice: &str,
    core_count: usize,
) {
    let haystacks = corpus.haystacks();
    let weights = vec![1.0f32; needles.len()];
    let mut scores = vec![0.0f32; corpus.documents.len()];
    let mean_document_length = corpus.bytes as f32 / corpus.documents.len().max(1) as f32;
    let parameters = Bm25Params::normalized(mean_document_length);

    for_each_device_pass(core_count, |_pass, scope_name, device, _cores| {
        let name = format!("bm25-{slice}/stringzillas::Substrings<{scope_name}>");
        let engine = match Substrings::new(device, needles, CaseSensitivity::Cased) {
            Ok(engine) => engine,
            Err(error) => return report_skipped(&name, error),
        };
        measure_throughput(&name, ReportAs::Bytes, budget, || {
            engine
                .score_bm25_into(device, &haystacks, &weights, None, parameters, &mut scores)
                .expect("scoring failed");
            report(black_box(scores.len()), corpus.bytes)
        });
    });
}

/// The index-then-query rival, measured on one slice only: its build is quadratically more expensive
/// than the scan it replaces, so sweeping it across every dictionary size measures tokenization.
fn bench_scoring_index(budget: &BenchBudget, corpus: &Corpus, needles: &[&[u8]], slice: &str) {
    use bm25::{EmbedderBuilder, Scorer};

    let documents: Vec<String> = corpus
        .documents
        .iter()
        .map(|document| String::from_utf8_lossy(document).into_owned())
        .collect();
    let mean_document_length = corpus.bytes as f32 / corpus.documents.len().max(1) as f32;
    let embedder = EmbedderBuilder::<u32>::with_avgdl(mean_document_length).build();

    if slice == VocabularySlice::MostFrequentOnePercent.name() {
        measure_throughput(
            &format!("bm25-{slice}/bm25::Scorer::build"),
            ReportAs::Bytes,
            budget,
            || {
                let mut scorer = Scorer::<usize>::new();
                for (index, document) in documents.iter().enumerate() {
                    scorer.upsert(&index, embedder.embed(document));
                }
                report(black_box(documents.len()), corpus.bytes)
            },
        );
    }

    let mut scorer = Scorer::<usize>::new();
    for (index, document) in documents.iter().enumerate() {
        scorer.upsert(&index, embedder.embed(document));
    }
    // The query is the vocabulary slice itself, the same question the scan answers, so the cell widens
    // with the dictionary instead of measuring one arbitrary document.
    let query: String = needles
        .iter()
        .map(|needle| String::from_utf8_lossy(needle).into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    let query_embedding = embedder.embed(&query);
    measure_throughput(
        &format!("bm25-{slice}/bm25::Scorer::matches"),
        ReportAs::Bytes,
        budget,
        || {
            let matched = scorer.matches(&query_embedding);
            report(black_box(matched.len()), corpus.bytes)
        },
    );
}

// endregion: Scoring

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();
    println!("Multi-Pattern Search Benchmarks");
    println!("- szs::Substrings: one dictionary walked over every haystack, on CPU cores or a GPU");
    println!(
        "- aho_corasick / daachorse / regex: single-threaded CPU rivals, the <1cpu> row's peers"
    );

    // Leaked so the vocabulary and the document slices below borrow for the whole run rather than
    // threading a lifetime through every sweep cell.
    let tape: &'static _ = Box::leak(Box::new(
        load_dataset("lines", COMPUTE_BOUND_SLICE, "data/xlsum/xlsum.csv").unwrap_nice(),
    ));
    let corpus_bytes: &'static [u8] = tape.parent();

    let vocabulary = build_vocabulary(corpus_bytes);
    println!(
        "- vocabulary: {} terms after dropping hapax and the top one percent",
        vocabulary.len()
    );
    if vocabulary.is_empty() {
        eprintln!("The corpus yielded no vocabulary; is the dataset path right?");
        return;
    }

    // One unified-memory tape serves every scope, so the GPU cells need no copy of their own and the
    // measured region is the walk rather than the plumbing.
    let mut unified_tape: BytesTape<u64, UnifiedAlloc> = BytesTape::new_in(UnifiedAlloc);
    unified_tape
        .extend(tape.iter())
        .expect("Failed to build the unified-memory corpus tape");
    let documents: Vec<&'static [u8]> = tape.iter().collect();
    let corpus = Corpus {
        tape: unified_tape,
        bytes: documents.iter().map(|document| document.len() as u64).sum(),
        documents,
    };
    println!(
        "- corpus: {} documents, {} bytes",
        corpus.documents.len(),
        corpus.bytes
    );

    let topology = fu::Topology::new().expect("Failed to probe the CPU topology");
    let core_count = resolve_core_count(&topology);

    let budget = BenchBudget::from_env(1.0, 5.0);
    for slice in VOCABULARY_SLICES {
        let needles = needle_slice_of(&vocabulary, slice);
        let needle_bytes: usize = needles.iter().map(|needle| needle.len()).sum();
        println!(
            "\n# {} - {} needles, {} bytes",
            slice.name(),
            needles.len(),
            needle_bytes
        );

        bench_counting(&budget, &corpus, &needles, slice.name(), core_count);
        bench_cover(&budget, &corpus, &needles, slice.name(), core_count);
        bench_locating(&budget, &corpus, &needles, slice.name(), core_count);
        bench_rewriting(&budget, &corpus, &needles, slice.name(), core_count);
        bench_scoring(&budget, &corpus, &needles, slice.name(), core_count);
        bench_scoring_index(&budget, &corpus, &needles, slice.name());
    }
}
