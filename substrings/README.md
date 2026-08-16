# Multi-Pattern Search Benchmarks

Benchmarks for matching a __whole dictionary of needles__ against a __whole collection of documents__ in one pass — the workhorse of log scanning, protocol dispatch, content filtering, and signature matching.

## Overview

Every engine here is an Aho-Corasick automaton compiled once from the needle set and reused across every later call, so the dictionary is paid for outside the measured loop.
What separates them is the automaton's memory layout and what the walk is allowed to report.

- `stringzillas::Substrings` splits a goto-completed automaton into a dense 256-wide row per frequently-visited state and a double array for the rest, so a step is one load with no failure-following at runtime. It runs across a slice of CPU cores or a CUDA GPU, advancing many documents concurrently rather than making any single document faster.
- `aho-corasick` is the reference Rust implementation, benchmarked as a dense `DFA` and as a `ContiguousNFA` — the two ends of the same speed-for-memory trade the tier split makes internally.
- `daachorse` is a double-array Aho-Corasick, the direct analog of our cold tier on its own.
- `regex` over an alternation of escaped literals is the baseline most people actually write, and its literal prefilters make it a real competitor rather than a strawman.
- `pyahocorasick` and `ahocorasick_rs` are the Python side's C and Rust automata.
- `bm25` and `bm25s` are index-then-query scorers, present in the scoring group only.

Throughput is reported in __MB/s of document bytes consumed__, the rate at which the automaton advances over the corpus.

## Methodology

Each table fixes one input shape: `xlsum.csv` split into lines as documents, searched with one slice of that corpus's own frequency-ordered vocabulary as the dictionary.
The vocabulary is whitespace-cut words of 3 to 32 bytes, with every term occurring once and the most frequent one percent dropped; slices are taken by term count, so the frequent and the rare slice of the same percentage hold the same number of needles and differ only in byte totals, since Zipf makes frequent terms short and their automata correspondingly shallower.
This is the recipe `bench/substrings.cpp` uses inside StringZilla, so these numbers sit beside the cross-backend tables in `include/stringzillas/substrings/README.md` rather than floating free.

Four operations are measured.
__Count__ tallies every overlapping match.
__Cover__ resolves a leftmost-longest cover, which is a different walk and should not be differenced against counting.
__Replace__ rewrites every document, substituting each needle with itself so the output stays the size of the input and the cell measures the rewrite rather than an expansion factor.
__BM25__ reduces each document to one float, weighting every needle uniformly, so each slice heading doubles as a query size.

Two framing rules, without which the table lies.

__The head-to-head row is `<1cpu>`.__
Every rival is single-threaded and CPU-only, so `<Ncpu>` and `<1gpu>` are what the engine adds, not what it beats them by.

__Scoring is scan versus index.__
`bm25` and `bm25s` tokenize the corpus and build an inverted index once, then answer many queries in microseconds; `stringzillas::Substrings` scans raw bytes against a fixed dictionary and builds nothing per query.
Build and query are therefore reported as separate cells, and neither is comparable to the scan on its own.
The regime this engine is for is a corpus seen once — a firehose being filtered, or a candidate set handed over by a first-stage retriever — where the index does not exist yet and building it is the whole cost.

Uncased matching is not benchmarked here.
`aho-corasick`'s `ascii_case_insensitive` folds ASCII only, while `stringzillas::Substrings` applies full Unicode case folding to both sides of the comparison, so the two solve different problems.
The cross-backend uncased numbers live in StringZilla's own tables.

## Performance

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

A `-` cell is a measurement not yet taken.

| Library                                        | Count | Cover | Replace | BM25 |
| :--------------------------------------------- | ----: | ----: | ------: | ---: |
| `aho_corasick::DFA<1xSPR>`                      |     - |     - |       - |    - |
| `aho_corasick::ContiguousNFA<1xSPR>`            |     - |     - |       - |    - |
| `daachorse<1xSPR>`                              |     - |     - |       - |    - |
| `regex::find_iter<1xSPR>`                       |     - |     - |       - |    - |
| `stringzillas::Substrings<1xSPR>`               |     - |     - |       - |    - |
| `stringzillas::Substrings<16xSPR>`              |     - |     - |       - |    - |
| `stringzillas::Substrings<H100>`                |     - |     - |       - |    - |

> Measured over the most frequent one percent of the vocabulary.

## Reproducing

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=data/xlsum/xlsum.csv \
    STRINGWARS_DATASET_LIMIT=64mb \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_substrings --bench bench_substrings
```

With a CUDA GPU, add the feature; the `<1gpu>` cells stay silent without one:

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=data/xlsum/xlsum.csv \
    cargo bench --features "cuda bench_substrings" --bench bench_substrings
```

The Python side mirrors the same sweep:

```sh
uv pip install -r requirements.txt
STRINGWARS_DATASET=data/xlsum/xlsum.csv \
    STRINGWARS_DATASET_LIMIT=64mb \
    uv run --no-project python substrings/bench.py
```

`STRINGWARS_FILTER` narrows a run to one operation, one vocabulary slice, or one library, since every benchmark name is `<operation>-<slice>/<library>::<call>`:

```sh
STRINGWARS_FILTER='cover-most_frequent_1%' cargo bench --features bench_substrings --bench bench_substrings
STRINGWARS_FILTER='stringzillas' uv run --no-project python substrings/bench.py
```

A note on the Python numbers: `pyahocorasick` and `ahocorasick_rs` return one Python object per match, so a dense dictionary charges them interpreter overhead the byte-level engines never pay, and their gap widens with the match count rather than with the corpus size.
`stringzillas.Substrings` returns NumPy arrays, so its per-match cost stays in C.
