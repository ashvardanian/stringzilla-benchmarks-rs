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
- `bm25`, `bm25s`, `rank_bm25`, `sklearn.TfidfVectorizer` and `cuml.TfidfVectorizer` are index-then-query scorers, present in the scoring group only. `rank_bm25` is the most installed of them, `TfidfVectorizer` is the TF-IDF reference rather than a BM25, and cuML is its GPU counterpart. No comparable GPU BM25 library exists.

Throughput is reported in __MB/s of document bytes consumed__, the rate at which the automaton advances over the corpus.

## Methodology

Each table fixes one input shape: `xlsum.csv` split into lines as documents, searched with one slice of that corpus's own frequency-ordered vocabulary as the dictionary.
The vocabulary is whitespace-cut words of 3 to 32 bytes, with every term occurring once and the most frequent one percent dropped; slices are taken by term count, so the frequent and the rare slice of the same percentage hold the same number of needles and differ only in byte totals, since Zipf makes frequent terms short and their automata correspondingly shallower.
This is the recipe `bench/substrings.cpp` uses inside StringZilla, so these numbers sit beside the cross-backend tables in `include/stringzillas/substrings/README.md` rather than floating free.

Each table below fixes one vocabulary slice, so a row is an engine and a column is the question asked of it.

Five operations are measured.
__Count__ tallies every overlapping match.
__Cover__ resolves a leftmost-longest cover, which is a different walk and should not be differenced against counting.
__Find__ materializes every overlapping match rather than tallying it, which is where output bandwidth starts to show.
__Replace__ rewrites every document, substituting each needle with itself so the output stays the size of the input and the cell measures the rewrite rather than an expansion factor.
__BM25__ reduces each document to one float, weighting every needle uniformly, so each slice heading doubles as a query size.

Two framing rules, without which the table lies.

__The head-to-head row is `<1xSPR>`.__
Every rival is single-threaded and CPU-only, so `<16xSPR>` and `<H100>` are what the engine adds, not what it beats them by.

__Scoring is scan versus index.__
`bm25`, `bm25s`, `rank_bm25` and `sklearn.TfidfVectorizer` tokenize the corpus and build an inverted index once, then answer many queries against it; `stringzillas::Substrings` scans raw bytes against a fixed dictionary and builds nothing per query.
Build and query are therefore separate rows, and a query rate is only meaningful once the build above it has been paid for.
The regime this engine is for is a corpus seen once — a firehose being filtered, or a candidate set handed over by a first-stage retriever — where the index does not exist yet and building it is the whole cost.

Uncased matching is not benchmarked here.
`aho-corasick`'s `ascii_case_insensitive` folds ASCII only, while `stringzillas::Substrings` applies full Unicode case folding to both sides of the comparison, so the two solve different problems.
The cross-backend uncased numbers live in StringZilla's own tables.

## Performance

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

One run per language produced every cell: 64 MiB of `xlsum.csv` split into lines — 13,699 documents — on one Xeon Platinum 8468 and one idle H100 80GB HBM3.
A `–` cell is an operation that engine is never asked for, not a slow result: `daachorse` has no rewrite, `regex` is only ever asked for a leftmost cover, the `aho_corasick` kind split is a counting-only knob, and the index scorers answer the scoring question alone.
Python's `re.finditer` is the one omission that is about cost rather than coverage — see the note under __Reproducing__.

The five tables below cover the four walk-shaped operations.
Scoring gets a table of its own after them, because it is the one place where a rival can answer the same question by a different route: an index engine pays a __build__ before it can answer anything, then queries cheaply, while a scan answers from raw bytes with no build at all.
That table carries the build in one column, since a build never sees the dictionary and does not vary with the slice, and the query in one column per vocabulary size, since a query takes the slice itself and widens with it.

Three results the tables state plainly.

__The dictionary's shape decides the single-core winner, not the engine.__
`stringzillas::Substrings<1xSPR>` takes counting, cover and find on the frequent one percent, and `daachorse` takes counting on the other four slices — by 3.2x on the rare one percent, where its double array holds a shallow automaton that stays in cache.
The rewrite column never goes to us on one core: `aho_corasick::replace_all_bytes` leads it on every slice, since our rewrite pays for a leftmost cover the counting path does not.

__Scoring can outrun counting on the GPU.__
On the rare one percent the H100 scores BM25 at 47.3 GB/s against 32.2 GB/s for the count it is built on, because scoring reduces each document to one float while counting materializes a tally per needle.

__The vectorizer rows are not a like-for-like race, and their query column should not be read as one.__
`TfidfVectorizer` lowercases and cuts tokens on `\b\w\w+\b`, so 21.5% of our needles never reach its vocabulary at all — punctuation-attached words and non-Latin scripts among them — and it answers about four fifths of the question the scan answers.
It also matches whole tokens where the scan matches substrings anywhere in a document, and scores a TF-IDF dot product rather than BM25 with saturation.
Its query touches an index of postings rather than the corpus, so an equal MB/s does not mean equal work; only the build column, which is one honest pass over the corpus, compares directly.
Read those rows as the cost of the index-then-query route, not as a faster answer to the same question.

__The two harnesses agree where they should.__
`stringzillas::Substrings<1xSPR>` counts at 239.8 MB/s from Rust and 240.6 MB/s from Python over the same corpus and dictionary, which is what makes the Python rivals below comparable to the Rust ones above rather than a separate table.
The two vocabularies differ by 16 terms out of 346,205, since the Rust harness cuts words on ASCII whitespace and the Python one on Unicode whitespace.

### Most Frequent 1% of the Vocabulary

| Library                            |             Count |             Cover |              Find |          Replace |
| :--------------------------------- | ----------------: | ----------------: | ----------------: | ---------------: |
| Rust                               |                   |                   |                   |                  |
| `aho_corasick<DFA>`                |        181.1 MB/s |                 – |                 – |                – |
| `aho_corasick<ContiguousNFA>`      |        106.1 MB/s |                 – |                 – |                – |
| `aho_corasick`                     |                 – |        106.0 MB/s |        105.6 MB/s |       100.6 MB/s |
| `daachorse`                        |        129.2 MB/s |        128.8 MB/s |                 – |                – |
| `regex::find_iter`                 |                 – |        106.2 MB/s |                 – |                – |
| `stringzillas::Substrings<1xSPR>`  |        239.8 MB/s |        183.6 MB/s |        112.9 MB/s |        89.1 MB/s |
| `stringzillas::Substrings<16xSPR>` |      2,520.0 MB/s |      1,380.0 MB/s |      1,180.0 MB/s |       719.4 MB/s |
| `stringzillas::Substrings<H100>`   | __28,610.0 MB/s__ | __21,230.0 MB/s__ | __13,130.0 MB/s__ | __8,180.0 MB/s__ |
|                                    |                   |                   |                   |                  |
| Python                             |                   |                   |                   |                  |
| `pyahocorasick`                    |         23.0 MB/s |                 – |                 – |                – |
| `ahocorasick_rs`                   |         81.9 MB/s |         84.3 MB/s |         81.2 MB/s |                – |
| `re.finditer`                      |                 – |        188.4 kB/s |                 – |                – |
| `stringzillas.Substrings<1xSPR>`   |        241.2 MB/s |        184.0 MB/s |         74.2 MB/s |        71.7 MB/s |
| `stringzillas.Substrings<16xSPR>`  |      2,210.0 MB/s |      1,380.0 MB/s |        618.3 MB/s |       292.3 MB/s |
| `stringzillas.Substrings<H100>`    | __27,360.0 MB/s__ | __19,420.0 MB/s__ |  __3,970.0 MB/s__ | __1,030.0 MB/s__ |

### Most Frequent 10% of the Vocabulary

| Library                            |             Count |            Cover |             Find |          Replace |
| :--------------------------------- | ----------------: | ---------------: | ---------------: | ---------------: |
| Rust                               |                   |                  |                  |                  |
| `aho_corasick<DFA>`                |         54.0 MB/s |                – |                – |                – |
| `aho_corasick<ContiguousNFA>`      |         74.0 MB/s |                – |                – |                – |
| `aho_corasick`                     |                 – |        77.2 MB/s |        71.4 MB/s |        65.0 MB/s |
| `daachorse`                        |        111.9 MB/s |       126.7 MB/s |                – |                – |
| `regex::find_iter`                 |                 – |        77.4 MB/s |                – |                – |
| `stringzillas::Substrings<1xSPR>`  |         90.8 MB/s |        65.1 MB/s |        42.5 MB/s |        30.4 MB/s |
| `stringzillas::Substrings<16xSPR>` |        780.9 MB/s |       547.6 MB/s |       392.6 MB/s |       247.8 MB/s |
| `stringzillas::Substrings<H100>`   | __15,220.0 MB/s__ | __8,150.0 MB/s__ | __3,810.0 MB/s__ | __4,340.0 MB/s__ |
|                                    |                   |                  |                  |                  |
| Python                             |                   |                  |                  |                  |
| `pyahocorasick`                    |         15.1 MB/s |                – |                – |                – |
| `ahocorasick_rs`                   |         47.3 MB/s |        53.3 MB/s |        50.3 MB/s |                – |
| `stringzillas.Substrings<1xSPR>`   |         95.1 MB/s |        66.8 MB/s |        27.7 MB/s |        29.4 MB/s |
| `stringzillas.Substrings<16xSPR>`  |        794.2 MB/s |       496.9 MB/s |       158.9 MB/s |       195.2 MB/s |
| `stringzillas.Substrings<H100>`    | __15,230.0 MB/s__ | __8,120.0 MB/s__ |   __617.2 MB/s__ | __2,160.0 MB/s__ |

### Least Frequent 1% of the Vocabulary

| Library                            |             Count |             Cover |              Find |          Replace |
| :--------------------------------- | ----------------: | ----------------: | ----------------: | ---------------: |
| Rust                               |                   |                   |                   |                  |
| `aho_corasick<DFA>`                |        465.0 MB/s |                 – |                 – |                – |
| `aho_corasick<ContiguousNFA>`      |        417.7 MB/s |                 – |                 – |                – |
| `aho_corasick`                     |                 – |        435.9 MB/s |        432.0 MB/s |       430.7 MB/s |
| `daachorse`                        |      1,170.0 MB/s |      1,070.0 MB/s |                 – |                – |
| `regex::find_iter`                 |                 – |        440.6 MB/s |                 – |                – |
| `stringzillas::Substrings<1xSPR>`  |        360.2 MB/s |        354.9 MB/s |        182.7 MB/s |       175.2 MB/s |
| `stringzillas::Substrings<16xSPR>` |      4,090.0 MB/s |      4,040.0 MB/s |      2,000.0 MB/s |     2,010.0 MB/s |
| `stringzillas::Substrings<H100>`   | __32,170.0 MB/s__ | __25,980.0 MB/s__ | __28,160.0 MB/s__ | __8,540.0 MB/s__ |
|                                    |                   |                   |                   |                  |
| Python                             |                   |                   |                   |                  |
| `pyahocorasick`                    |          7.6 MB/s |                 – |                 – |                – |
| `ahocorasick_rs`                   |        228.5 MB/s |        237.7 MB/s |        236.4 MB/s |                – |
| `stringzillas.Substrings<1xSPR>`   |        376.4 MB/s |        360.3 MB/s |        125.9 MB/s |       157.4 MB/s |
| `stringzillas.Substrings<16xSPR>`  |      3,680.0 MB/s |      3,420.0 MB/s |      1,340.0 MB/s |     1,010.0 MB/s |
| `stringzillas.Substrings<H100>`    | __25,040.0 MB/s__ | __26,680.0 MB/s__ | __15,400.0 MB/s__ | __2,670.0 MB/s__ |

### Least Frequent 10% of the Vocabulary

| Library                            |             Count |             Cover |             Find |          Replace |
| :--------------------------------- | ----------------: | ----------------: | ---------------: | ---------------: |
| Rust                               |                   |                   |                  |                  |
| `aho_corasick<DFA>`                |        141.2 MB/s |                 – |                – |                – |
| `aho_corasick<ContiguousNFA>`      |        174.6 MB/s |                 – |                – |                – |
| `aho_corasick`                     |                 – |        167.7 MB/s |       170.8 MB/s |       153.3 MB/s |
| `daachorse`                        |        277.9 MB/s |        279.3 MB/s |                – |                – |
| `regex::find_iter`                 |                 – |        178.1 MB/s |                – |                – |
| `stringzillas::Substrings<1xSPR>`  |        171.5 MB/s |        150.6 MB/s |        86.1 MB/s |        73.3 MB/s |
| `stringzillas::Substrings<16xSPR>` |      1,740.0 MB/s |      1,380.0 MB/s |       872.2 MB/s |       702.1 MB/s |
| `stringzillas::Substrings<H100>`   | __18,050.0 MB/s__ | __11,380.0 MB/s__ | __9,660.0 MB/s__ | __4,810.0 MB/s__ |
|                                    |                   |                   |                  |                  |
| Python                             |                   |                   |                  |                  |
| `pyahocorasick`                    |          5.0 MB/s |                 – |                – |                – |
| `ahocorasick_rs`                   |        108.8 MB/s |        114.8 MB/s |       121.1 MB/s |                – |
| `stringzillas.Substrings<1xSPR>`   |        182.2 MB/s |        153.1 MB/s |        61.2 MB/s |        75.1 MB/s |
| `stringzillas.Substrings<16xSPR>`  |      1,250.0 MB/s |      1,140.0 MB/s |       538.3 MB/s |       423.9 MB/s |
| `stringzillas.Substrings<H100>`    | __17,960.0 MB/s__ | __10,900.0 MB/s__ | __3,990.0 MB/s__ | __2,350.0 MB/s__ |

### Entire Vocabulary

| Library                            |            Count |            Cover |             Find |          Replace |
| :--------------------------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| Rust                               |                  |                  |                  |                  |
| `aho_corasick<DFA>`                |        24.6 MB/s |                – |                – |                – |
| `aho_corasick<ContiguousNFA>`      |        40.9 MB/s |                – |                – |                – |
| `aho_corasick`                     |                – |        42.3 MB/s |        40.5 MB/s |        35.7 MB/s |
| `daachorse`                        |        65.6 MB/s |        81.4 MB/s |                – |                – |
| `regex::find_iter`                 |                – |        49.6 MB/s |                – |                – |
| `stringzillas::Substrings<1xSPR>`  |        43.4 MB/s |        26.7 MB/s |        22.2 MB/s |        14.4 MB/s |
| `stringzillas::Substrings<16xSPR>` |       552.6 MB/s |       315.7 MB/s |       248.1 MB/s |       145.0 MB/s |
| `stringzillas::Substrings<H100>`   | __9,310.0 MB/s__ | __4,000.0 MB/s__ | __1,670.0 MB/s__ | __2,020.0 MB/s__ |
|                                    |                  |                  |                  |                  |
| Python                             |                  |                  |                  |                  |
| `pyahocorasick`                    |         8.1 MB/s |                – |                – |                – |
| `ahocorasick_rs`                   |        24.2 MB/s |        32.3 MB/s |        25.4 MB/s |                – |
| `stringzillas.Substrings<1xSPR>`   |        62.6 MB/s |        41.1 MB/s |        13.4 MB/s |        17.2 MB/s |
| `stringzillas.Substrings<16xSPR>`  |       602.5 MB/s |       264.0 MB/s |       101.0 MB/s |       145.7 MB/s |
| `stringzillas.Substrings<H100>`    | __8,740.0 MB/s__ | __3,780.0 MB/s__ |   __255.0 MB/s__ | __1,340.0 MB/s__ |

### BM25 Scoring Across Vocabulary Sizes

| Library                            | Index Build |  Most Frequent 1% | Most Frequent 10% | Least Frequent 1% | Least Frequent 10% | Entire Vocabulary |
| :--------------------------------- | ----------: | ----------------: | ----------------: | ----------------: | -----------------: | ----------------: |
| Rust                               |             |                   |                   |                   |                    |                   |
| `stringzillas::Substrings<1xSPR>`  |           – |        214.5 MB/s |         73.3 MB/s |        366.2 MB/s |         165.5 MB/s |         36.6 MB/s |
| `stringzillas::Substrings<16xSPR>` |           – |      2,300.0 MB/s |        702.8 MB/s |      3,910.0 MB/s |       1,530.0 MB/s |        375.2 MB/s |
| `stringzillas::Substrings<H100>`   |           – | __20,870.0 MB/s__ |  __8,460.0 MB/s__ | __47,270.0 MB/s__ |  __17,130.0 MB/s__ |  __4,460.0 MB/s__ |
| `bm25::Scorer`                     |   13.6 MB/s |          5.6 MB/s |        518.0 kB/s |          5.9 MB/s |         474.9 kB/s |                 – |
|                                    |             |                   |                   |                   |                    |                   |
| Python                             |             |                   |                   |                   |                    |                   |
| `stringzillas.Substrings<1xSPR>`   |           – |        218.6 MB/s |         88.5 MB/s |        377.0 MB/s |         191.8 MB/s |         45.9 MB/s |
| `stringzillas.Substrings<16xSPR>`  |           – |      2,180.0 MB/s |        685.9 MB/s |      3,740.0 MB/s |       1,660.0 MB/s |        481.0 MB/s |
| `stringzillas.Substrings<H100>`    |           – |     29,010.0 MB/s |     12,200.0 MB/s | __49,860.0 MB/s__ |      23,180.0 MB/s |      7,120.0 MB/s |
| `bm25s.BM25`                       |   15.7 MB/s |      8,730.0 MB/s |        854.6 MB/s |      9,480.0 MB/s |         972.7 MB/s |         85.7 MB/s |
| `rank_bm25.BM25Okapi`              |   19.8 MB/s |          6.1 MB/s |        633.2 kB/s |          6.5 MB/s |         631.8 kB/s |         64.8 kB/s |
| `sklearn.TfidfVectorizer`          |   14.5 MB/s |      3,340.0 MB/s |      1,740.0 MB/s |      5,250.0 MB/s |       3,810.0 MB/s |      2,100.0 MB/s |
| `cuml.TfidfVectorizer<H100>`       |  343.1 MB/s | __39,610.0 MB/s__ | __25,130.0 MB/s__ |     35,440.0 MB/s |  __30,290.0 MB/s__ |  __9,820.0 MB/s__ |

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
PYTHONPATH=. STRINGWARS_DATASET=data/xlsum/xlsum.csv \
    STRINGWARS_DATASET_LIMIT=64mb \
    uv run --no-project python substrings/bench.py
```

`STRINGWARS_FILTER` narrows a run to one operation, one vocabulary slice, or one library, since every benchmark name is `<operation>-<slice>/<library>::<call>`:

```sh
STRINGWARS_FILTER='cover-most_frequent_1%' cargo bench --features bench_substrings --bench bench_substrings
PYTHONPATH=. STRINGWARS_FILTER='stringzillas' uv run --no-project python substrings/bench.py
```

A note on the Python numbers: `pyahocorasick` and `ahocorasick_rs` return one Python object per match, so a dense dictionary charges them interpreter overhead the byte-level engines never pay, and their gap widens with the match count rather than with the corpus size.
`stringzillas.Substrings` returns NumPy arrays, so its per-match cost stays in C.

`re.finditer` carries no cell in the tables above, and the code measures it on the narrowest dictionary alone.
Python's `re` walks an alternation branch by branch with no literal prefilter, so at 3,462 needles a single pass over 64 MiB ran past twelve minutes without finishing — under 0.09 MB/s, roughly 1,200x behind Rust's `regex` on the same slice.
Rust's `regex` carries such a prefilter and sweeps all five slices, which is the comparison worth drawing.
Reproduce the Python row with a smaller `--dataset-limit`, where it costs seconds rather than hours.
