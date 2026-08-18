# Repeated Levenshtein Dictionary Search

This benchmark covers the case where a dictionary is built once and searched many times. It is separate from the dense distance matrices in [`similarities/`](../similarities/).

Each exact result contains the query ID, the original dictionary ID, and the plain Levenshtein distance. Duplicate dictionary entries keep separate IDs. An adjacent swap counts as two edits.

## Comparisons

Direct timing comparisons require the same complete output:

* StringZilla returns every original ID and exact distance.
* RapidFuzz scans the dictionary and materializes the same result. It is also the correctness oracle.
* SymSpell has an exact compatibility mode that checks its suggestions again with allocation-free plain Levenshtein distance. It is comparable only on unique lowercase dictionaries.

The native SymSpell, Rust `fst`, Tantivy, and Lucene modes return different information. Their results provide useful ecosystem context, but they are not used for direct speedup claims.

Every comparable runner writes the same binary result format. The files must match before timings are reported.

## Queries

`queries.cpp` creates deterministic queries from an existing dictionary. The mixed workload gives equal weight to:

* exact queries;
* one substitution, insertion, or deletion;
* two substitutions, insertions, or deletions;
* one insertion plus one deletion;
* one adjacent swap;
* a five-symbol extension of the sampled source word.

These labels describe how the query was created from one source word. They do not assume that the query has no other dictionary matches. The result oracle decides the complete answer.

Final runs should include short English words, longer URLs, DNA strings, valid non-ASCII text, and a duplicate-heavy synthetic dictionary. Each run records the dictionary and query hashes.

## Timing

StringZilla reports four separate measurements:

* `cold_end_to_end` starts with a fresh index reader and includes output sizing, allocation, retry, and materialization. Dictionary construction is reported separately.
* `warm_presized` measures repeated batches after scratch and exact output capacity are available.
* `steady_growable` starts from eight output slots per query and includes a resize and retry when needed.
* `single_query_latency` reports p50, p95, and p99 for one-query calls with a reusable grow-only output buffer.

Threshold-specialized runs build separate indexes for `k=1`, `k=2`, and `k=4`. Shared-index runs build once for `k=4` and query the same index at every bound from one through four.

The same complete-output comparison can sweep larger bounds by setting both runners to the same maximum. This is kept separate from the low-bound ecosystem table because several indexed tools only support small edit distances. It checks that StringZilla changes its internal search path without changing results or developing a performance cliff.

Published runs use at least 20 measured repetitions, keep raw output, randomize runner order, pin CPU and memory placement, and record compiler versions, dependency revisions, CPU frequency settings, result counts, output bytes, build time, retained index size, peak build memory, and reader scratch. Warm and cold results are never combined into one number.

## Running the comparison

The shared low-bound track expects a unique lowercase ASCII dictionary because that is the common contract supported by all six adapters. From the StringWars root:

```bash
python3 levenshtein/run.py words_alpha.txt \
    --stringzilla-root ../StringZilla \
    --rapidfuzz-root ../rapidfuzz-cpp \
    --cpu 2
```

The command builds every adapter, generates 10,000 mixed queries with seed 243, and validates results before recording timings. StringZilla, RapidFuzz, and exact SymSpell must write identical result files at `k=1` and `k=2`. FST, Tantivy, and Lucene must return the same totals. Missing output, zero work, or any mismatch stops the run.

The default 20 repetitions run each adapter in a deterministic shuffled order. Every repetition starts a new process. StringZilla uses separate threshold-specific indexes for `k=1` and `k=2` and keeps cold, warm, growable, and single-query latency measurements separate.

`target/levenshtein/manifest.json` records input hashes, repository revisions, tool versions, execution order, commands, relevant environment settings, wall time, and peak process memory. The neighboring `logs/` directory keeps the runners' original key-value output. These generated files are archived with a published run instead of committed to the repository.

CI uses the same command with the small dictionary under `levenshtein/data/`, one repetition, and `--validation-only`. `--skip-build` is available when the expected products already exist under the selected build directory.

The larger-bound crossover remains a separate StringZilla and RapidFuzz run. It is not mixed into this command because SymSpell, Tantivy, and Lucene stop at small bounds, and because high-result sweeps answer a different performance question.
