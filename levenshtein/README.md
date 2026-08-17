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

## Reproducing the correctness check

Build StringZilla first, then compile the query generator and the two complete-output runners from the StringWars root. Pin the RapidFuzz revision used by the final run.

```bash
cmake -S ../StringZilla -B ../StringZilla/build -DCMAKE_BUILD_TYPE=Release
cmake --build ../StringZilla/build -j --target stringzillas_cpus_static

g++ -std=c++20 -O3 -DNDEBUG -I ../StringZilla/include \
    levenshtein/queries.cpp -o levenshtein_queries

g++ -std=c++20 -O3 -DNDEBUG -march=native -DSZ_DYNAMIC_DISPATCH=1 \
    -I ../StringZilla/include -I ../StringZilla/forkunion/include \
    levenshtein/stringzilla.cpp ../StringZilla/build/libstringzillas_cpus_static.a \
    ../StringZilla/build/forkunion/libforkunion_static.a -pthread -o stringzilla_levenshtein

g++ -std=c++20 -O3 -DNDEBUG -march=native -I ../rapidfuzz-cpp \
    levenshtein/rapidfuzz.cpp -o rapidfuzz_levenshtein

./levenshtein_queries words_alpha.txt queries.txt 10000 mixed 243

SZ_LEVENSHTEIN_MAX_DISTANCE=1 SZ_LEVENSHTEIN_REPEATS=20 \
    SZ_LEVENSHTEIN_MODES=warm,steady,latency \
    ./stringzilla_levenshtein words_alpha.txt queries.txt 10000 stringzilla-results

SZ_LEVENSHTEIN_MAX_DISTANCE=2 SZ_LEVENSHTEIN_REPEATS=20 \
    SZ_LEVENSHTEIN_MODES=warm,steady,latency \
    ./stringzilla_levenshtein words_alpha.txt queries.txt 10000 stringzilla-results

RF_MAX_DISTANCE=2 RF_REPEATS=20 RF_MODE=materialized \
    ./rapidfuzz_levenshtein words_alpha.txt queries.txt 10000 rapidfuzz-results

cmp stringzilla-results.k1.bin rapidfuzz-results.k1.bin
cmp stringzilla-results.k2.bin rapidfuzz-results.k2.bin
```

To check the larger-bound crossover with one index, run:

```bash
SZ_LEVENSHTEIN_INDEX_PLAN=shared SZ_LEVENSHTEIN_MAX_DISTANCE=10 \
    SZ_LEVENSHTEIN_REPEATS=20 SZ_LEVENSHTEIN_MODES=warm,steady \
    ./stringzilla_levenshtein words_alpha.txt queries.txt 10000 stringzilla-wide

RF_MAX_DISTANCE=10 RF_REPEATS=20 RF_MODE=materialized \
    ./rapidfuzz_levenshtein words_alpha.txt queries.txt 10000 rapidfuzz-wide

for k in 1 2 3 4 5 6 7 8 9 10; do
    cmp "stringzilla-wide.k${k}.bin" "rapidfuzz-wide.k${k}.bin"
done
```

Run the Rust adapters through the normal benchmark target:

```bash
RUSTFLAGS="-C target-cpu=native" STRINGWARS_REPEATS=20 \
    cargo bench --features bench_levenshtein --bench bench_levenshtein -- \
    words_alpha.txt queries.txt 10000
```

The final result tables and raw run artifacts are added only after this protocol passes on every reported machine.
