# Multi-Way Hashing & Probabilistic Membership Benchmarks

Benchmarks for the hashing that backs probabilistic membership structures — Bloom filters and XOR/binary-fuse filters — across Rust and Python.

## Overview

A membership filter needs several independent hashes of the same short key: a _d_-ary cuckoo table probes _d_ buckets, a Bloom filter sets _k_ bits, a Count-Min sketch updates _k_ counters.
Most hash functions can only produce those by being called _k_ times, re-reading the key on each call.
StringZilla's `hash_multiseed` instead normalizes the key into AES blocks once and replays cheap per-seed rounds, emitting all _k_ hashes in a single pass.
This suite measures that primitive directly, then checks whether feeding StringZilla hashes into real filters actually helps.

Only decision-relevant comparisons are kept.
A hash that is slower, weaker, _and_ lacks a multi-seed path adds nothing, so there are no strawman columns for every library — the two baselines that change the conclusion are StringZilla's own naive per-seed calls (isolating what the multi-seed path amortizes) and `xxh3_128` using its full 128-bit output per call (the strongest single-pass alternative).
The cheap `g_i = h1 + i·h2` double-hashing shortcut that production Bloom filters use is deliberately left out: it fabricates extra _linearly dependent_ bits from one hash, so crediting them as digest bits would flatter it.

## Multi-Hash Generation

Producing a digest of independent hash bits per word over `xlsum.csv`, where the column axis is the digest size in bits.
Throughput is reported in produced digest bits/s so the 64-bit and 128-bit hashes line up on one scale: `stringzilla::hash` is StringZilla's hash called once per 64-bit seed; `xxh3::xxh3_128` is called once per seed and keeps its full 128-bit output, so it re-prepares the input only every 128 bits.
Every value is independent and each variant writes into one preallocated buffer (a NumPy array in Python), so neither double-hashing nor per-call allocation skews the comparison.

### Intel Xeon4 Sapphire Rapids

| Variant                       |            128 bits |           256 bits |           512 bits |          1024 bits |
| ----------------------------- | ------------------: | -----------------: | -----------------: | -----------------: |
| Rust                          |                     |                    |                    |                    |
| `xxh3::xxh3_128`              |       8.26 G bits/s |     14.57 G bits/s |     21.76 G bits/s |     29.30 G bits/s |
| `stringzilla::hash`           |      10.93 G bits/s |     16.59 G bits/s |     20.17 G bits/s |     21.77 G bits/s |
| `stringzilla::hash_multiseed` |  __11.36 G bits/s__ | __22.18 G bits/s__ | __41.72 G bits/s__ | __71.85 G bits/s__ |
|                               |                     |                    |                    |                    |
| Python                        |                     |                    |                    |                    |
| `xxhash.xxh3_128`             |     219.56 M bits/s |    251.51 M bits/s |    281.48 M bits/s |    307.62 M bits/s |
| `stringzilla.hash`            |     300.58 M bits/s |    380.23 M bits/s |    458.91 M bits/s |    506.59 M bits/s |
| `stringzilla.hash_multiseed`  | __860.00 M bits/s__ |  __1.67 G bits/s__ |  __3.37 G bits/s__ |  __6.48 G bits/s__ |

> Measured June 17, 2026, single-threaded, hashing short words from `xlsum.csv`.

The multi-seed path prepares the input once and replays cheap per-seed rounds, so its throughput climbs almost linearly with the digest width while the naive variants plateau — StringZilla's own `hash` flattens near 22 G bits/s because it re-prepares the key every 64 bits.
`xxh3_128` keeps its full 128-bit output, so it re-prepares only every 128 bits and overtakes `stringzilla::hash` once the digest reaches 512 bits, but it never catches `hash_multiseed`.
In Python the picture inverts for the baselines: per-call interpreter dispatch dominates, so `stringzilla.hash` (one native call per 64 bits) stays ahead of `xxhash.xxh3_128` (whose 128-bit output costs an extra big-integer split), while `hash_multiseed` — a single native call that fills the whole buffer — runs an order of magnitude ahead of both.

## Probabilistic Membership

Building each filter from the unique words, then querying a held-out 20% to measure the false-positive rate.
Each filter is compared StringZilla-fed against its practical default with the structure held fixed.

### Intel Xeon4 Sapphire Rapids

| Variant                          |             Build |              Query |      bits/key |    FPR |
| -------------------------------- | ----------------: | -----------------: | ------------: | -----: |
| Rust                             |                   |                    |               |        |
| `fastbloom<siphash>`             |    14.30 M keys/s |     12.22 M keys/s | 9.59 bits/key | 1.063% |
| `fastbloom<stringzilla>`         |    26.42 M keys/s | __23.65 M keys/s__ | 9.59 bits/key | 1.034% |
| `xorf::BinaryFuse8<xxh3>`        |    13.52 M keys/s |     31.66 M keys/s | 9.15 bits/key | 0.353% |
| `xorf::BinaryFuse8<stringzilla>` |    14.04 M keys/s | __39.51 M keys/s__ | 9.15 bits/key | 0.395% |
|                                  |                   |                    |               |        |
| Python                           |                   |                    |               |        |
| `pyprobables<fnv>`               |     0.08 M keys/s |      0.09 M keys/s | 9.59 bits/key | 1.032% |
| `pyprobables<stringzilla>`       | __0.40 M keys/s__ |  __0.40 M keys/s__ | 9.59 bits/key | 0.978% |

> Measured June 17, 2026, single-threaded, over `xlsum.csv` words at a 1% target false-positive rate.

Feeding StringZilla helps exactly where the filter accepts a precomputed hash.
In Rust, `fastbloom`'s `insert_hash` / `contains_hash` take a single `sz::hash` and expand it internally, roughly doubling build and query throughput at identical bits-per-key and FPR, and `xorf` — built from a deduplicated `u64` array — queries faster with StringZilla keys.
In Python, `pyprobables` opens the same door: `add_alt` / `check_alt` take precomputed hashes, so one native `hash_multiseed` call replaces its default per-key FNV-1a loop and runs filter build and query ~5× faster at matching bits-per-key and FPR.
The lesson mirrors Layer 1 — StringZilla's hashing wins whenever a structure lets it hash each key once and hand over the result, in either language; it loses only when the API dribbles the key through a per-element hash callback (as `rbloom`'s `hash_func` does, with no precomputed-hash entry point).

---

See [README.md](../README.md) for dataset information and replication instructions.
