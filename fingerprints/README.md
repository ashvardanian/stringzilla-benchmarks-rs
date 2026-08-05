# Fingerprinting and Sketching Benchmarks

Benchmarks for byte-level fingerprinting and sketching algorithms across CPU and GPU implementations.

## Overview

In large-scale Retrieval workloads a common technique is to convert variable-length messy strings into some fixed-length representations.
Those are often called "fingerprints" or "sketches", like "Min-Hashing" or "Count-Min-Sketching".
There are a million variations of those algorithms, all resulting in different speed-vs-accuracy tradeoffs.

Two of the approximations worth considering are:

- The number of collisions of produced individual hashes within fingerprints
- The bit-distribution entropy of the produced fingerprints

Adjusting all implementations to the same tokenization scheme, one may experience the following numbers:

## Performance and Quality Metrics

Fingerprint throughput is measured at __512 dimensions__.

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                              |  ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------------ | ----------------: | -----------------: |
| `serial::MinHash<ByteGrams><1xSPR>`  |         0.23 MB/s |          0.20 MB/s |
|                                      | 54.72% collisions |  30.03% collisions |
|                                      |    0.8530 entropy |     0.7916 entropy |
|                                      |                   |                    |
| `pc::MinHash<ByteGrams><1xSPR>`      |         1.58 MB/s |          2.04 MB/s |
|                                      | 63.68% collisions |  46.80% collisions |
|                                      |    0.9343 entropy |     0.8704 entropy |
|                                      |                   |                    |
| `stringzillas::Fingerprints<1xSPR>`  |         0.31 MB/s |          0.26 MB/s |
| `stringzillas::Fingerprints<16xSPR>` |         4.02 MB/s |          4.07 MB/s |
| `stringzillas::Fingerprints<H100>`   |    __98.54 MB/s__ |    __706.64 MB/s__ |
|                                      | 64.64% collisions |  48.30% collisions |
|                                      |    0.9980 entropy |     0.9977 entropy |

> Measured June 17, 2026.

## Quality Analysis

The trickiest part, however, is analyzing the retrieval quality of those fingerprints and comparing them to other approaches.
So, how many bits per fingerprint are needed to achieve a specific recall rate for a given dataset?
Or, how does the average Levenshtein distance among the top-k nearest neighbors change with the fingerprint size?
It must clearly decrease, but how fast, and how does that compare to ground truth?

For detailed quality analysis, please check out the [HashEvals](https://github.com/ashvardanian/HashEvals) repository.

---

See [README.md](README.md) for dataset information and replication instructions.
