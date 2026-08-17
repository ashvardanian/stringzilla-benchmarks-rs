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
| Rust                                 |                   |                    |
| `serial::MinHash<ByteGrams><1xSPR>`  |         0.38 MB/s |          0.38 MB/s |
|                                      | 56.13% collisions |  31.81% collisions |
|                                      |    0.8530 entropy |     0.7916 entropy |
|                                      |                   |                    |
| `pc::MinHash<ByteGrams><1xSPR>`      |         2.07 MB/s |          2.64 MB/s |
|                                      | 63.90% collisions |  46.69% collisions |
|                                      |    0.9341 entropy |     0.8687 entropy |
|                                      |                   |                    |
| `stringzillas::Fingerprints<1xSPR>`  |         1.01 MB/s |          0.98 MB/s |
| `stringzillas::Fingerprints<16xSPR>` |         6.69 MB/s |         11.72 MB/s |
| `stringzillas::Fingerprints<H100>`   |   __137.04 MB/s__ |    __803.77 MB/s__ |
|                                      | 64.64% collisions |  48.41% collisions |
|                                      |    0.9980 entropy |     0.9977 entropy |
|                                      |                   |                    |
| Python                               |                   |                    |
| `datasketch.MinHash`                 |         0.03 MB/s |          0.03 MB/s |
| `stringzillas.Fingerprints<1xSPR>`   |         0.99 MB/s |          0.97 MB/s |
| `stringzillas.Fingerprints<16xSPR>`  |         4.68 MB/s |         10.88 MB/s |
| `stringzillas.Fingerprints<H100>`    |        35.44 MB/s |        259.23 MB/s |

> Measured on a 16-core Xeon Platinum 8468 and one idle H100 80GB HBM3, over `acgt_100` and `acgt_1k`.

## Quality Analysis

The trickiest part, however, is analyzing the retrieval quality of those fingerprints and comparing them to other approaches.
So, how many bits per fingerprint are needed to achieve a specific recall rate for a given dataset?
Or, how does the average Levenshtein distance among the top-k nearest neighbors change with the fingerprint size?
It must clearly decrease, but how fast, and how does that compare to ground truth?

For detailed quality analysis, please check out the [HashEvals](https://github.com/ashvardanian/HashEvals) repository.

---

See [README.md](README.md) for dataset information and replication instructions.
