# Random Generation and Lookup Table Benchmarks

Benchmarks for random byte generation and lookup table operations across Rust and Python implementations.

## Overview

Some of the most common operations in data processing are random generation and lookup tables.
That's true not only for strings but for any data type, and StringZilla has been extensively used in Image Processing and Bioinformatics for those purposes.

## Random Byte Generation

### Intel Xeon4 Sapphire Rapids

| Library                        |   Short Words |     Long Lines |
| ------------------------------ | ------------: | -------------: |
| Rust                           |               |                |
| `getrandom::fill`              |     0.03 GB/s |      0.46 GB/s |
| `rand_chacha::ChaCha20Rng`     |     0.06 GB/s |      2.00 GB/s |
| `rand_xoshiro::Xoshiro128Plus` |     0.40 GB/s |      4.03 GB/s |
| `zeroize::zeroize`             |     0.46 GB/s |      4.98 GB/s |
| `stringzilla::fill_random`     | __1.01 GB/s__ |  __8.58 GB/s__ |
|                                |               |                |
| Python                         |               |                |
| `numpy.PCG64`                  |     0.01 GB/s |      1.87 GB/s |
| `numpy.Philox`                 |     0.01 GB/s |      1.45 GB/s |
| `pycryptodome.AES-CTR`         |     0.01 GB/s |      0.37 GB/s |
| `stringzilla.random`           | __0.11 GB/s__ | __18.46 GB/s__ |

> Measured June 17, 2026.

## Lookup Tables

Performing in-place lookups in a precomputed table of 256 bytes:

### Intel Xeon4 Sapphire Rapids

| Library                          |   Short Words |     Long Lines |
| -------------------------------- | ------------: | -------------: |
| Rust                             |               |                |
| serial code                      | __0.47 GB/s__ |      4.06 GB/s |
| `stringzilla::lookup_inplace`    |     0.42 GB/s | __10.22 GB/s__ |
|                                  |               |                |
| Python                           |               |                |
| `bytes.translate<new>`           | __0.12 GB/s__ |      2.68 GB/s |
| `numpy.take<new>`                |     0.01 GB/s |      0.86 GB/s |
| `opencv.LUT<new>`                |     0.01 GB/s |      2.00 GB/s |
| `opencv.LUT<inplace>`            |     0.01 GB/s |      2.16 GB/s |
| `stringzilla.translate<new>`     |     0.09 GB/s |      7.94 GB/s |
| `stringzilla.translate<inplace>` |     0.07 GB/s |  __8.02 GB/s__ |

> Measured June 17, 2026.

---

See [README.md](README.md) for dataset information and replication instructions.
