# Memory Benchmarks

Benchmarks for random byte generation, lookup-table transforms, and the fill, copy and move
primitives, across Rust and Python implementations.

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
| `stringzilla::fill_random`     | __1.01 GB/s__ |  __8.58 GB/s__ |
|                                |               |                |
| Python                         |               |                |
| `numpy.PCG64`                  |     0.01 GB/s |      1.87 GB/s |
| `numpy.Philox`                 |     0.01 GB/s |      1.45 GB/s |
| `pycryptodome.AES-CTR`         |     0.01 GB/s |      0.37 GB/s |
| `stringzilla.fill_random`      |             — |              — |
| `stringzilla.random`           | __0.11 GB/s__ | __18.46 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                        |   Short Words |     Long Lines |
| ------------------------------ | ------------: | -------------: |
| Rust                           |               |                |
| `getrandom::fill`              |             — |      0.24 GB/s |
| `rand_chacha::ChaCha20Rng`     |     0.41 GB/s |      0.89 GB/s |
| `rand_xoshiro::Xoshiro128Plus` |     1.11 GB/s |      6.10 GB/s |
| `stringzilla::fill_random`     | __1.23 GB/s__ | __37.09 GB/s__ |
|                                |               |                |
| Python                         |               |                |
| `numpy.PCG64`                  |     0.03 GB/s |      2.34 GB/s |
| `numpy.Philox`                 |     0.03 GB/s |      3.07 GB/s |
| `pycryptodome.AES-CTR`         |     0.02 GB/s |      0.46 GB/s |
| `stringzilla.fill_random`      |     0.14 GB/s |     25.87 GB/s |
| `stringzilla.random`           | __0.45 GB/s__ | __46.93 GB/s__ |

> Measured July 29, 2026.

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
| `numpy.indexing<new>`            |             — |              — |
| `numpy.indexing<inplace>`        |             — |              — |
| `numpy.take<new>`                |     0.01 GB/s |      0.86 GB/s |
| `numpy.take<inplace>`            |             — |              — |
| `opencv.LUT<new>`                |     0.01 GB/s |      2.00 GB/s |
| `opencv.LUT<inplace>`            |     0.01 GB/s |      2.16 GB/s |
| `stringzilla.translate<new>`     |     0.09 GB/s |      7.94 GB/s |
| `stringzilla.translate<inplace>` |     0.07 GB/s |  __8.02 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                          |   Short Words |     Long Lines |
| -------------------------------- | ------------: | -------------: |
| Rust                             |               |                |
| serial code                      | __1.16 GB/s__ |      5.35 GB/s |
| `stringzilla::lookup_inplace`    |     0.85 GB/s | __14.76 GB/s__ |
|                                  |               |                |
| Python                           |               |                |
| `bytes.translate<new>`           | __0.23 GB/s__ |      4.73 GB/s |
| `numpy.indexing<new>`            |     0.03 GB/s |      1.18 GB/s |
| `numpy.indexing<inplace>`        |     0.02 GB/s |      1.13 GB/s |
| `numpy.take<new>`                |     0.02 GB/s |      0.76 GB/s |
| `numpy.take<inplace>`            |     0.02 GB/s |      0.73 GB/s |
| `opencv.LUT<new>`                |     0.02 GB/s |      3.00 GB/s |
| `opencv.LUT<inplace>`            |     0.03 GB/s |      3.15 GB/s |
| `stringzilla.translate<new>`     |     0.19 GB/s |     11.36 GB/s |
| `stringzilla.translate<inplace>` |     0.15 GB/s | __12.08 GB/s__ |

> Measured July 29, 2026.

## Memory Fills

Overwriting every token in place with one constant byte — the `memset` pattern.
`zeroize::zeroize` belongs here rather than with the random generators: it writes zeros, and it does so through volatile stores that the compiler may not widen or elide.

### Apple M5 Pro

| Library                 |   Short Words |     Long Lines |
| ----------------------- | ------------: | -------------: |
| Rust                    |               |                |
| `stringzilla::fill`     |     1.12 GB/s |     41.54 GB/s |
| `std::ptr::write_bytes` |     1.25 GB/s | __52.57 GB/s__ |
| `slice::fill`           |     1.26 GB/s |     52.50 GB/s |
| `zeroize::zeroize`      | __1.27 GB/s__ |      4.26 GB/s |

> Measured July 29, 2026.

## Memory Copies

Copying every token into a matching slice of a second, non-overlapping arena — the `memcpy` pattern.

### Apple M5 Pro

| Library                         |   Short Words |     Long Lines |
| ------------------------------- | ------------: | -------------: |
| Rust                            |               |                |
| `stringzilla::copy`             |     1.15 GB/s |     37.79 GB/s |
| `slice::copy_from_slice`        | __1.25 GB/s__ | __42.26 GB/s__ |
| `std::ptr::copy_nonoverlapping` |     1.07 GB/s |     42.14 GB/s |

> Measured July 29, 2026.

## Memory Moves

Shifting every token 8 bytes forward inside its own buffer, so source and destination overlap — the `memmove` pattern.

### Apple M5 Pro

| Library              |   Short Words |     Long Lines |
| -------------------- | ------------: | -------------: |
| Rust                 |               |                |
| `stringzilla::move_` |     1.10 GB/s |     36.24 GB/s |
| `std::ptr::copy`     |     1.13 GB/s | __42.95 GB/s__ |
| `slice::copy_within` | __1.15 GB/s__ |     42.89 GB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
