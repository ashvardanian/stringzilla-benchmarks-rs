# Substring and Byte-Set Search Benchmarks

Benchmarks for substring search and character-set matching across Rust and Python implementations.

## Substring Search

Substring search is one of the most common operations in text processing, and one of the slowest.
Most of the time, programmers don't think about replacing the `str::find` method, as it's already expected to be optimized.
In many languages it's offloaded to the C standard library [`memmem`](https://man7.org/linux/man-pages/man3/memmem.3.html) or [`strstr`](https://en.cppreference.com/w/c/string/byte/strstr) for `NULL`-terminated strings.
The C standard library is, however, also implemented by humans, and a better solution can be created.

### Forward Search

### Intel Xeon4 Sapphire Rapids

| Library                | Short Word Queries | Long Line Queries |
| ---------------------- | -----------------: | ----------------: |
| Rust                   |                    |                   |
| `std::str::find`       |          9.34 GB/s |        11.45 GB/s |
| `memmem::find`         |          9.52 GB/s |        11.29 GB/s |
| `memmem::Finder`       |          9.99 GB/s |        11.33 GB/s |
| `stringzilla::find`    |     __11.41 GB/s__ |    __11.52 GB/s__ |
|                        |                    |                   |
| Python                 |                    |                   |
| `str.find`             |          0.73 GB/s |         1.14 GB/s |
| `pyahocorasick.iter`   |                  — |                 — |
| `stringzilla.Str.find` |      __3.37 GB/s__ |    __11.64 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                | Short Word Queries | Long Line Queries |
| ---------------------- | -----------------: | ----------------: |
| Rust                   |                    |                   |
| `std::str::find`       |         35.70 GB/s |        50.65 GB/s |
| `memmem::find`         |         35.68 GB/s |        50.56 GB/s |
| `memmem::Finder`       |     __37.85 GB/s__ |    __50.71 GB/s__ |
| `stringzilla::find`    |         30.04 GB/s |        33.19 GB/s |
|                        |                    |                   |
| Python                 |                    |                   |
| `str.find`             |          3.04 GB/s |        18.03 GB/s |
| `pyahocorasick.iter`   |          1.32 GB/s |         1.24 GB/s |
| `stringzilla.Str.find` |     __26.80 GB/s__ |    __38.08 GB/s__ |

> Measured July 29, 2026.

### Reverse Search

Interestingly, the reverse order search is almost never implemented in SIMD, assuming fewer people ever need it.
Still, those are provided by StringZilla mostly for parsing tasks and feature parity.

### Intel Xeon4 Sapphire Rapids

| Library                 | Short Word Queries | Long Line Queries |
| ----------------------- | -----------------: | ----------------: |
| Rust                    |                    |                   |
| `std::str::rfind`       |          2.94 GB/s |         5.21 GB/s |
| `memmem::rfind`         |          2.93 GB/s |         5.02 GB/s |
| `memmem::FinderRev`     |          2.96 GB/s |         5.02 GB/s |
| `stringzilla::rfind`    |     __10.79 GB/s__ |    __11.45 GB/s__ |
|                         |                    |                   |
| Python                  |                    |                   |
| `str.rfind`             |          1.39 GB/s |         3.80 GB/s |
| `stringzilla.Str.rfind` |      __7.76 GB/s__ |    __11.63 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                 | Short Word Queries | Long Line Queries |
| ----------------------- | -----------------: | ----------------: |
| Rust                    |                    |                   |
| `std::str::rfind`       |          6.08 GB/s |         4.46 GB/s |
| `memmem::rfind`         |          6.07 GB/s |         4.47 GB/s |
| `memmem::FinderRev`     |          6.12 GB/s |         4.44 GB/s |
| `stringzilla::rfind`    |     __28.92 GB/s__ |    __32.85 GB/s__ |
|                         |                    |                   |
| Python                  |                    |                   |
| `str.rfind`             |          2.83 GB/s |         4.13 GB/s |
| `stringzilla.Str.rfind` |     __22.56 GB/s__ |    __32.71 GB/s__ |

> Measured July 29, 2026.

## Byte-Set Search

StringWars takes a few representative examples of various character sets that appear in real parsing or string validation tasks:

- tabulation characters, like `\n\r\v\f`;
- HTML and XML markup characters, like `</>&'\"=[]`;
- numeric characters, like `0123456789`.

It's common in such cases, to pre-construct some library-specific filter-object or Finite State Machine (FSM) to search for a set of characters.
Once that object is constructed, all of its inclusions in each token (word or line) are counted.

### Intel Xeon4 Sapphire Rapids

| Library                         |   Short Words |    Long Lines |
| ------------------------------- | ------------: | ------------: |
| Rust                            |               |               |
| `bstr::find_byteset`            |             — |             — |
| `regex::find_iter`              |     0.20 GB/s |     5.07 GB/s |
| `aho_corasick::find_iter`       |     0.34 GB/s |     0.51 GB/s |
| `stringzilla::find_byteset`     | __1.20 GB/s__ | __8.34 GB/s__ |
|                                 |               |               |
| Python                          |               |               |
| `re.finditer`                   |     0.05 GB/s |     0.21 GB/s |
| `stringzilla.Str.find_first_of` | __0.12 GB/s__ | __9.35 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                         |     Short Words |      Long Lines |
| ------------------------------- | --------------: | --------------: |
| Rust                            |                 |                 |
| `bstr::find_byteset`            |       1.47 GB/s |       3.67 GB/s |
| `regex::find_iter`              |     409.15 MB/s |       9.41 GB/s |
| `aho_corasick::find_iter`       |     728.46 MB/s |     985.73 MB/s |
| `stringzilla::find_byteset`     |   __1.54 GB/s__ |  __13.10 GB/s__ |
|                                 |                 |                 |
| Python                          |                 |                 |
| `re.finditer`                   |     751.84 MB/s |     617.54 MB/s |
| `stringzilla.Str.find_first_of` |   __4.01 GB/s__ |   __4.37 GB/s__ |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
