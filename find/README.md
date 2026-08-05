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
| `stringzilla.Str.find` |      __3.37 GB/s__ |    __11.64 GB/s__ |

> Measured June 17, 2026.

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
| `bstr::iter`                    |     0.26 GB/s |     0.36 GB/s |
| `regex::find_iter`              |     0.20 GB/s |     5.07 GB/s |
| `aho_corasick::find_iter`       |     0.34 GB/s |     0.51 GB/s |
| `stringzilla::find_byteset`     | __1.20 GB/s__ | __8.34 GB/s__ |
|                                 |               |               |
| Python                          |               |               |
| `re.finditer`                   |     0.05 GB/s |     0.21 GB/s |
| `stringzilla.Str.find_first_of` | __0.12 GB/s__ | __9.35 GB/s__ |

> Measured June 17, 2026.

---

See [README.md](README.md) for dataset information and replication instructions.
