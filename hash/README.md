# Hash Benchmarks

Benchmarks for hashing functions across Rust and Python implementations.

Many great hashing libraries exist in Rust, C, and C++.
Typical top choices are `aHash`, `xxHash`, `blake3`, `CityHash`, `MurmurHash`, `crc32fast`, or the native `std::hash`.
Many of them have similar pitfalls:

- They are not always documented to have a certain reproducible output and are recommended for use only for local in-memory construction of hash tables, not for serialization or network communication.
- They don't always support streaming and require the whole input to be available in memory at once.
- They don't always pass the SMHasher test suite, especially with `--extra` checks enabled.
- They generally don't have a dynamic dispatch mechanism to simplify shipping of precompiled software.
- They are rarely available for multiple programming languages.

StringZilla addresses those issues and seems to provide competitive performance.

## Single Hash

On Intel Sapphire Rapids CPU, on `xlsum.csv` dataset, the following numbers can be expected for hashing individual whitespace-delimited words and newline-delimited lines.
The __Ports__ column marks availability in other languages, like C, C++, Python, Java, Go, JavaScript.
The __Arm__ column marks Arm support — most hash functions run on both x86 and Arm, but gxHash and many MurmurHash and CityHash implementations don't.

### Intel Xeon4 Sapphire Rapids

| Library                     | Bits  | Ports |  Arm  |   Short Words |     Long Lines |
| --------------------------- | :---: | :---: | :---: | ------------: | -------------: |
| Rust                        |       |       |       |               |                |
| `std::hash`                 |  64   |   -   |   +   |     0.32 GB/s |      3.98 GB/s |
| `crc32fast::hash`           |  32   |   +   |   +   |     0.39 GB/s |      9.49 GB/s |
| `xxh3::xxh3_64`             |  64   |   +   |   +   |     0.66 GB/s |     10.00 GB/s |
| `aHash::hash_one`           |  64   |   -   |   +   |     0.74 GB/s |      9.05 GB/s |
| `foldhash::hash_one`        |  64   |   -   |   +   |     0.72 GB/s |      8.71 GB/s |
| `wyhash::wyhash`            |  64   |   +   |   +   |             — |              — |
| `murmurhash32::murmurhash3` |  32   |   +   |   +   |             — |              — |
| `stringzilla::hash`         |  64   |   +   |   +   | __0.95 GB/s__ | __12.22 GB/s__ |
|                             |       |       |       |               |                |
| Python                      |       |       |       |               |                |
| `xxhash.xxh3_64`            |  64   |   +   |   +   |     0.03 GB/s |      6.16 GB/s |
| `google_crc32c.value`       |  32   |   +   |   +   |     0.06 GB/s |      7.10 GB/s |
| `mmh3.hash32`               |  32   |   +   |   +   |     0.06 GB/s |      2.76 GB/s |
| `mmh3.hash64`               |  64   |   +   |   +   |     0.05 GB/s |      4.82 GB/s |
| `mmh3.hash128`              |  128  |   +   |   +   |             — |              — |
| `cityhash.CityHash64`       |  64   |   +   |   -   |     0.07 GB/s |      5.71 GB/s |
| `cityhash.CityHash128`      |  128  |   +   |   -   |             — |              — |
| `stringzilla.hash`          |  64   |   +   |   +   | __0.07 GB/s__ |  __7.99 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                     | Bits  | Ports |  Arm  |   Short Words |     Long Lines |
| --------------------------- | :---: | :---: | :---: | ------------: | -------------: |
| Rust                        |       |       |       |               |                |
| `std::hash`                 |  64   |   -   |   +   |     0.83 GB/s |      5.76 GB/s |
| `crc32fast::hash`           |  32   |   +   |   +   |     0.73 GB/s |     11.56 GB/s |
| `xxh3::xxh3_64`             |  64   |   +   |   +   | __2.12 GB/s__ |     47.00 GB/s |
| `aHash::hash_one`           |  64   |   -   |   +   |     2.02 GB/s |     21.68 GB/s |
| `foldhash::hash_one`        |  64   |   -   |   +   |     2.01 GB/s | __59.12 GB/s__ |
| `wyhash::wyhash`            |  64   |   +   |   +   |     1.01 GB/s |     27.02 GB/s |
| `murmurhash32::murmurhash3` |  32   |   +   |   +   |     0.99 GB/s |      3.47 GB/s |
| `stringzilla::hash`         |  64   |   +   |   +   |     1.09 GB/s |     33.75 GB/s |
|                             |       |       |       |               |                |
| Python                      |       |       |       |               |                |
| `xxhash.xxh3_64`            |  64   |   +   |   +   |     0.50 GB/s | __35.42 GB/s__ |
| `google_crc32c.value`       |  32   |   +   |   +   |     0.26 GB/s |      5.96 GB/s |
| `mmh3.hash32`               |  32   |   +   |   +   |     0.20 GB/s |      3.27 GB/s |
| `mmh3.hash64`               |  64   |   +   |   +   |     0.14 GB/s |      7.47 GB/s |
| `mmh3.hash128`              |  128  |   +   |   +   |     0.17 GB/s |      7.75 GB/s |
| `cityhash.CityHash64`       |  64   |   +   |   -   | __0.55 GB/s__ |     18.39 GB/s |
| `cityhash.CityHash128`      |  128  |   +   |   -   |     0.17 GB/s |     18.19 GB/s |
| `stringzilla.hash`          |  64   |   +   |   +   |     0.36 GB/s |     32.78 GB/s |

> Measured July 29, 2026.

## Streaming Hash

In larger systems, we often need the ability to incrementally hash the data.
This is especially important in distributed systems, where the data is too large to fit into memory at once.

### Intel Xeon4 Sapphire Rapids

| Library                    | Bits  | Ports |   Short Words |    Long Lines |
| -------------------------- | :---: | :---: | ------------: | ------------: |
| Rust                       |       |       |               |               |
| `std::hash::DefaultHasher` |  64   |   -   |     0.49 GB/s |     4.05 GB/s |
| `aHash::AHasher`           |  64   |   -   | __1.29 GB/s__ |     8.59 GB/s |
| `foldhash::FoldHasher`     |  64   |   -   |     1.10 GB/s |     8.85 GB/s |
| `crc32fast::Hasher`        |  32   |   +   |     0.39 GB/s |     9.47 GB/s |
| `stringzilla::Hasher`      |  64   |   +   |     0.44 GB/s | __9.84 GB/s__ |
|                            |       |       |               |               |
| Python                     |       |       |               |               |
| `xxhash.xxh3_64`           |  64   |   +   |     0.06 GB/s |     6.93 GB/s |
| `google_crc32c.Checksum`   |  32   |   +   |     0.05 GB/s |     7.19 GB/s |
| `stringzilla.Hasher`       |  64   |   +   | __0.08 GB/s__ | __8.29 GB/s__ |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                    | Bits  | Ports |   Short Words |     Long Lines |
| -------------------------- | :---: | :---: | ------------: | -------------: |
| Rust                       |       |       |               |                |
| `std::hash::DefaultHasher` |  64   |   -   |     1.01 GB/s |      5.86 GB/s |
| `aHash::AHasher`           |  64   |   -   | __2.44 GB/s__ |     21.59 GB/s |
| `foldhash::FoldHasher`     |  64   |   -   |     2.22 GB/s | __58.74 GB/s__ |
| `crc32fast::Hasher`        |  32   |   +   |     0.79 GB/s |     11.33 GB/s |
| `stringzilla::Hasher`      |  64   |   +   |     0.72 GB/s |     18.28 GB/s |
|                            |       |       |               |                |
| Python                     |       |       |               |                |
| `xxhash.xxh3_64`           |  64   |   +   |     0.21 GB/s | __15.93 GB/s__ |
| `google_crc32c.Checksum`   |  32   |   +   |     0.15 GB/s |      5.75 GB/s |
| `stringzilla.Hasher`       |  64   |   +   | __0.46 GB/s__ |     14.14 GB/s |

> Measured July 29, 2026.

## Checksum and Cryptographic Hashing

For reference, one may want to put those numbers next to check-sum calculation speeds on one end of complexity and cryptographic hashing speeds on the other end.

### Intel Xeon4 Sapphire Rapids

| Library                | Bits  | Ports |   Short Words |     Long Lines |
| ---------------------- | :---: | :---: | ------------: | -------------: |
| Rust                   |       |       |               |                |
| `stringzilla::bytesum` |  64   |   +   | __0.98 GB/s__ | __12.62 GB/s__ |
| `blake3::hash`         |  256  |   +   |     0.11 GB/s |      1.77 GB/s |
| `sha2::Sha256`         |  256  |   +   |             — |              — |
| `ring::SHA256`         |  256  |   +   |             — |              — |
| `stringzilla::Sha256`  |  256  |   +   |             — |              — |
|                        |       |       |               |                |
| Python                 |       |       |               |                |
| `stringzilla.bytesum`  |  64   |   +   | __0.08 GB/s__ |  __8.37 GB/s__ |
| `blake3.blake3`        |  256  |   +   |     0.02 GB/s |      1.68 GB/s |
| `hashlib.sha256`       |  256  |   +   |             — |              — |
| `stringzilla.Sha256`   |  256  |   +   |             — |              — |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                  | Bits  | Ports |   Short Words |     Long Lines |
| ------------------------ | :---: | :---: | ------------: | -------------: |
| Rust                     |       |       |               |                |
| `stringzilla::bytesum`   |  64   |   +   | __1.07 GB/s__ | __27.90 GB/s__ |
| `blake3::hash`           |  256  |   +   |     0.16 GB/s |      1.78 GB/s |
| `sha2::Sha256`           |  256  |   +   |     0.38 GB/s |      3.30 GB/s |
| `ring::SHA256`           |  256  |   +   |     0.23 GB/s |      3.29 GB/s |
| `stringzilla::Sha256`    |  256  |   +   |     0.29 GB/s |      3.30 GB/s |
|                          |       |       |               |                |
| Python                   |       |       |               |                |
| `stringzilla.bytesum`    |  64   |   +   | __0.43 GB/s__ | __20.73 GB/s__ |
| `blake3.blake3`          |  256  |   +   |     0.03 GB/s |      1.61 GB/s |
| `hashlib.sha256`         |  256  |   +   |     0.05 GB/s |      2.90 GB/s |
| `stringzilla.Sha256`     |  256  |   +   |     0.12 GB/s |      3.16 GB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
