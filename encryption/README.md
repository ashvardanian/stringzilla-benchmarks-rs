# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations across Rust and Python implementations.

## Overview

These benchmarks compare ChaCha20-Poly1305 and AES-256-GCM AEAD throughput across different libraries.
Rust covers `ring`, `openssl`, `libsodium`, and `stringzilla`; Python covers `cryptography` (OpenSSL backend), `pynacl` (libsodium), `pycryptodome`, and `stringzilla`.
The `cryptography` and `ring`/`openssl` paths reach the same OpenSSL/AES-NI kernels, while `pycryptodome` pays a large per-message Python-object overhead that dominates on short lines.

Each Rust row seals and opens into a buffer sized before the timed loop, so the rows price the cipher rather than the allocator.
OpenSSL is the exception: it builds one cipher context per message, which the crate gives no way to re-key, and at 100-byte lines that setup is most of the call — roughly 750 ns of 1.62 µs.

StringZilla also contributes an AES-256-CTR row.
Counter mode is unauthenticated, so it is not comparable to the AEAD rows on security — it is the floor, the same AES round function without a tag to accumulate, and the gap to `stringzilla::aes256` is what GHASH costs.

## Encryption

### Intel Xeon4 Sapphire Rapids

| Library                         | \~100 bytes lines | \~1,000 bytes lines |
| ------------------------------- | ----------------: | ------------------: |
| Rust                            |                   |                     |
| `ring::aes256`                  |         0.81 GB/s |           3.96 GB/s |
| `ring::chacha20`                |         0.46 GB/s |           1.38 GB/s |
| `libsodium::chacha20`           |         0.21 GB/s |           0.80 GB/s |
| `libsodium::xchacha20`          |         0.18 GB/s |           0.74 GB/s |
| `openssl::aes256`               |         0.11 GB/s |           0.92 GB/s |
| `openssl::chacha20`             |         0.06 GB/s |           0.53 GB/s |
| `stringzilla::aes256`           |     __0.66 GB/s__ |       __4.45 GB/s__ |
|                                 |                   |                     |
| _`stringzilla::aes256ctr`_      |       _3.17 GB/s_ |        _10.16 GB/s_ |
|                                 |                   |                     |
| Python                          |                   |                     |
| `cryptography.AESGCM`           |       159.81 MB/s |           1.17 GB/s |
| `cryptography.ChaCha20Poly1305` |        62.37 MB/s |         506.90 MB/s |
| `pynacl.chacha20poly1305_ietf`  |        32.23 MB/s |         247.86 MB/s |
| `pycryptodome.ChaCha20Poly1305` |         6.12 MB/s |          50.36 MB/s |
| `pycryptodome.AES-GCM`          |         3.08 MB/s |          28.10 MB/s |
| `stringzilla.Aes256GcmKey`      |   __346.03 MB/s__ |       __2.18 GB/s__ |
|                                 |                   |                     |
| _`stringzilla.Aes256CtrKey`_    |     _636.28 MB/s_ |         _3.50 GB/s_ |

> Measured August 5, 2026.

## Decryption

### Intel Xeon4 Sapphire Rapids

| Library                         | \~100 bytes lines | \~1,000 bytes lines |
| ------------------------------- | ----------------: | ------------------: |
| Rust                            |                   |                     |
| `ring::aes256`                  |         0.88 GB/s |           4.13 GB/s |
| `ring::chacha20`                |         0.44 GB/s |           1.44 GB/s |
| `libsodium::chacha20`           |         0.20 GB/s |           0.81 GB/s |
| `libsodium::xchacha20`          |         0.18 GB/s |           0.75 GB/s |
| `openssl::aes256`               |         0.11 GB/s |           0.92 GB/s |
| `openssl::chacha20`             |         0.06 GB/s |           0.52 GB/s |
| `stringzilla::aes256`           |     __0.68 GB/s__ |       __4.71 GB/s__ |
|                                 |                   |                     |
| _`stringzilla::aes256ctr`_      |       _3.16 GB/s_ |        _10.38 GB/s_ |
|                                 |                   |                     |
| Python                          |                   |                     |
| `cryptography.AESGCM`           |       128.96 MB/s |           1.01 GB/s |
| `cryptography.ChaCha20Poly1305` |        55.92 MB/s |         455.88 MB/s |
| `pynacl.chacha20poly1305_ietf`  |        29.46 MB/s |         184.05 MB/s |
| `pycryptodome.ChaCha20Poly1305` |         3.33 MB/s |          23.34 MB/s |
| `pycryptodome.AES-GCM`          |         2.02 MB/s |          18.59 MB/s |
| `stringzilla.Aes256GcmKey`      |   __241.28 MB/s__ |       __1.83 GB/s__ |
|                                 |                   |                     |
| `stringzilla.Aes256CtrKey`      |     _354.86 MB/s_ |         _2.49 GB/s_ |

> Measured August 5, 2026.

## Key Setup

Expanding a key is per-connection work rather than per-message, but at 100-byte lines it is a large share of the call.
StringZilla's GCM key derives eight Galois hash powers beyond the round-key schedule its CTR key stops at, which is the whole distance between those two rows.
`libsodium` only copies 32 bytes here, deferring expansion to the first message, which is why it sits an order of magnitude below the rest.

| Library                  | Median latency |
| ------------------------ | -------------: |
| `libsodium::chacha20`    |        7.89 ns |
| `ring::chacha20`         |       39.55 ns |
| `stringzilla::aes256ctr` |       61.62 ns |
| `stringzilla::aes256`    |       96.09 ns |
| `ring::aes256`           |      142.22 ns |
| `openssl::chacha20`      |      718.46 ns |
| `openssl::aes256`        |      780.69 ns |

> Measured August 5, 2026.

---

See [README.md](README.md) for dataset information and replication instructions.
