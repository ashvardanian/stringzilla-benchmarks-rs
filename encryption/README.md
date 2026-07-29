# Encryption and Decryption Benchmarks

Benchmarks for encryption and decryption operations across Rust and Python implementations.

## Overview

These benchmarks compare ChaCha20-Poly1305 and AES-256-GCM AEAD throughput across different libraries.
Rust covers `ring`, `openssl`, and `libsodium`; Python covers `cryptography` (OpenSSL backend), `pynacl` (libsodium), and `pycryptodome`.
The `cryptography` and `ring`/`openssl` paths reach the same OpenSSL/AES-NI kernels, while `pycryptodome` pays a large per-message Python-object overhead that dominates on short lines.

## Encryption

### Intel Xeon4 Sapphire Rapids

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |        0.16 GB/s |          0.53 GB/s |
| `libsodium::xchacha20`          |                — |                  — |
| `ring::chacha20`                |        0.27 GB/s |          0.80 GB/s |
| `ring::aes256`                  |    __0.39 GB/s__ |      __2.20 GB/s__ |
| `openssl::chacha20`             |                — |                  — |
| `openssl::aes256`               |                — |                  — |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |   __73.93 MB/s__ |    __694.70 MB/s__ |
| `cryptography.ChaCha20Poly1305` |       33.31 MB/s |        361.65 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       17.56 MB/s |        159.58 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        3.87 MB/s |         36.46 MB/s |
| `pycryptodome.AES-GCM`          |        1.39 MB/s |         19.60 MB/s |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |      385.78 MB/s |        726.36 MB/s |
| `libsodium::xchacha20`          |      317.73 MB/s |        695.69 MB/s |
| `ring::chacha20`                |      598.89 MB/s |          1.68 GB/s |
| `ring::aes256`                  |    __2.03 GB/s__ |      __5.56 GB/s__ |
| `openssl::chacha20`             |      216.13 MB/s |          1.10 GB/s |
| `openssl::aes256`               |      349.85 MB/s |          2.54 GB/s |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |  __264.19 MB/s__ |      __2.02 GB/s__ |
| `cryptography.ChaCha20Poly1305` |      161.27 MB/s |        937.92 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       27.30 MB/s |         61.44 MB/s |
| `pycryptodome.ChaCha20Poly1305` |       16.75 MB/s |        128.77 MB/s |
| `pycryptodome.AES-GCM`          |        7.33 MB/s |         54.64 MB/s |

> Measured July 29, 2026.

## Decryption

### Intel Xeon4 Sapphire Rapids

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |        0.29 GB/s |          1.13 GB/s |
| `libsodium::xchacha20`          |                — |                  — |
| `ring::chacha20`                |        0.34 GB/s |          0.74 GB/s |
| `ring::aes256`                  |    __0.65 GB/s__ |      __2.11 GB/s__ |
| `openssl::chacha20`             |                — |                  — |
| `openssl::aes256`               |                — |                  — |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |   __70.98 MB/s__ |    __573.12 MB/s__ |
| `cryptography.ChaCha20Poly1305` |       35.67 MB/s |        283.99 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       19.04 MB/s |        139.83 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        2.23 MB/s |         17.63 MB/s |
| `pycryptodome.AES-GCM`          |        1.34 MB/s |         13.47 MB/s |

> Measured June 17, 2026.

### Apple M5 Pro

| Library                         | ~100 bytes lines | ~1,000 bytes lines |
| ------------------------------- | ---------------: | -----------------: |
| Rust                            |                  |                    |
| `libsodium::chacha20`           |      374.29 MB/s |        719.57 MB/s |
| `libsodium::xchacha20`          |      307.40 MB/s |        690.94 MB/s |
| `ring::chacha20`                |      599.25 MB/s |          1.34 GB/s |
| `ring::aes256`                  |    __2.07 GB/s__ |      __5.60 GB/s__ |
| `openssl::chacha20`             |      212.98 MB/s |          1.10 GB/s |
| `openssl::aes256`               |      351.82 MB/s |          2.57 GB/s |
|                                 |                  |                    |
| Python                          |                  |                    |
| `cryptography.AESGCM`           |  __270.09 MB/s__ |      __2.02 GB/s__ |
| `cryptography.ChaCha20Poly1305` |      164.68 MB/s |        881.55 MB/s |
| `pynacl.chacha20poly1305_ietf`  |       26.65 MB/s |         61.09 MB/s |
| `pycryptodome.ChaCha20Poly1305` |        9.19 MB/s |         78.77 MB/s |
| `pycryptodome.AES-GCM`          |        5.36 MB/s |         42.03 MB/s |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
