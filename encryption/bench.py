# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cryptography",
#   "pycryptodome",
#   "pynacl",
# ]
# ///
"""
AEAD encryption/decryption benchmarks in Python, mirroring encryption/bench.rs.

Both files compare the two AEAD ciphers that dominate TLS 1.3 and the Noise framework — AES-256-GCM
(hardware-accelerated) and ChaCha20-Poly1305 (software-optimized) — across the common Python crypto
libraries: `cryptography` (OpenSSL backend), `pycryptodome`, and `pynacl` (libsodium).

Throughput is reported in bytes/s over the plaintext, encrypting/decrypting one token per call.

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run encryption/bench.py --dataset data/acgt/acgt_100.txt --tokens lines
  uv run encryption/bench.py --dataset data/acgt/acgt_1k.txt --tokens lines -k "chacha"
"""

import argparse
import re
import sys
from collections.abc import Callable

import nacl.bindings as libsodium
from Crypto.Cipher import AES, ChaCha20_Poly1305
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    paced_items,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
)

KEY = bytes(32)  # 256-bit key (all zeros — content is irrelevant to throughput)


def nonce_for(counter: int) -> bytes:
    """A 96-bit IETF nonce derived from a per-message counter, matching the Rust harness."""
    return counter.to_bytes(12, "little")


def log_system_info():
    from importlib.metadata import version as pkg_version

    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- cryptography: {pkg_version('cryptography')}")
    print(f"- pycryptodome: {pkg_version('pycryptodome')}")
    print(f"- pynacl: {pkg_version('pynacl')}")
    print()


def bench_encrypt(name: str, tokens: list[bytes], encrypt: Callable[[bytes, bytes], object], time_limit: float):
    """Encrypt one token per call under the time budget; report bytes/s over the plaintext."""
    start = now_nanoseconds()
    deadline = start + int(time_limit * 1e9)
    count = 0
    total_bytes = 0
    for token in paced_items(tokens, deadline):
        encrypt(token, nonce_for(count))
        count += 1
        total_bytes += len(token)
    seconds = (now_nanoseconds() - start) / 1e9
    report_stats(name, "bytes", seconds, count, total_bytes)


def bench_decrypt(
    name: str,
    blobs: list[object],
    plaintext_lengths: list[int],
    decrypt: Callable[[object, bytes], object],
    time_limit: float,
):
    """Decrypt one previously-encrypted token per call; report bytes/s over the original plaintext."""
    start = now_nanoseconds()
    deadline = start + int(time_limit * 1e9)
    count = 0
    total_bytes = 0
    index = 0
    end = start
    while True:
        decrypt(blobs[index], nonce_for(index))
        total_bytes += plaintext_lengths[index]
        count += 1
        index = (index + 1) % len(blobs)
        end = now_nanoseconds()
        if end >= deadline:
            break
    report_stats(name, "bytes", (end - start) / 1e9, count, total_bytes)


# Each cipher is a (label, encrypt, decrypt) triple. `encrypt(data, nonce)` returns an opaque blob;
# `decrypt(blob, nonce)` consumes it. Nonces are supplied by the harness as a per-message counter.
def cryptography_aesgcm():
    cipher = AESGCM(KEY)
    return ("cryptography.AESGCM", lambda d, n: cipher.encrypt(n, d, None), lambda b, n: cipher.decrypt(n, b, None))


def cryptography_chacha():
    cipher = ChaCha20Poly1305(KEY)
    return (
        "cryptography.ChaCha20Poly1305",
        lambda d, n: cipher.encrypt(n, d, None),
        lambda b, n: cipher.decrypt(n, b, None),
    )


def pycryptodome_aesgcm():
    def encrypt(data, nonce):
        return AES.new(KEY, AES.MODE_GCM, nonce=nonce).encrypt_and_digest(data)

    def decrypt(blob, nonce):
        ciphertext, tag = blob
        return AES.new(KEY, AES.MODE_GCM, nonce=nonce).decrypt_and_verify(ciphertext, tag)

    return ("pycryptodome.AES-GCM", encrypt, decrypt)


def pycryptodome_chacha():
    def encrypt(data, nonce):
        return ChaCha20_Poly1305.new(key=KEY, nonce=nonce).encrypt_and_digest(data)

    def decrypt(blob, nonce):
        ciphertext, tag = blob
        return ChaCha20_Poly1305.new(key=KEY, nonce=nonce).decrypt_and_verify(ciphertext, tag)

    return ("pycryptodome.ChaCha20Poly1305", encrypt, decrypt)


def pynacl_chacha():
    return (
        "pynacl.chacha20poly1305_ietf",
        lambda d, n: libsodium.crypto_aead_chacha20poly1305_ietf_encrypt(d, None, n, KEY),
        lambda b, n: libsodium.crypto_aead_chacha20poly1305_ietf_decrypt(b, None, n, KEY),
    )


CIPHERS = [
    cryptography_aesgcm,
    cryptography_chacha,
    pycryptodome_aesgcm,
    pycryptodome_chacha,
    pynacl_chacha,
]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AEAD encryption/decryption across Python crypto libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as error:
            parser.error(f"Invalid regex for --filter: {error}")

    dataset = load_dataset(args.dataset, as_bytes=True, size_limit=args.dataset_limit)
    tokens = tokenize_dataset(dataset, resolve_tokens(args.tokens, "lines"))
    if not tokens:
        print("No tokens found in dataset")
        return 1

    total_bytes = sum(len(token) for token in tokens)
    print(f"Dataset: {len(tokens):,} tokens, {total_bytes:,} bytes, {total_bytes / len(tokens):.1f} avg token length")
    log_system_info()

    plaintext_lengths = [len(token) for token in tokens]

    print("\n# encryption")
    for build in CIPHERS:
        label, encrypt, _ = build()
        if should_run(f"encryption/{label}", filter_pattern):
            bench_encrypt(label, tokens, encrypt, args.time_limit)

    print("\n# decryption")
    for build in CIPHERS:
        label, encrypt, decrypt = build()
        if should_run(f"decryption/{label}", filter_pattern):
            blobs = [encrypt(token, nonce_for(index)) for index, token in enumerate(tokens)]
            bench_decrypt(label, blobs, plaintext_lengths, decrypt, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
