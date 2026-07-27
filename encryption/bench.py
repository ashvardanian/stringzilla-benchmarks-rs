# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cryptography",
#   "pycryptodome",
#   "pynacl",
# ]
# ///
"""AEAD benchmarks in Python across cryptography, pycryptodome and libsodium. Mirrors `encryption/bench.rs`."""

import argparse
import sys
from collections.abc import Callable

import nacl.bindings as libsodium
from Crypto.Cipher import AES, ChaCha20_Poly1305
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from utils import (
    MeasureSpec,
    add_common_args,
    finish,
    log_dataset,
    log_timing_overhead,
    measure,
    pass_over,
    resolve_dataset,
    set_filter,
    should_run,
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


def bench_encrypt(name: str, tokens: list[bytes], encrypt: Callable[[bytes, bytes], object]):
    """One pass encrypts every token; bytes/s is over the plaintext."""
    nonces = [nonce_for(index) for index in range(len(tokens))]
    work = MeasureSpec(
        report="bytes",
        elements=len(tokens),
        total_bytes=sum(len(token) for token in tokens),
    )
    measure(name, work, pass_over(encrypt, tokens, nonces))


def bench_decrypt(
    name: str,
    blobs: list[object],
    plaintext_lengths: list[int],
    decrypt: Callable[[object, bytes], object],
):
    """One pass decrypts every blob; bytes/s is over the original plaintext."""
    nonces = [nonce_for(index) for index in range(len(blobs))]
    work = MeasureSpec(report="bytes", elements=len(blobs), total_bytes=sum(plaintext_lengths))
    measure(name, work, pass_over(decrypt, blobs, nonces))


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

    set_filter(args.filter)

    dataset = resolve_dataset("encryption", as_bytes=True, dataset_path=args.dataset)
    tokens = dataset.tokens
    log_dataset(dataset)
    log_timing_overhead()
    log_system_info()

    plaintext_lengths = [len(token) for token in tokens]

    print("\n# encryption")
    for build in CIPHERS:
        label, encrypt, _ = build()
        bench_encrypt(f"encryption/{label}", tokens, encrypt)

    print("\n# decryption")
    for build in CIPHERS:
        label, encrypt, decrypt = build()
        if should_run(f"decryption/{label}"):
            blobs = [encrypt(token, nonce_for(index)) for index, token in enumerate(tokens)]
            bench_decrypt(f"decryption/{label}", blobs, plaintext_lengths, decrypt)

    finish()
    return 0


if __name__ == "__main__":
    exit(main())
