# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "xxhash",
#   "blake3",
#   "google-crc32c",
#   "mmh3",
#   "cityhash",
# ]
# ///
"""Hash benchmarks in Python: stateless, stateful and checksum digests. Mirrors `hash/bench.rs`."""

import argparse
import hashlib
import sys
from collections.abc import Callable
from importlib.metadata import version as pkg_version
from typing import Any

import blake3
import cityhash
import google_crc32c
import mmh3
import stringzilla as sz
import xxhash

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
)


def log_system_info():
    """Log Python version and hash library versions."""
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- xxHash: {xxhash.VERSION}")
    print(f"- Blake3: {blake3.__version__}")
    print(f"- google-crc32c: {pkg_version('google-crc32c')}")
    print(f"- mmh3: {pkg_version('mmh3')}")
    print(f"- cityhash: {pkg_version('cityhash')}")
    print()  # Add blank line


def bench_hash_function(
    name: str,
    tokens: list[bytes],
    hash_func: Callable[[bytes], Any],
    work: MeasureSpec,
) -> None:
    """
    Benchmark a stateless hash function over the whole working set.

    One pass hashes every token, driven from C so the interpreter never appears in
    the measured region — a Python-level loop body costs ~50-80 ns per item and
    would be attributed to the kernel.
    """
    measure(name, work, pass_over(hash_func, tokens))


def run_stateless_benchmarks(
    tokens: list[bytes],
    work: MeasureSpec,
):
    print("\nStateless Hash Benchmarks")

    # Python built-in hash
    bench_hash_function("stateless/hash", tokens, lambda x: hash(x), work)

    # xxHash
    bench_hash_function("stateless/xxhash.xxh3_64", tokens, xxhash.xxh3_64_intdigest, work)

    # StringZilla hashes
    bench_hash_function("stateless/stringzilla.hash", tokens, lambda x: sz.hash(x), work)

    # Google CRC32C (Castagnoli) one-shot
    bench_hash_function("stateless/google_crc32c.value", tokens, lambda x: google_crc32c.value(x), work)

    # MurmurHash3 — stateless
    bench_hash_function("stateless/mmh3.hash32", tokens, lambda x: mmh3.hash(x, signed=False), work)
    bench_hash_function("stateless/mmh3.hash64", tokens, lambda x: mmh3.hash64(x, signed=False)[0], work)
    bench_hash_function("stateless/mmh3.hash128", tokens, lambda x: mmh3.hash128(x, signed=False), work)

    # CityHash — stateless
    bench_hash_function("stateless/cityhash.CityHash64", tokens, lambda x: cityhash.CityHash64(x), work)
    bench_hash_function("stateless/cityhash.CityHash128", tokens, lambda x: cityhash.CityHash128(x), work)


def bench_stateful_hash(
    name: str,
    tokens: list[bytes],
    hasher_factory: Callable,
    work: MeasureSpec,
) -> None:
    """
    Benchmark a stateful hash by streaming the whole working set through one hasher.

    The hasher is rebuilt per pass so every sample does identical work; the old
    version built it once for the entire run, which is not what the Rust side does.
    """

    def one_pass() -> None:
        hasher = hasher_factory()
        pass_over(hasher.update, tokens)()
        hasher.digest() if hasattr(hasher, "digest") else hasher.intdigest()

    measure(name, work, one_pass)


def run_stateful_benchmarks(
    tokens: list[bytes],
    work: MeasureSpec,
):
    print("\nStateful Hash Benchmarks")

    # xxHash stateful
    bench_stateful_hash("stateful/xxhash.xxh3_64", tokens, lambda: xxhash.xxh3_64(), work)

    # StringZilla stateful hasher
    bench_stateful_hash("stateful/stringzilla.Hasher", tokens, lambda: sz.Hasher(), work)

    # Google CRC32C (Castagnoli) stateful
    bench_stateful_hash("stateful/google_crc32c.Checksum", tokens, lambda: google_crc32c.Checksum(), work)


def run_checksum_benchmarks(
    tokens: list[bytes],
    work: MeasureSpec,
):
    print("\nChecksum Hash Benchmarks")

    # StringZilla bytesum - reference lower bound
    bench_hash_function("checksum/stringzilla.bytesum", tokens, lambda x: sz.bytesum(x), work)

    # Blake3 - cryptographic hash
    bench_hash_function("checksum/blake3.blake3", tokens, lambda x: blake3.blake3(x).digest(), work)

    # SHA256 via hashlib (Python standard library)
    bench_hash_function("checksum/hashlib.sha256", tokens, lambda x: hashlib.sha256(x).digest(), work)

    # SHA256 via StringZilla
    bench_hash_function("checksum/stringzilla.Sha256", tokens, lambda x: sz.Sha256().update(x).digest(), work)


_main_epilog = """
Examples:

  %(prog)s --dataset README.md --tokens lines

  # Test only specific hash functions
  %(prog)s --dataset data.txt --tokens lines -k "xxhash|stringzilla"

  # Compare stateless vs stateful hashing
  %(prog)s --dataset large.txt --tokens words -k "hash"

  # Test cryptographic hash performance
  %(prog)s --dataset text.txt --tokens lines -k "blake3"
"""


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark hash functions with StringZilla and other implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )

    add_common_args(parser)

    args = parser.parse_args()

    # Compile filter pattern
    set_filter(args.filter)

    # Resolve the working set from the shared manifest, identically to `utils.rs`.
    dataset = resolve_dataset("hash", as_bytes=True, dataset_path=args.dataset)
    tokens = dataset.tokens
    log_dataset(dataset)
    log_system_info()

    # Run benchmarks
    work = MeasureSpec(report="bytes", elements=dataset.token_count, total_bytes=dataset.token_bytes)
    log_timing_overhead()
    run_stateless_benchmarks(tokens, work)
    run_stateful_benchmarks(tokens, work)
    run_checksum_benchmarks(tokens, work)

    finish()
    return 0


if __name__ == "__main__":
    exit(main())
