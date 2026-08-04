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
"""
Python hash function benchmarks comparing various implementations.

Benchmarks hash function performance using consistent methodology with the Rust
hash/bench.rs implementation, focusing on three categories of hashing patterns.

Benchmark categories:
- Stateless: Hash each token independently (non-cryptographic)
- Stateful: Incremental hashing across all tokens (non-cryptographic)
- Checksum: Cryptographic hashes and reference bounds

Hash functions compared:
- Built-in Python: hash()
- StringZilla: sz.hash(), sz.bytesum(), sz.Sha256
- xxHash: xxh3_64, xxh64, xxh32 variants
- Blake3: Modern cryptographic hash
- hashlib: SHA256 for comparison with StringZilla

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_TOKENS: Tokenization mode ('lines', 'words', 'file')

Examples:
  uv run hash/bench.py --dataset README.md --tokens lines
  uv run hash/bench.py --dataset data/xlsum/xlsum.csv --tokens words -k "xxhash"
  STRINGWARS_DATASET=README.md STRINGWARS_TOKENS=lines uv run hash/bench.py
"""

import argparse
import hashlib
import re
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
    add_common_args,
    load_dataset,
    now_nanoseconds,
    paced_items,
    report_stats,
    resolve_tokens,
    should_run,
    tokenize_dataset,
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
    time_limit_seconds: float = 10.0,
) -> None:
    """
    Benchmark a stateless hash function and report throughput.

    Processes tokens until time limit is reached, then reports results.
    """
    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    processed_tokens = 0
    processed_bytes = 0

    # Stateless: hash each token independently.
    for token in paced_items(tokens, deadline_nanoseconds):
        _ = hash_func(token)
        processed_tokens += 1
        processed_bytes += len(token)

    end_time = now_nanoseconds()

    seconds = (end_time - start_time) / 1e9
    report_stats(name, "bytes", seconds, processed_tokens, processed_bytes)


def run_stateless_benchmarks(
    tokens: list[bytes],
    filter_pattern: re.Pattern | None = None,
    time_limit_seconds: float = 10.0,
):
    """Run stateless hash benchmarks (hash each token independently)."""
    print("\nStateless Hash Benchmarks")

    # Python built-in hash
    if should_run("stateless/hash", filter_pattern):
        bench_hash_function("hash", tokens, lambda x: hash(x), time_limit_seconds)

    # xxHash
    if should_run("stateless/xxhash.xxh3_64", filter_pattern):
        bench_hash_function("xxhash.xxh3_64", tokens, lambda x: xxhash.xxh3_64(x).intdigest(), time_limit_seconds)

    # StringZilla hashes
    if should_run("stateless/stringzilla.hash", filter_pattern):
        bench_hash_function("stringzilla.hash", tokens, lambda x: sz.hash(x), time_limit_seconds)

    # Google CRC32C (Castagnoli) one-shot
    if should_run("stateless/google_crc32c.value", filter_pattern):
        bench_hash_function("google_crc32c.value", tokens, lambda x: google_crc32c.value(x), time_limit_seconds)

    # MurmurHash3 — stateless
    if should_run("stateless/mmh3.hash32", filter_pattern):
        bench_hash_function("mmh3.hash32", tokens, lambda x: mmh3.hash(x, signed=False), time_limit_seconds)
    if should_run("stateless/mmh3.hash64", filter_pattern):
        bench_hash_function("mmh3.hash64", tokens, lambda x: mmh3.hash64(x, signed=False)[0], time_limit_seconds)
    if should_run("stateless/mmh3.hash128", filter_pattern):
        bench_hash_function("mmh3.hash128", tokens, lambda x: mmh3.hash128(x, signed=False), time_limit_seconds)

    # CityHash — stateless
    if should_run("stateless/cityhash.CityHash64", filter_pattern):
        bench_hash_function("cityhash.CityHash64", tokens, lambda x: cityhash.CityHash64(x), time_limit_seconds)
    if should_run("stateless/cityhash.CityHash128", filter_pattern):
        bench_hash_function("cityhash.CityHash128", tokens, lambda x: cityhash.CityHash128(x), time_limit_seconds)


def bench_stateful_hash(
    name: str,
    tokens: list[bytes],
    hasher_factory: Callable,
    time_limit_seconds: float = 10.0,
) -> None:
    """Benchmark a stateful hash function and report throughput."""
    start_time = now_nanoseconds()
    deadline_nanoseconds = start_time + int(time_limit_seconds * 1e9)

    processed_tokens = 0
    processed_bytes = 0

    hasher = hasher_factory()
    for token in paced_items(tokens, deadline_nanoseconds):
        hasher.update(token)
        processed_tokens += 1
        processed_bytes += len(token)

    _ = hasher.digest() if hasattr(hasher, "digest") else hasher.intdigest()
    end_time = now_nanoseconds()

    seconds = (end_time - start_time) / 1e9
    report_stats(name, "bytes", seconds, processed_tokens, processed_bytes)


def run_stateful_benchmarks(
    tokens: list[bytes],
    filter_pattern: re.Pattern | None = None,
    time_limit_seconds: float = 10.0,
):
    """Run stateful hash benchmarks (incremental/streaming hashing)."""
    print("\nStateful Hash Benchmarks")

    # xxHash stateful
    if should_run("stateful/xxhash.xxh3_64", filter_pattern):
        bench_stateful_hash("xxhash.xxh3_64", tokens, lambda: xxhash.xxh3_64(), time_limit_seconds)

    # StringZilla stateful hasher
    if should_run("stateful/stringzilla.Hasher", filter_pattern):
        bench_stateful_hash("stringzilla.Hasher", tokens, lambda: sz.Hasher(), time_limit_seconds)

    # Google CRC32C (Castagnoli) stateful
    if should_run("stateful/google_crc32c.Checksum", filter_pattern):
        bench_stateful_hash("google_crc32c.Checksum", tokens, lambda: google_crc32c.Checksum(), time_limit_seconds)


def run_checksum_benchmarks(
    tokens: list[bytes],
    filter_pattern: re.Pattern | None = None,
    time_limit_seconds: float = 10.0,
):
    """Run checksum/cryptographic hash benchmarks."""
    print("\nChecksum Hash Benchmarks")

    # StringZilla bytesum - reference lower bound
    if should_run("checksum/stringzilla.bytesum", filter_pattern):
        bench_hash_function("stringzilla.bytesum", tokens, lambda x: sz.bytesum(x), time_limit_seconds)

    # Blake3 - cryptographic hash
    if should_run("checksum/blake3.blake3", filter_pattern):
        bench_hash_function("blake3.blake3", tokens, lambda x: blake3.blake3(x).digest(), time_limit_seconds)

    # SHA256 via hashlib (Python standard library)
    if should_run("checksum/hashlib.sha256", filter_pattern):
        bench_hash_function("hashlib.sha256", tokens, lambda x: hashlib.sha256(x).digest(), time_limit_seconds)

    # SHA256 via StringZilla
    if should_run("checksum/stringzilla.Sha256", filter_pattern):
        bench_hash_function("stringzilla.Sha256", tokens, lambda x: sz.Sha256().update(x).digest(), time_limit_seconds)


_main_epilog = """
Examples:

  # Benchmark all hash functions with default settings
  %(prog)s --dataset README.md --tokens lines

  # Test only specific hash functions
  %(prog)s --dataset README.md --tokens lines -k "xxhash|stringzilla"

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
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    # Load and tokenize dataset
    dataset = load_dataset(args.dataset, as_bytes=True, size_limit=args.dataset_limit)
    tokens_mode = resolve_tokens(args.tokens, "words")
    tokens = tokenize_dataset(dataset, tokens_mode)

    if not tokens:
        print("No tokens found in dataset")
        return 1

    # Report dataset info
    total_bytes = sum(len(token) for token in tokens)
    avg_token_length = total_bytes / len(tokens) if tokens else 0
    print(f"Dataset: {len(tokens):,} tokens, {total_bytes:,} bytes, {avg_token_length:.1f} avg token length")
    log_system_info()

    # Run benchmarks
    run_stateless_benchmarks(tokens, filter_pattern, args.time_limit)
    run_stateful_benchmarks(tokens, filter_pattern, args.time_limit)
    run_checksum_benchmarks(tokens, filter_pattern, args.time_limit)

    return 0


if __name__ == "__main__":
    exit(main())
