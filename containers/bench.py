# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "pyprobables",
#   "xxhash",
#   "numpy",
# ]
# ///
"""Hash-container benchmarks in Python: multi-seed digests and filters. Mirrors `containers/bench.rs`."""

import argparse
import sys
from array import array
from collections.abc import Callable

import numpy as np
import stringzilla as sz
import xxhash
from probables import BloomFilter

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

# Sixteen fixed odd seeds shared by every multi-hash variant, matching containers/bench.rs.
SEEDS = [
    0x9E3779B97F4A7C15,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0xD1B54A32D192ED03,
    0xA0761D6478BD642F,
    0xE7037ED1A0B428DB,
    0x8EBC6AF09C88C6E3,
    0x589965CC75374CC3,
    0x1D8E4E27C47D124F,
    0xEB44ACCAB455D165,
    0x2545F4914F6CDD1D,
    0xFF51AFD7ED558CCD,
    0xC4CEB9FE1A85EC53,
    0xBF58476D1CE4E5B9,
    0x94175CC1BAB35C97,
    0x4CF5AD432745937F,
]

MASK_64 = (1 << 64) - 1
TARGET_FALSE_POSITIVE_RATE = 0.01


def log_system_info():
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- xxHash: {xxhash.VERSION}")
    print()


def bench_multihash(
    name: str,
    tokens: list[bytes],
    produce: Callable[[bytes], object],
    digest_bits: int,
):
    """One pass emits `digest_bits` digest bits for every token."""
    work = MeasureSpec(
        report="bits",
        elements=len(tokens) * digest_bits,
        total_bytes=sum(len(token) for token in tokens),
    )
    measure(name, work, pass_over(produce, tokens))


def bench_build(name: str, count: int, total_bytes: int, build: Callable[[], object]):
    """One pass rebuilds the whole filter from every key."""
    work = MeasureSpec(report="hashes", elements=count, total_bytes=total_bytes)
    measure(name, work, build)


def bench_query(name: str, probes: list[bytes], query: Callable[[bytes], bool]):
    """One pass queries every probe."""
    work = MeasureSpec(
        report="hashes",
        elements=len(probes),
        total_bytes=sum(len(probe) for probe in probes),
    )
    measure(name, work, pass_over(query, probes))


def report_quality(
    label: str,
    num_bits: int,
    inserted_count: int,
    absent: list[bytes],
    contains: Callable[[bytes], bool],
):
    """Report the measured false-positive rate over the held-out absent words plus bits-per-key."""
    false_positives = sum(1 for token in absent if contains(token))
    rate = false_positives / len(absent) * 100 if absent else 0.0
    bits = f"{num_bits / max(inserted_count, 1):5.2f} bits/key" if num_bits else "    n/a    "
    print(f"    {label:<38} {bits}, measured FPR {rate:.3f}%")


def run_multihash(tokens: list[bytes]):
    for digest_bits in (128, 256, 512, 1024):
        print(f"# multihash ({digest_bits}-bit digest)")
        sz_hashes = digest_bits // 64
        xxh3_calls = digest_bits // 128
        seeds = array("Q", SEEDS[:sz_hashes])
        # One preallocated digest buffer reused every call, so no variant pays per-call allocation.
        out = np.empty(sz_hashes, dtype=np.uint64)

        bench_multihash(
            "multihash/stringzilla.hash_multiseed",
            tokens,
            lambda token, seeds=seeds, out=out: sz.hash_multiseed(token, seeds, out),
            digest_bits,
        )

        def sz_hash_fill(token, seeds=seeds, out=out):
            for index, seed in enumerate(seeds):
                out[index] = sz.hash(token, seed)

        bench_multihash(
            "multihash/stringzilla.hash",
            tokens,
            sz_hash_fill,
            digest_bits,
        )

        # One full 128-bit xxh3 hash per seed — every bit is independent, no double-hashing —
        # split across two 64-bit slots of the shared buffer.
        def xxh3_fill(token, n=xxh3_calls, out=out):
            for index in range(n):
                wide = xxhash.xxh3_128_intdigest(token, seed=SEEDS[index])
                out[2 * index] = wide & MASK_64
                out[2 * index + 1] = wide >> 64

        bench_multihash("multihash/xxhash.xxh3_128", tokens, xxh3_fill, digest_bits)


def run_filters(unique: list[bytes]):
    inserted_count = min(len(unique) * 8 // 10, 1_000_000) or 1
    inserted = unique[:inserted_count]
    absent = unique[inserted_count:]
    inserted_bytes = sum(len(token) for token in inserted)
    print(f"# filters ({inserted_count} inserted, {len(absent)} held-out absent)")

    # pyprobables Bloom filter, hashing each key with its default FNV-1a internally.
    bloom = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
    for token in inserted:
        bloom.add(token)
    report_quality("bloom/pyprobables<fnv>", bloom.number_bits, inserted_count, absent, bloom.check)

    def build_bloom():
        filter_ = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
        for token in inserted:
            filter_.add(token)

    bench_build("bloom/pyprobables.add<fnv>", inserted_count, inserted_bytes, build_bloom)
    bench_query("bloom/pyprobables.check<fnv>", inserted, bloom.check)

    # Same filter, fed StringZilla's `hash_multiseed` digest through the precomputed-hash API: one
    # native call fills the reused buffer, then `add_alt`/`check_alt` consume it — no per-key callback.
    bloom_sz = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
    seeds = array("Q", SEEDS[: bloom_sz.number_hashes])
    out = np.empty(bloom_sz.number_hashes, dtype=np.uint64)

    def sz_digest(token):
        sz.hash_multiseed(token, seeds, out)
        return out

    for token in inserted:
        bloom_sz.add_alt(sz_digest(token))
    report_quality(
        "bloom/pyprobables<stringzilla>",
        bloom_sz.number_bits,
        inserted_count,
        absent,
        lambda t: bloom_sz.check_alt(sz_digest(t)),
    )

    def build_bloom_sz():
        filter_ = BloomFilter(est_elements=inserted_count, false_positive_rate=TARGET_FALSE_POSITIVE_RATE)
        for token in inserted:
            filter_.add_alt(sz_digest(token))

    bench_build("bloom/pyprobables.add<stringzilla>", inserted_count, inserted_bytes, build_bloom_sz)
    bench_query(
        "bloom/pyprobables.check<stringzilla>",
        inserted,
        lambda t: bloom_sz.check_alt(sz_digest(t)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-way word hashing and probabilistic membership structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    set_filter(args.filter)

    dataset = resolve_dataset("containers", as_bytes=True, dataset_path=args.dataset)
    log_dataset(dataset)
    log_timing_overhead()
    tokens = dataset.tokens
    if not tokens:
        print("No tokens found in dataset")
        return 1

    unique = list(dict.fromkeys(tokens))
    total_bytes = sum(len(token) for token in tokens)
    print(f"Dataset: {len(tokens):,} tokens, {total_bytes:,} bytes, {len(unique):,} unique")
    log_system_info()

    run_multihash(tokens)
    run_filters(unique)
    finish()
    return 0


if __name__ == "__main__":
    exit(main())
