# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.0.0",
#   "numpy",
#   "pycryptodome",
#   "opencv-python",
# ]
# ///
"""Low-level memory benchmarks in Python: lookup tables, PRNG fills, copies. Mirrors `memory/bench.rs`."""

import argparse
import sys
from collections.abc import Callable, Iterable
from itertools import repeat

import Crypto as pycryptodome
import cv2
import numpy as np
import stringzilla as sz
from Crypto.Cipher import AES as PyCryptoDomeAES

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
    print(f"- Python: {sys.version.split()[0]}, {sys.platform}")
    print(f"- StringZilla: {sz.__version__} with {sz.__capabilities_str__}")
    print(f"- NumPy: {np.__version__}")
    print(f"- PyCryptoDome: {pycryptodome.__version__}")
    print(f"- OpenCV: {cv2.__version__} (defaults to {cv2.getNumThreads()} threads)")
    print()


def sz_translate_allocating(haystack: bytes, look_up_table: bytes) -> int:
    """StringZilla translation with allocation (bytes input)."""
    result = sz.translate(haystack, look_up_table)
    return len(result)


def sz_translate_inplace(haystack: memoryview, look_up_table: bytes) -> int:
    """StringZilla translation in-place (memoryview input)."""
    sz.translate(haystack, look_up_table, inplace=True)
    return len(haystack)


def bytes_translate(haystack_bytes: bytes, lut: bytes) -> int:
    result = haystack_bytes.translate(lut)
    return len(result)


def opencv_lut_allocating(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """OpenCV LUT with allocation."""
    result = cv2.LUT(haystack_array, lut)
    return len(result)


def opencv_lut_inplace(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """OpenCV LUT in-place."""
    cv2.LUT(haystack_array, lut, dst=haystack_array)
    return len(haystack_array)


def numpy_lut_indexing_allocating(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy array indexing (always allocating)."""
    result = lut[haystack_array]
    return len(result)


def numpy_lut_indexing_inplace(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy array indexing in-place."""
    haystack_array[:] = lut[haystack_array]
    return len(haystack_array)


def numpy_lut_take_allocating(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy take function (always allocating)."""
    result = np.take(lut, haystack_array)
    return len(result)


def numpy_lut_take_inplace(haystack_array: np.ndarray, lut: np.ndarray) -> int:
    """NumPy take function in-place."""
    np.take(lut, haystack_array, out=haystack_array)
    return len(haystack_array)


def bench_translate(
    name: str,
    tokens,
    table: bytes,
    operation: Callable[[object, bytes], int],
) -> None:
    # The broadcast table column used to be built *inside* the timed region, so a
    # multi-million-element list allocation was charged to every contender.
    work = MeasureSpec(
        report="bytes",
        elements=len(tokens),
        total_bytes=sum(len(token) for token in tokens),
    )
    measure(name, work, pass_over(operation, tokens, repeat(table)))


def sizes_from_tokens(tokens: Iterable[bytes]) -> list[int]:
    return [len(token) for token in tokens if len(token) > 0]


def bench_generator(name: str, sizes: list[int], generate_bytes: Callable[[int], bytes]) -> None:
    work = MeasureSpec(report="bytes", elements=len(sizes), total_bytes=sum(sizes))
    measure(name, work, pass_over(generate_bytes, sizes))


def make_pycryptodome_aes_ctr():
    key = b"\x00" * 16
    cipher = PyCryptoDomeAES.new(key, PyCryptoDomeAES.MODE_CTR, nonce=b"")

    def generate_bytes(size: int) -> bytes:
        return cipher.encrypt(b"\x00" * size)

    return generate_bytes


def make_stringzilla_fill_random():
    def generate_bytes(size: int):
        buffer = bytearray(size)
        sz.fill_random(buffer, 0)
        return buffer

    return generate_bytes


def make_numpy_pcg64():
    generator = np.random.Generator(np.random.PCG64(0))
    random_raw = generator.bit_generator.random_raw

    def generate_bytes(size: int) -> bytes:
        words = (size + 7) // 8
        raw_words = random_raw(words)
        return raw_words.view(np.uint8)[:size].tobytes()

    return generate_bytes


def make_numpy_philox():
    generator = np.random.Generator(np.random.Philox(0))
    random_raw = generator.bit_generator.random_raw

    def generate_bytes(size: int) -> bytes:
        words = (size + 7) // 8
        raw_words = random_raw(words)
        return raw_words.view(np.uint8)[:size].tobytes()

    return generate_bytes


_main_epilog = """
Examples:

  %(prog)s --dataset README.md --tokens lines

  # Filter to only translations
  %(prog)s --dataset README.md --tokens words -k "translate"
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory-related benchmarks: LUT transforms and random generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_main_epilog,
    )
    add_common_args(parser)
    args = parser.parse_args()

    # Dataset
    dataset = resolve_dataset("memory", as_bytes=True, dataset_path=args.dataset)
    tokens_b = dataset.tokens
    log_dataset(dataset)
    log_timing_overhead()
    log_system_info()

    # Compile filter
    set_filter(args.filter)

    # Disable OpenCV multithreading for more consistent results
    cv2.setNumThreads(1)

    # Lookup-table transforms
    print()
    print("LUT Transforms")

    reverse = bytes(reversed(range(256)))
    reverse_np = np.arange(255, -1, -1, dtype=np.uint8)

    tokens_np = [np.array(np.frombuffer(token, dtype=np.uint8)) for token in tokens_b]
    tokens_mv = [memoryview(bytearray(token)) for token in tokens_b]

    # Python bytes.translate (always allocating)
    bench_translate("lookup-table/bytes.translate<new>", tokens_b, reverse, bytes_translate)

    # OpenCV allocating
    bench_translate("lookup-table/opencv.LUT<new>", tokens_np, reverse_np, opencv_lut_allocating)

    # OpenCV in-place
    bench_translate("lookup-table/opencv.LUT<inplace>", tokens_np, reverse_np, opencv_lut_inplace)

    # NumPy indexing allocating
    bench_translate("lookup-table/numpy.indexing<new>", tokens_np, reverse_np, numpy_lut_indexing_allocating)

    # NumPy indexing in-place
    bench_translate("lookup-table/numpy.indexing<inplace>", tokens_np, reverse_np, numpy_lut_indexing_inplace)

    # NumPy take allocating
    bench_translate("lookup-table/numpy.take<new>", tokens_np, reverse_np, numpy_lut_take_allocating)

    # NumPy take in-place
    bench_translate("lookup-table/numpy.take<inplace>", tokens_np, reverse_np, numpy_lut_take_inplace)

    # StringZilla allocating
    bench_translate("lookup-table/stringzilla.translate<new>", tokens_b, reverse, sz_translate_allocating)

    # StringZilla in-place (need memoryviews for each token)
    bench_translate("lookup-table/stringzilla.translate<inplace>", tokens_mv, reverse, sz_translate_inplace)

    # Random byte generation
    print()
    print("Random Byte Generation")
    sizes = sizes_from_tokens(tokens_b)

    bench_generator("generate-random/pycryptodome.AES-CTR", sizes, make_pycryptodome_aes_ctr())
    bench_generator("generate-random/stringzilla.fill_random", sizes, make_stringzilla_fill_random())
    bench_generator("generate-random/stringzilla.random", sizes, sz.random)
    bench_generator("generate-random/numpy.PCG64", sizes, make_numpy_pcg64())
    bench_generator("generate-random/numpy.Philox", sizes, make_numpy_philox())

    finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
