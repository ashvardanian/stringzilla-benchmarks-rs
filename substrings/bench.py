# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.1.0",
#   "stringzillas-cpus>=5.1.0",
#   "pyahocorasick",
#   "ahocorasick-rs",
#   "bm25s",
#   "numpy",
# ]
# ///
"""
Multi-pattern search benchmarks in Python: one dictionary of needles walked over every document.

- stringzillas.Substrings: count, find, replace and BM25 over a compiled Aho-Corasick automaton
- ahocorasick_rs: the `aho-corasick` crate bound to Python, all three match kinds
- pyahocorasick: a C automaton with an overlapping walk and a longest-match walk
- re: an alternation of escaped literals, the baseline most people actually write
- bm25s: an index-then-query scorer, for the scoring group only

Two framing rules, or the table lies. The head-to-head row is `<1cpu>`, since every rival is
single-threaded and CPU-only. And scoring is scan versus index: `bm25s` tokenizes a corpus and builds a
sparse index once, then answers many queries, while `Substrings` scans raw bytes against a fixed
dictionary and builds nothing per query - so build and query are reported separately.

The dictionary comes from the corpus itself: whitespace-cut words of 3 to 32 bytes, with every term
occurring once and the most frequent one percent dropped, then sliced by frequency rank. That is the
same recipe `bench/substrings.cpp` uses in the StringZilla tree, so the numbers sit beside
`include/stringzillas/substrings/README.md`.

Environment variables:
- STRINGWARS_DATASET: Path to input dataset file
- STRINGWARS_DATASET_LIMIT: Read at most this many bytes ('64mb', '1gb', '0' for all)
- STRINGWARS_TOKENS: Tokenization mode shaping the documents ('lines', 'words', 'file')
- STRINGWARS_FILTER: Regex selecting which benchmarks to run

Examples:
  uv run --with stringzillas-cpus substrings/bench.py --dataset data/xlsum/xlsum.csv
  STRINGWARS_DATASET_LIMIT=64mb uv run --with stringzillas-cpus substrings/bench.py -k "count"
"""

import argparse
import collections
import re
import sys

import numpy as np
import stringzilla as sz
import stringzillas as szs

from utils import (
    add_common_args,
    load_dataset,
    now_nanoseconds,
    report_stats,
    should_run,
    tokenize_dataset,
)

try:
    import ahocorasick as pyahocorasick
except ImportError:
    pyahocorasick = None

try:
    import ahocorasick_rs
except ImportError:
    ahocorasick_rs = None

try:
    import bm25s
except ImportError:
    bm25s = None

# region: Vocabulary

#: Shortest word admitted into the dictionary: anything under three bytes matches at nearly every
#: position, so the benchmark would measure match materialization rather than the automaton walk.
VOCABULARY_MIN_WORD_BYTES = 3

#: Longest word admitted. Longer whitespace-cut tokens are unsegmented CJK runs and URLs rather than
#: words, and the longest needle sets the warm-up every GPU chunk re-walks.
VOCABULARY_MAX_WORD_BYTES = 32

#: Which end of the frequency-ordered vocabulary a dictionary is drawn from, and how much of it.
VOCABULARY_SLICES = [
    "most_frequent_1%",
    "most_frequent_10%",
    "least_frequent_1%",
    "least_frequent_10%",
    "entire",
]


def build_vocabulary(corpus: str) -> list:
    """Counts word frequencies over the whole corpus and drops both noisy ends, returning the survivors
    ordered by descending frequency with ties broken by content so the ranking is reproducible."""
    frequencies = collections.Counter(
        word for word in corpus.split() if VOCABULARY_MIN_WORD_BYTES <= len(word.encode()) <= VOCABULARY_MAX_WORD_BYTES
    )
    distinct_count = len(frequencies)
    ranked = sorted(
        ((term, count) for term, count in frequencies.items() if count >= 2),
        key=lambda entry: (-entry[1], entry[0]),
    )
    # The top cutoff keeps its base: a fraction of ALL distinct terms, hapax included.
    return [term for term, _ in ranked[distinct_count // 100 :]]


def needle_slice_of(vocabulary: list, slice_name: str) -> list:
    """One contiguous slice of the frequency-ordered vocabulary."""
    available = len(vocabulary)
    if slice_name == "most_frequent_1%":
        first, wanted = 0, available // 100
    elif slice_name == "most_frequent_10%":
        first, wanted = 0, available // 10
    elif slice_name == "least_frequent_1%":
        wanted = available // 100
        first = available - wanted
    elif slice_name == "least_frequent_10%":
        wanted = available // 10
        first = available - wanted
    else:
        first, wanted = 0, available
    return vocabulary[first : first + max(1, min(wanted, available))]


# endregion: Vocabulary

# region: Harness


def bench_one(name: str, corpus_bytes: int, time_limit: float, filter_pattern, routine) -> None:
    """Runs `routine` until the deadline, reporting one full corpus pass per call."""
    if not should_run(name, filter_pattern):
        return
    routine()  # ? One uncounted warm-up, so caches and frequency settle before the clock starts.

    start = now_nanoseconds()
    deadline = start + int(time_limit * 1e9)
    calls, matched = 0, 0
    while now_nanoseconds() < deadline:
        matched += routine()
        calls += 1
    seconds = (now_nanoseconds() - start) / 1e9
    print(f"{name}: {matched:,} results over {calls:,} passes", file=sys.stderr)
    report_stats(name, "bytes", seconds, calls, corpus_bytes * calls)


# endregion: Harness

# region: Searching


def bench_counting(documents, tape, needles, slice_name, corpus_bytes, scopes, time_limit, filter_pattern):
    """Counts every overlapping match of every needle, the cheapest walk each engine offers."""
    for scope_name, device, capabilities in scopes:
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def count_with_stringzillas(engine=engine, device=device):
            return int(engine.count(tape, device=device, policy="overlapping")[1])

        bench_one(
            f"count-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            count_with_stringzillas,
        )

    if ahocorasick_rs is not None:
        automaton = ahocorasick_rs.AhoCorasick(needles, matchkind=ahocorasick_rs.MatchKind.Standard)

        def count_with_ahocorasick_rs():
            return sum(len(automaton.find_matches_as_indexes(document, overlapping=True)) for document in documents)

        bench_one(
            f"count-{slice_name}/ahocorasick_rs.find_matches_as_indexes",
            corpus_bytes,
            time_limit,
            filter_pattern,
            count_with_ahocorasick_rs,
        )

    if pyahocorasick is not None:
        automaton = pyahocorasick.Automaton()
        for index, needle in enumerate(needles):
            automaton.add_word(needle, index)
        automaton.make_automaton()

        def count_with_pyahocorasick():
            return sum(sum(1 for _ in automaton.iter(document)) for document in documents)

        bench_one(
            f"count-{slice_name}/pyahocorasick.Automaton.iter",
            corpus_bytes,
            time_limit,
            filter_pattern,
            count_with_pyahocorasick,
        )


def bench_cover(documents, tape, needles, slice_name, corpus_bytes, scopes, time_limit, filter_pattern):
    """Counts a leftmost cover, the walk a rewrite needs and the one every engine spells differently."""
    for scope_name, device, capabilities in scopes:
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def cover_with_stringzillas(engine=engine, device=device):
            return int(engine.count(tape, device=device, policy="leftmost-longest")[1])

        bench_one(
            f"cover-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            cover_with_stringzillas,
        )

    if ahocorasick_rs is not None:
        automaton = ahocorasick_rs.AhoCorasick(needles, matchkind=ahocorasick_rs.MatchKind.LeftmostLongest)

        def cover_with_ahocorasick_rs():
            return sum(len(automaton.find_matches_as_indexes(document)) for document in documents)

        bench_one(
            f"cover-{slice_name}/ahocorasick_rs.find_matches_as_indexes<LeftmostLongest>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            cover_with_ahocorasick_rs,
        )

    # An alternation of escaped literals is what a caller reaches for before learning about automata.
    # It resolves leftmost-first, so it belongs beside the covers rather than the overlapping walk.
    alternation = re.compile("|".join(re.escape(needle) for needle in needles))

    def cover_with_re():
        return sum(sum(1 for _ in alternation.finditer(document)) for document in documents)

    bench_one(
        f"cover-{slice_name}/re.finditer",
        corpus_bytes,
        time_limit,
        filter_pattern,
        cover_with_re,
    )


def bench_rewriting(tape, needles, slice_name, corpus_bytes, scopes, time_limit, filter_pattern):
    """Substitutes every match under a cover, the one verb whose output is another corpus.

    Replacing every needle with itself keeps the output the size of the input, so the cell measures the
    rewrite rather than an expansion factor, and its result is checkable by eye.
    """
    replacements = sz.Strs(needles)
    for scope_name, device, capabilities in scopes:
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def replace_with_stringzillas(engine=engine, device=device):
            return len(engine.replace(tape, replacements, device=device)[0])

        bench_one(
            f"replace-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            replace_with_stringzillas,
        )


# endregion: Searching

# region: Scoring


def bench_scoring(documents, tape, needles, slice_name, corpus_bytes, scopes, time_limit, filter_pattern):
    """Scores every document against the dictionary, which is the query."""
    weights = np.ones(len(needles), dtype=np.float32)
    mean_document_length = corpus_bytes / max(1, len(documents))

    for scope_name, device, capabilities in scopes:
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def score_with_stringzillas(engine=engine, device=device):
            return len(engine.score_bm25(tape, weights, mean_document_length, device=device))

        bench_one(
            f"bm25-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            score_with_stringzillas,
        )


def bench_scoring_index(documents, slice_name, corpus_bytes, time_limit, filter_pattern):
    """The index-then-query rival, measured on one slice only: its build is a tokenization pass over the
    whole corpus, which is the cost the scan above never pays."""
    if bm25s is None:
        return

    def build_index():
        tokenized = bm25s.tokenize(documents, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokenized, show_progress=False)
        return len(documents)

    bench_one(
        f"bm25-{slice_name}/bm25s.BM25.index",
        corpus_bytes,
        time_limit,
        filter_pattern,
        build_index,
    )

    tokenized = bm25s.tokenize(documents, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=False)
    query = bm25s.tokenize([documents[0]], show_progress=False)

    def query_index():
        results, _ = retriever.retrieve(query, k=min(10, len(documents)), show_progress=False)
        return int(results.size)

    bench_one(
        f"bm25-{slice_name}/bm25s.BM25.retrieve",
        corpus_bytes,
        time_limit,
        filter_pattern,
        query_index,
    )


# endregion: Scoring


def main():
    parser = argparse.ArgumentParser(description="Multi-pattern search benchmarks")
    add_common_args(
        parser, default_dataset="data/xlsum/xlsum.csv", default_tokens="lines", default_dataset_limit="64mb"
    )
    args = parser.parse_args()

    corpus = load_dataset(args.dataset, as_bytes=False, size_limit=args.dataset_limit)
    documents = tokenize_dataset(corpus, args.tokens)
    corpus_bytes = sum(len(document.encode()) for document in documents)
    filter_pattern = re.compile(args.filter) if args.filter else None

    vocabulary = build_vocabulary(corpus)
    print(f"- vocabulary: {len(vocabulary):,} terms after dropping hapax and the top one percent")
    print(f"- corpus: {len(documents):,} documents, {corpus_bytes:,} bytes")
    if not vocabulary:
        print("The corpus yielded no vocabulary; is the dataset path right?", file=sys.stderr)
        return

    tape = sz.Strs(documents)
    scopes = [("1cpu", szs.DeviceScope(cpu_cores=1), ("serial",))]
    scopes.append(("Ncpu", szs.DeviceScope(cpu_cores=0), ("serial", "parallel")))
    if "cuda" in szs.__capabilities__:
        scopes.append(("1gpu", szs.DeviceScope(gpu_device=0), ("cuda",)))

    for slice_name in VOCABULARY_SLICES:
        needles = needle_slice_of(vocabulary, slice_name)
        needle_bytes = sum(len(needle.encode()) for needle in needles)
        print(f"\n# {slice_name} - {len(needles):,} needles, {needle_bytes:,} bytes")

        bench_counting(documents, tape, needles, slice_name, corpus_bytes, scopes, args.time_limit, filter_pattern)
        bench_cover(documents, tape, needles, slice_name, corpus_bytes, scopes, args.time_limit, filter_pattern)
        bench_rewriting(tape, needles, slice_name, corpus_bytes, scopes, args.time_limit, filter_pattern)
        bench_scoring(documents, tape, needles, slice_name, corpus_bytes, scopes, args.time_limit, filter_pattern)

    print("\n# index-then-query, for the scoring regime this engine does not occupy")
    bench_scoring_index(documents, VOCABULARY_SLICES[0], corpus_bytes, args.time_limit, filter_pattern)


if __name__ == "__main__":
    main()
