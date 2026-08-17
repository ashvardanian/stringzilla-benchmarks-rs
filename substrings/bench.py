# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stringzilla>=5.1.0",
#   "stringzillas-cpus>=5.1.0",
#   "pyahocorasick",
#   "ahocorasick-rs",
#   "bm25s",
#   "rank-bm25",
#   "scikit-learn",
#   "numpy",
#   # cuml-cu12 adds the GPU TF-IDF row; see requirements-cuda.txt
# ]
# ///
"""
Multi-pattern search benchmarks in Python: one dictionary of needles walked over every document.

- stringzillas.Substrings: count, find, replace and BM25 over a compiled Aho-Corasick automaton
- ahocorasick_rs: the `aho-corasick` crate bound to Python, all three match kinds
- pyahocorasick: a C automaton with an overlapping walk and a longest-match walk
- re: an alternation of escaped literals, the baseline most people actually write
- bm25s, rank_bm25 and sklearn.TfidfVectorizer: index-then-query scorers, for the scoring group only

Two framing rules, or the table lies. The head-to-head row is `<1cpu>`, since every rival is
single-threaded and CPU-only. And scoring is scan versus index: the scorers tokenize a corpus and build
an inverted index once, then answer many queries, while `Substrings` scans raw bytes against a fixed
dictionary and builds nothing per query - so build and query are reported separately. A build never sees
the dictionary and is therefore the same work for every slice; a query is the slice itself, so it widens
with the dictionary exactly as the scan does.

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
import gc
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

try:
    import rank_bm25
except ImportError:
    rank_bm25 = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

try:
    import cudf
    from cuml.feature_extraction.text import TfidfVectorizer as CumlTfidfVectorizer
except ImportError:
    cudf = None
    CumlTfidfVectorizer = None

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


#: Which device one pass of a sweep runs on, and the capabilities that scope may use.
DEVICE_PASSES = [
    ("1cpu", {"cpu_cores": 1}, ("serial",)),
    ("Ncpu", {"cpu_cores": 0}, ("serial", "parallel")),
    ("1gpu", {"gpu_device": 0}, ("cuda",)),
]


def for_each_device_pass(body) -> None:
    """Calls `body(scope_name, device, capabilities)` once per available device, holding each scope only
    for that pass. An idle ForkUnion pool spin-waits, so a multi-core scope alive during a
    single-threaded cell consumes the very cores that cell is being measured on; single-threaded cells
    therefore belong to the `1cpu` pass, which runs before any pool exists.
    """
    for scope_name, arguments, capabilities in DEVICE_PASSES:
        if "gpu_device" in arguments and "cuda" not in szs.__capabilities__:
            continue
        scope = szs.DeviceScope(**arguments)
        try:
            body(scope_name, scope, capabilities)
        finally:
            del scope
            gc.collect()


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


def bench_counting(documents, tape, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """Counts every overlapping match of every needle, the cheapest walk each engine offers."""

    def one_pass(scope_name, device, capabilities):
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def count_with_stringzillas():
            return int(engine.count(tape, device=device, policy="overlapping")[1])

        bench_one(
            f"count-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            count_with_stringzillas,
        )
        # Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if scope_name == "1cpu":
            bench_counting_rivals(documents, needles, slice_name, corpus_bytes, time_limit, filter_pattern)

    for_each_device_pass(one_pass)


def bench_counting_rivals(documents, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """The single-threaded overlapping-count rivals, run only in the single-core pass."""
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


def bench_cover(documents, tape, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """Counts a leftmost cover, the walk a rewrite needs and the one every engine spells differently."""

    def one_pass(scope_name, device, capabilities):
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def cover_with_stringzillas():
            return int(engine.count(tape, device=device, policy="leftmost-longest")[1])

        bench_one(
            f"cover-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            cover_with_stringzillas,
        )
        # Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if scope_name == "1cpu":
            bench_cover_rivals(documents, needles, slice_name, corpus_bytes, time_limit, filter_pattern)

    for_each_device_pass(one_pass)


def bench_cover_rivals(documents, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """The single-threaded leftmost-cover rivals, run only in the single-core pass."""
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
    #
    # Measured on the narrowest dictionary alone: `re` scans the alternation branch by branch with no
    # literal prefilter, so its cost grows with the needle count and one pass over the corpus already
    # runs for minutes at a few thousand needles. Rust's `regex` carries a prefilter and does sweep
    # every slice, which is where the cross-slice shape of this baseline is visible.
    if slice_name != VOCABULARY_SLICES[0]:
        return
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


def bench_locating(documents, tape, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """Materializes every match rather than tallying it, which is where output bandwidth starts to show."""

    def one_pass(scope_name, device, capabilities):
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def find_with_stringzillas():
            # Four parallel uint64 arrays; the offsets one is as long as the match count.
            return len(engine.find(tape, device=device, policy="overlapping")[2])

        bench_one(
            f"find-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            find_with_stringzillas,
        )
        # Every rival is single-threaded, so it is measured in the pass that has no thread pool alive.
        if scope_name == "1cpu":
            bench_locating_rivals(documents, needles, slice_name, corpus_bytes, time_limit, filter_pattern)

    for_each_device_pass(one_pass)


def bench_locating_rivals(documents, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """The single-threaded locating rivals, run only in the single-core pass."""
    if ahocorasick_rs is None:
        return
    automaton = ahocorasick_rs.AhoCorasick(needles, matchkind=ahocorasick_rs.MatchKind.Standard)

    def find_with_ahocorasick_rs():
        total = 0
        for document in documents:
            total += len(automaton.find_matches_as_indexes(document, overlapping=True))
        return total

    bench_one(
        f"find-{slice_name}/ahocorasick_rs.find_matches_as_indexes",
        corpus_bytes,
        time_limit,
        filter_pattern,
        find_with_ahocorasick_rs,
    )


def bench_rewriting(tape, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """Substitutes every match under a cover, the one verb whose output is another corpus.

    Replacing every needle with itself keeps the output the size of the input, so the cell measures the
    rewrite rather than an expansion factor, and its result is checkable by eye.
    """
    replacements = sz.Strs(needles)

    def one_pass(scope_name, device, capabilities):
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def replace_with_stringzillas():
            return len(engine.replace(tape, replacements, device=device)[0])

        bench_one(
            f"replace-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            replace_with_stringzillas,
        )

    for_each_device_pass(one_pass)


# endregion: Searching

# region: Scoring


def bench_scoring(documents, tape, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """Scores every document against the dictionary, which is the query."""
    weights = np.ones(len(needles), dtype=np.float32)
    mean_document_length = corpus_bytes / max(1, len(documents))

    def one_pass(scope_name, device, capabilities):
        engine = szs.Substrings(sz.Strs(needles), device=device, capabilities=capabilities)

        def score_with_stringzillas():
            return len(engine.score_bm25(tape, weights, mean_document_length, device=device))

        bench_one(
            f"bm25-{slice_name}/stringzillas.Substrings<{scope_name}>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            score_with_stringzillas,
        )

    for_each_device_pass(one_pass)


def bench_index_build(documents, slice_name, corpus_bytes, time_limit, filter_pattern):
    """Builds each inverted index once, the cost the scan never pays.

    A build tokenizes the whole corpus and never sees the dictionary, so it is the same work for every
    vocabulary slice; only the narrowest slice measures it and the tables repeat that one figure.
    """
    if bm25s is not None:

        def build_bm25s():
            retriever = bm25s.BM25()
            retriever.index(bm25s.tokenize(documents, show_progress=False), show_progress=False)
            return len(documents)

        bench_one(f"bm25-{slice_name}/bm25s.BM25.index", corpus_bytes, time_limit, filter_pattern, build_bm25s)

    if rank_bm25 is not None:

        def build_rank_bm25():
            rank_bm25.BM25Okapi([document.split() for document in documents])
            return len(documents)

        bench_one(f"bm25-{slice_name}/rank_bm25.BM25Okapi", corpus_bytes, time_limit, filter_pattern, build_rank_bm25)

    if TfidfVectorizer is not None:

        def build_tfidf():
            TfidfVectorizer().fit_transform(documents)
            return len(documents)

        bench_one(
            f"bm25-{slice_name}/sklearn.TfidfVectorizer.fit", corpus_bytes, time_limit, filter_pattern, build_tfidf
        )

    if CumlTfidfVectorizer is not None:
        device_documents = cudf.Series(documents)

        def build_cuml_tfidf():
            CumlTfidfVectorizer().fit_transform(device_documents)
            return len(documents)

        bench_one(
            f"bm25-{slice_name}/cuml.TfidfVectorizer.fit<1gpu>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            build_cuml_tfidf,
        )


def bench_index_query(documents, needles, slice_name, corpus_bytes, time_limit, filter_pattern):
    """Scores the whole corpus against the slice, the same question the scan answers.

    The query is the vocabulary slice itself rather than a sampled document, so a wider dictionary is a
    wider query and the cell moves with the slice exactly as the scan's does.
    """
    query_text = " ".join(needles)

    if bm25s is not None:
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(documents, show_progress=False), show_progress=False)
        query_tokens = bm25s.tokenize([query_text], show_progress=False)

        def query_bm25s():
            results, _ = retriever.retrieve(query_tokens, k=min(10, len(documents)), show_progress=False)
            return int(results.size)

        bench_one(f"bm25-{slice_name}/bm25s.BM25.retrieve", corpus_bytes, time_limit, filter_pattern, query_bm25s)

    if rank_bm25 is not None:
        okapi = rank_bm25.BM25Okapi([document.split() for document in documents])
        query_terms = query_text.split()

        def query_rank_bm25():
            return int(okapi.get_scores(query_terms).size)

        bench_one(f"bm25-{slice_name}/rank_bm25.get_scores", corpus_bytes, time_limit, filter_pattern, query_rank_bm25)

    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer()
        document_matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query_text])

        def query_tfidf():
            return int((document_matrix @ query_vector.T).shape[0])

        bench_one(
            f"bm25-{slice_name}/sklearn.TfidfVectorizer.transform",
            corpus_bytes,
            time_limit,
            filter_pattern,
            query_tfidf,
        )

    if CumlTfidfVectorizer is not None:
        device_vectorizer = CumlTfidfVectorizer()
        device_matrix = device_vectorizer.fit_transform(cudf.Series(documents))
        device_query = device_vectorizer.transform(cudf.Series([query_text]))

        def query_cuml_tfidf():
            return int((device_matrix @ device_query.T).shape[0])

        bench_one(
            f"bm25-{slice_name}/cuml.TfidfVectorizer.transform<1gpu>",
            corpus_bytes,
            time_limit,
            filter_pattern,
            query_cuml_tfidf,
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

    for slice_name in VOCABULARY_SLICES:
        needles = needle_slice_of(vocabulary, slice_name)
        needle_bytes = sum(len(needle.encode()) for needle in needles)
        print(f"\n# {slice_name} - {len(needles):,} needles, {needle_bytes:,} bytes")

        bench_counting(documents, tape, needles, slice_name, corpus_bytes, args.time_limit, filter_pattern)
        bench_cover(documents, tape, needles, slice_name, corpus_bytes, args.time_limit, filter_pattern)
        bench_locating(documents, tape, needles, slice_name, corpus_bytes, args.time_limit, filter_pattern)
        bench_rewriting(tape, needles, slice_name, corpus_bytes, args.time_limit, filter_pattern)
        bench_scoring(documents, tape, needles, slice_name, corpus_bytes, args.time_limit, filter_pattern)
        if slice_name == VOCABULARY_SLICES[0]:
            bench_index_build(documents, slice_name, corpus_bytes, args.time_limit, filter_pattern)
        bench_index_query(documents, needles, slice_name, corpus_bytes, args.time_limit, filter_pattern)


if __name__ == "__main__":
    main()
