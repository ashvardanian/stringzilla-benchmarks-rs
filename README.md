# StringWa.rs

![StringWa.rs Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringWa.rs.jpg?raw=true)

_Not to pick a fight, but let there be String Wars!_ 😅
Jokes aside, many __great__ libraries for string processing exist.
_Mostly, of course, written in C and C++, but some in Rust as well._ 😅

Where Rust decimates C and C++, however, is the __simplicity__ of dependency management, making it great for benchmarking low-level software!
So, to accelerate the development of the [`stringzilla`](https://github.com/ashvardanian/StringZilla) C library, I've created this repository to compare it against:

- [`memchr`](https://github.com/BurntSushi/memchr) for substring search.
- [`rapidfuzz`](https://github.com/rapidfuzz/rapidfuzz-rs) for edit distances.
- [`aHash`](https://github.com/tkaitchuck/aHash) for hashing.
- [`aho_corasick`](https://github.com/BurntSushi/aho-corasick) for multi-pattern search.
- [`tantivy`](https://github.com/quickwit-oss/tantivy) for document retrieval.

Of course, the functionality of the projects is different, as are the APIs and the usage patterns.
So, I focus on the workloads for which StringZilla was designed and compare the throughput of the core operations.

## Substring Search Benchmarks 

Substring search is one of the most common operations in text processing, and one of the slowest.
StringZilla was designed to supersede LibC and implement those core operations in CPU-friendly manner, using branchless operations, SWAR, and SIMD assembly instructions.
Notably, Rust has a `memchr` crate that provides a similar functionality, and it's used in many popular libraries.
This repository provides basic benchmarking scripts for comparing the throughput of [`stringzilla`](https://github.com/ashvardanian/StringZilla) and [`memchr`](https://github.com/BurntSushi/memchr).
For normal order and reverse order search, over ASCII and UTF8 input data, the following numbers can be expected.

|               |         ASCII ⏩ |         ASCII ⏪ |         UTF8 ⏩ |          UTF8 ⏪ |
| ------------- | --------------: | --------------: | -------------: | --------------: |
| Intel:        |                 |                 |                |                 |
| `memchr`      |       5.89 GB/s |       1.08 GB/s |      8.73 GB/s |       3.35 GB/s |
| `stringzilla` |   __8.37__ GB/s |   __8.21__ GB/s | __11.21__ GB/s |  __11.20__ GB/s |
| Arm:          |                 |                 |                |                 |
| `memchr`      |       6.38 GB/s |       1.12 GB/s | __13.20__ GB/s |       3.56 GB/s |
| `stringzilla` |   __6.56__ GB/s |   __5.56__ GB/s |      9.41 GB/s |   __8.17__ GB/s |
|               |                 |                 |                |                 |
| Average       | __1.2x__ faster | __6.2x__ faster |              - | __2.8x__ faster |


> For Intel the benchmark was run on AWS `r7iz` instances with Sapphire Rapids cores.
> For Arm the benchmark was run on AWS `r7g` instances with Graviton 3 cores.
> The ⏩ signifies forward search, and ⏪ signifies reverse order search.
> At the time of writing, the latest versions of `memchr` and `stringzilla` were used - 2.7.1 and 3.3.0, respectively.

## Replicating the Results

Before running benchmarks, you can test your Rust environment running:

```bash
cargo install cargo-criterion --locked
```

Each benchmark includes a warm-up, to ensure that the CPU caches are filled and the results are not affected by cold start or SIMD-related frequency scaling.
To run them on Linux and MacOS, pass the dataset path as an environment variable:

- Substring Search:

    ```bash
    STRINGWARS_DATASET=README.md cargo criterion --features bench_find bench_find --jobs 8
    ```

    As part of the benchmark, the input "haystack" file is whitespace-tokenized into an array of strings.
    In every benchmark iteration, a new "needle" is taken from that array of tokens.
    All inclusions of that token in the haystack are counted, and the throughput is calculated.

- Edit Distance:

    ```bash
    STRINGWARS_MODE=lines STRINGWARS_ERROR_BOUND=15 STRINGWARS_DATASET=README.md cargo criterion --features bench_levenshtein bench_levenshtein --jobs 8
    STRINGWARS_MODE=words STRINGWARS_ERROR_BOUND=15 STRINGWARS_DATASET=README.md cargo criterion --features bench_levenshtein bench_levenshtein --jobs 8
    ```

    Edit distance benchmarks compute the Levenshtein distance between consecutive pairs of whitespace-delimited words or newline-delimited lines.
    They include byte-level and character-level operations and also run for the bounded case - when the maximum allowed distance is predefined.
    By default, the maximum allowed distance is set to 15% of the longer string in each pair.

- Hashing:

    ```bash
    STRINGWARS_MODE=file STRINGWARS_DATASET=README.md cargo criterion --features bench_hash bench_hash --jobs 8
    STRINGWARS_MODE=lines STRINGWARS_DATASET=README.md cargo criterion --features bench_hash bench_hash --jobs 8
    STRINGWARS_MODE=words STRINGWARS_DATASET=README.md cargo criterion --features bench_hash bench_hash --jobs 8
    ```

- Document retrieval with [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf):

    ```bash
    STRINGWARS_DATASET=README.md cargo criterion --features bench_tfidf bench_tfidf --jobs 8
    ```

    The TF-IDF benchmarks compute the term frequency-inverse document frequency for each word in the input file.
    The benchmark relies on a hybrid of StringZilla and SimSIMD to achieve the best performance.

On Windows using PowerShell you'd need to set the environment variable differently:

```powershell
$env:STRINGWARS_DATASET="README.md"
cargo criterion --jobs 8
```

## Datasets

### ASCII Corpus

For benchmarks on ASCII data I've used the English Leipzig Corpora Collection.
It's 124 MB in size, 1'000'000 lines long, and contains 8'388'608 tokens of mean length 5.

```bash
wget --no-clobber -O leipzig1M.txt https://introcs.cs.princeton.edu/python/42sort/leipzig1m.txt 
STRINGWARS_DATASET=leipzig1M.txt cargo criterion --jobs 8
```

### UTF8 Corpus

For richer mixed UTF data, I've used the XL Sum dataset for multilingual extractive summarization.
It's 4.7 GB in size (1.7 GB compressed), 1'004'598 lines long, and contains 268'435'456 tokens of mean length 8.
To download, unpack, and run the benchmarks, execute the following bash script in your terminal:

```bash
wget --no-clobber -O xlsum.csv.gz https://github.com/ashvardanian/xl-sum/releases/download/v1.0.0/xlsum.csv.gz
gzip -d xlsum.csv.gz
STRINGWARS_DATASET=xlsum.csv cargo criterion --jobs 8
```
