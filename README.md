# StringWars

## Text Processing on CPUs & GPUs, in Python & Rust

![StringWars Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringWars-v2.png?raw=true)

There are many __great__ libraries for string processing!
Mostly, of course, written in Assembly, C, and C++, but some in Rust as well.

Where Rust decimates C and C++, is the __simplicity__ of dependency management, making it great for benchmarking "Systems Software" and lining up apples-to-apples across native crates and their Python bindings.
So, to accelerate the development of the [`StringZilla`](https://github.com/ashvardanian/StringZilla) C, C++, and CUDA libraries (with Rust and Python bindings), I've created this repository to compare it against some of my & communities most beloved Rust projects, like:

- [`memchr`](https://github.com/BurntSushi/memchr) for substring search.
- [`rapidfuzz`](https://github.com/rapidfuzz/rapidfuzz-rs) and [`bio`](https://github.com/rust-bio/rust-bio) for edit distances and alignments.
- [`aHash`](https://github.com/tkaitchuck/aHash), [`xxhash-rust`](https://github.com/DoumanAsh/xxhash-rust), [`foldhash`](https://github.com/orlp/foldhash), and [`blake3`](https://github.com/BLAKE3-team/BLAKE3) for hashing.
- [`aho_corasick`](https://github.com/BurntSushi/aho-corasick) and [`regex`](https://github.com/rust-lang/regex) for multi-pattern search.
- [`arrow`](https://github.com/apache/arrow-rs) and [`polars`](https://github.com/pola-rs/polars) for collections and sorting.
- [`icu`](https://github.com/unicode-org/icu4x) for Unicode processing.
- [`ring`](https://github.com/briansmith/ring) and [`sodiumoxide`](https://github.com/sodiumoxide/sodiumoxide) for encryption.

Of course, the functionality of the projects is different, as are the APIs and the usage patterns.
So, I focus on the workloads for which StringZilla was designed and compare the throughput of the core operations.
Notably, I also favor modern hardware with support for a wider range SIMD instructions, like mask-equipped AVX-512 on x86 starting from the 2015 Intel Skylake-X CPUs or more recent predicated variable-length SVE and SVE2 on Arm, that aren't often supported by existing libraries and tooling.

> [!IMPORTANT]  
> The numbers in the tables below are provided for reference only and may vary depending on the CPU, compiler, dataset, and tokenization method.
> Most of them were obtained on Intel Sapphire Rapids __(SPR)__ and Granite Rapids __(GNR)__ CPUs and Nvidia Hopper-based __H100__ and Blackwell-based __RTX 6000__ Pro GPUs, using Rust with `-C target-cpu=native` optimization flag.
> To replicate the results, please refer to the [Replicating the Results](#replicating-the-results) section below.

## Benchmarks at a Glance

### Hash

Many hashing libraries exist, but they often lack reproducible outputs, streaming support, or cross-language availability.
Throughput on short words and long lines:

```
                    Short Words                  Long Lines
Rust:
stringzilla::hash   ████████████████████ 1.84    ████████████████████ 11.38 GB/s
aHash::hash_one     █████████████▍       1.23    ███████████████▏      8.61 GB/s
xxh3::xxh3_64       ███████████▊         1.08    ████████████████▋     9.48 GB/s
std::hash           ████▋                0.43    ██████▌               3.74 GB/s

Python:
stringzilla.hash    ████████████████████ 0.14    ████████████████████  9.19 GB/s
hash                ██████████████████▌  0.13    █████████▎            4.27 GB/s
xxhash.xxh3_64      █████▋               0.04    █████████████▉        6.38 GB/s
```

See [hash/README.md](hash/README.md) for details

### Multi-Way Hashing

Bloom and cuckoo filters need many independent hashes of the same key; StringZilla's `hash_multiseed` prepares the key once and replays cheap per-seed rounds, while the alternatives re-prepare it every 64–128 bits.
Digest throughput at a 1024-bit digest (16 independent hashes per word):

```
Rust:
stringzilla::hash_multiseed ████████████████████ 71.85 G bits/s
xxh3::xxh3_128              ████████▏            29.30 G bits/s
stringzilla::hash           ██████               21.77 G bits/s

Python:
stringzilla.hash_multiseed  ████████████████████ 6.48 G bits/s
stringzilla.hash            █▌                   0.51 G bits/s
xxhash.xxh3_128             ▉                    0.31 G bits/s
```

See [containers/README.md](containers/README.md) for details

### Case-Insensitive UTF-8 Search

Unicode-aware case-insensitive search with full case folding (ß↔SS, σ↔ς).
Throughput searching across ~100MB multilingual corpora:

```
Rust:
                      English                      German
stringzilla           ████████████████████ 12.79   ████████████████████ 10.67 GB/s
icu                   ▏                     0.08   ▏                     0.08 GB/s

                      Russian                      Korean
stringzilla           ████████████████████  7.12   ████████████████████ 35.10 GB/s
icu                   ▏                     0.14   ▏                     0.23 GB/s

Python:
                      English                      German
stringzilla           ████████████████████  5.61   ████████████████████  6.08 GB/s
regex                 ██▋                   0.77   ███                   0.90 GB/s

                      Russian                      Korean
stringzilla           ████████████████████  5.70   ████████████████████ 20.05 GB/s
regex                 ████████              2.30   ████▋                 4.59 GB/s
```

See [normalization/README.md](normalization/README.md) for details

### Exact Substring Search

Substring search is offloaded to C's `memmem` or `strstr` in most languages, but SIMD-optimized implementations can do better.
Throughput on long lines:

```
                    Left to right                Reverse order
Rust:
memmem::Finder      ████████████████████ 10.99
stringzilla         ███████████████████▋ 10.82   ████████████████████ 10.66 GB/s
std::str            ███████████████████▊ 10.88   ███████████▏          5.94 GB/s

Python:
stringzilla         ████████████████████ 11.79   ████████████████████ 11.56 GB/s
str                 ██                    1.23   ██████▋               3.84 GB/s
```

See [find/README.md](find/README.md) for details

### Byte-Set Search

Searching for character sets (tabs, HTML markup, digits) commonly uses regex or Aho-Corasick automata.
Throughput counting all matches on long lines:

```
Rust:
stringzilla         ████████████████████   8.17 GB/s
regex::find_iter    ████████████▊          5.22 GB/s
aho_corasick        █▏                     0.50 GB/s

Python:
stringzilla         ████████████████████   8.79 GB/s
re.finditer         ▍                      0.19 GB/s
```

See [find/README.md](find/README.md) for details

### UTF-8 Processing

Different scripts stress UTF-8 differently: Korean has 3-byte Hangul with single-byte whitespace (representative for tokenization), Arabic uses 2-byte characters, English is mostly 1-byte ASCII.
Throughput on AMD Zen5 Turin:

```
Newline splitting:
                      English                     Arabic
stringzilla           ████████████████ 15.45      ████████████████████ 18.34 GB/s
stdlib                ██                1.90      ██                    1.82 GB/s

Whitespace splitting:
                      English                     Korean
stringzilla           ████████████████████ 0.82   ████████████████████ 1.88 GB/s
stdlib                ██████████████████▊  0.77   ██████████▍          0.98 GB/s
icu::WhiteSpace       ██▋                  0.11   █▌                   0.15 GB/s
```

Case folding on bicameral scripts (Latin, Cyrillic, Greek, Armenian) plus Chinese for reference:

```
Case folding:
                      English 16x                 German 6x
stringzilla           ████████████████████ 7.53   ████████████████████ 2.59 GB/s
stdlib                ██▌                  0.48   ███▎                 0.43 GB/s

                      Russian 10x                 French 5x
stringzilla           ████████████████████ 2.20   ████████████████████ 1.84 GB/s
stdlib                ██                   0.22   ███▊                 0.35 GB/s

                      Greek 5x                    Armenian 4x
stringzilla           ████████████████████ 1.00   ████████████████████  908 MB/s
stdlib                ████▍                0.22   ████▉                 223 MB/s

                      Vietnamese 1.3x             Chinese 4x
stringzilla           ████████████████████  352   ████████████████████ 1.21 GB/s
stdlib                █████████████▏        265   █████▍                325 MB/s
```

Codepoint indexing — the byte offset of the Nth codepoint — is where SIMD pays off most: the standard library rescans the buffer byte-by-byte, while StringZilla jumps to it with a single SIMD scan.

```
Byte offset of the Nth codepoint, full corpora:
                      English                      Korean
stringzilla::find_nth ████████████████████  7.14   ████████████████████ 12.14 GB/s
std::char_indices     ██▎                   0.83   █▏                   0.72 GB/s
```

See [tokenization/README.md](tokenization/README.md) and [normalization/README.md](normalization/README.md) for details

### Sequence Operations

Dataframe libraries and search engines rely heavily on string sorting.
SIMD-accelerated comparisons and specialized radix sorts can outperform generic algorithms.
Every competitor is configured for a stable sort to match StringZilla, whose argsort writes the permutation into a caller-owned buffer — a NumPy `out=` array in Python — for a zero-allocation index sort.
Throughput on short words:

```
Rust:
stringzilla.argsort ████████████████████  209.32 M cmp/s
polars::sort        ███████████████████▋  205.21 M cmp/s
arrow::lexsort      ███████████▊          122.58 M cmp/s
std::sort_by_key    ███▌                   37.46 M cmp/s

Python:
polars.sort         ████████████████████  229.90 M cmp/s
stringzilla.argsort ███████████████████▏  219.92 M cmp/s
pyarrow.sort        ██████▎                72.99 M cmp/s
list.sort           ████▍                  51.10 M cmp/s
```

GPU: `cudf` on H100 reaches __9,463 M cmp/s__ on short words.

See [sequence/README.md](sequence/README.md) for details

### Random Generation

Random byte generation and lookup tables are common in image processing and bioinformatics.
Throughput on long lines:

```
Rust:
stringzilla         ████████████████████  10.57 GB/s
zeroize             ████████▉              4.73 GB/s
rand_xoshiro        ███████▎               3.85 GB/s

Python:
stringzilla         ████████████████████  20.37 GB/s
pycryptodome        ████████████▉         13.16 GB/s
numpy.Philox        █▌                     1.59 GB/s
```

See [memory/README.md](memory/README.md) for details

### Similarity Scoring

Edit distance is essential for search engines, data cleaning, NLP, and bioinformatics.
It's computationally expensive with O(n\*m) complexity, but GPUs and multi-core parallelism help.
Levenshtein distance on ~1,000 byte lines (MCUPS = Million Cell Updates Per Second):

```
Rust:
                        1 Core                       1 Socket
bio::levenshtein        █▏                      823
rapidfuzz               ████████████████████ 14,316
stringzilla<384x GNR>   ██████████████████▎  13,084  ████████████████████ 3,084,270 MCUPS
stringzilla<B200>                                    ██████▍                998,620 MCUPS
stringzilla<H100>                                    ██████                 925,890 MCUPS
```

See [similarities/README.md](similarities/README.md) for details

### Fingerprinting

Converting variable-length strings into fixed-length sketches (like Min-Hashing) enables fast approximate matching in large-scale retrieval.
Throughput on ~1,000 byte lines:

```
Rust:
                        1 Core                       1 Socket
pc::MinHash             ████████████████████   3.16
stringzilla<384xGNR>    ███▏                   0.51  ███████████████▍      302.30 MB/s
stringzilla<H100>                                    ████████████████████  392.37 MB/s
```

See [fingerprints/README.md](fingerprints/README.md) for details

### Encryption

ChaCha20 and AES256 encryption throughput comparison on long lines:

```
Rust:
ring::aes256        ████████████████████   2.89 GB/s
ring::chacha20      ████████▏              1.19 GB/s
libsodium::chacha20 █████                  0.71 GB/s
```

See [encryption/README.md](encryption/README.md) for details

## Replicating the Results

### Replicating the Results in Rust

To pull and compile all the dependencies, you can call:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --benches --all-features                  # to compile everything
RUSTFLAGS="-C target-cpu=native" cargo check --benches --all-features --all-targets    # to fail on warnings
```

By default StringWars links `stringzilla` in CPU mode.
If the machine has an NVIDIA GPU with CUDA installed, enable the CUDA kernels explicitly when running benches, for example:

```bash
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    STRINGWARS_FILTER=GPU \
    cargo bench --features "cuda bench_similarities" --bench bench_similarities --jobs 1
```

Wars always take long, and so do these benchmarks.
Every one of them includes a few seconds of a warm-up phase to ensure that the CPU caches are filled and the results are not affected by cold start or SIMD-related frequency scaling.
Each of them accepts a few environment variables to control the dataset, the tokenization, and the error bounds.
You can log those by printing file-level documentation using `awk` on Linux:

```bash
awk '/^\/\/!/ { print } !/^\/\/!/ { exit }' find/bench.rs
```

Commonly used environment variables are:

- `STRINGWARS_DATASET` - the path to the textual dataset file.
- `STRINGWARS_TOKENS` - the tokenization mode: `file`, `lines`, or `words`.
- `STRINGWARS_ERROR_BOUND` - the maximum allowed error in the Levenshtein distance.

Here is an example of a common benchmark run on a Unix-like system:

```bash
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo bench --features bench_hash --bench bench_hash --jobs $(nproc)
```

On Windows using PowerShell you'd need to set the environment variable differently:

```powershell
$env:STRINGWARS_DATASET="README.md"
cargo bench --features bench_hash --bench bench_hash --jobs $(nproc)
```

### Replicating the Results in Python

It's recommended to use `uv` for Python dependency management and running the benchmarks.
To install all dependencies for all benchmarks:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-cuda.txt
uv pip install --only-binary=:all: -r requirements.txt -r requirements-cuda.txt
```

To install dependencies for individual benchmarks:

```sh
PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com \
uv pip install '.[find,hash,memory,sequence,fingerprints,similarities,tokenization,normalization,containers,encryption]'
```

To run individual benchmarks, you can call:

```sh
uv run --no-project python find/bench.py --help
uv run --no-project python hash/bench.py --help
uv run --no-project python memory/bench.py --help
uv run --no-project python sequence/bench.py --help
uv run --no-project python similarities/bench.py --help
uv run --no-project python fingerprints/bench.py --help
uv run --no-project python tokenization/bench.py --help
uv run --no-project python normalization/bench.py --help
```

### Running Without Cloning

The Python benchmarks are self-contained [PEP 723](https://peps.python.org/pep-0723/) scripts, so `uv` can fetch a script and resolve its dependencies straight from a URL — no clone, no manual `pip install`:

```sh
uv run https://raw.githubusercontent.com/ashvardanian/StringWars/main/tokenization/bench.py \
    --dataset README.md --tokens file
```

The Rust benchmarks are a Cargo workspace with a path dependency on StringZilla, so there is no exact zero-clone equivalent — `cargo install --git` only installs `[[bin]]`/`[[example]]` targets, not `[[bench]]`.
The lightest path is a shallow clone:

```sh
git clone --depth 1 https://github.com/ashvardanian/StringWars && cd StringWars
RUSTFLAGS="-C target-cpu=native" cargo bench --features bench_hash --bench bench_hash --jobs $(nproc)
```

## Datasets

### UTF8 Corpus

For mixed UTF data, I've used the XL Sum dataset for multilingual extractive summarization.
It's 4.7 GB in size (1.7 GB compressed), 1'004'598 lines long, and contains 268'435'456 tokens of mean length 8.
To download, unpack, and run the benchmarks, execute the following bash script in your terminal:

```bash
mkdir -p data/xlsum && curl -fL -o data/xlsum/xlsum.csv.gz https://github.com/ashvardanian/xl-sum/releases/download/v1.0.0/xlsum.csv.gz
gzip -d data/xlsum/xlsum.csv.gz
STRINGWARS_DATASET=data/xlsum/xlsum.csv cargo bench --features bench_hash --bench bench_hash --jobs $(nproc)
```

Alternatively, for a much smaller and faster run, check out the Big List of Naughty Strings (BLNS):

```bash
mkdir -p data/blns && curl -fL -o data/blns/blns.txt https://raw.githubusercontent.com/minimaxir/big-list-of-naughty-strings/master/blns.txt
STRINGWARS_DATASET=data/blns/blns.txt cargo bench --features bench_hash --bench bench_hash --jobs $(nproc)
```

### Multilingual Wikipedia Corpus

The Cohere Wikipedia dataset provides pre-processed JSONL files for different languages.
This may be the optimal dataset for relative comparison of UTF-8 decoding and matching enginges in each individual environment.
Not all Wikipedia languages are available, but the following have been selected specifically:

- __Chinese (zh)__: 3-byte CJK characters, rare 1-byte punctuation
- __Korean (ko)__: 3-byte Hangul syllables, frequent 1-byte punctuation
- __Arabic (ar)__: 2-byte Arabic script, with regular 1-byte punctuation
- __French (fr)__: Mixed 1-2 byte Latin with high diacritic density
- __English (en)__: Mostly 1-byte ASCII baseline

To download and decompress one file from each language:

```bash
mkdir -p data/wikipedia-22-12
curl -fL -o data/wikipedia-22-12/wiki_en.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/en/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_en.jsonl.gz
curl -fL -o data/wikipedia-22-12/wiki_zh.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/zh/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_zh.jsonl.gz
curl -fL -o data/wikipedia-22-12/wiki_ko.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/ko/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_ko.jsonl.gz
curl -fL -o data/wikipedia-22-12/wiki_ar.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/ar/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_ar.jsonl.gz
curl -fL -o data/wikipedia-22-12/wiki_fr.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/fr/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_fr.jsonl.gz
curl -fL -o data/wikipedia-22-12/wiki_de.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/de/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_de.jsonl.gz
curl -fL -o data/wikipedia-22-12/wiki_es.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/es/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_es.jsonl.gz
curl -fL -o data/wikipedia-22-12/wiki_it.jsonl.gz https://huggingface.co/datasets/Cohere/wikipedia-22-12/resolve/main/it/000.jsonl.gz && gunzip data/wikipedia-22-12/wiki_it.jsonl.gz
```

Each JSONL file contains one JSON object per line with fields: `id`, `title`, `text` (paragraph content), `url`, `wiki_id`, and `paragraph_id`.

### CC-100 Corpus

The [CC-100](https://data.statmt.org/cc-100/) corpus provides large monolingual text files (1-80 GB) for 100+ languages, extracted from Common Crawl.
Files are XZ-compressed plain text with documents separated by double-newlines.

| Workload                    | Relevant Scripts                  | Best Test Languages                                  |
| --------------------------- | --------------------------------- | ---------------------------------------------------- |
| __Case Folding__            | Latin, Cyrillic, Greek, Armenian  | Turkish (I/i), German (ss->SS), Greek, Russian       |
| __Normalization__           | Indic, Arabic, Vietnamese, Korean | Vietnamese, Hindi, Korean, Arabic                    |
| __Whitespace Tokenization__ | Most scripts except CJK/Thai      | English, Russian, Arabic vs. Chinese, Japanese, Thai |
| __Grapheme Clusters__       | Indic, Thai, Khmer, Myanmar       | Thai, Tamil, Myanmar, Khmer                          |
| __RTL Handling__            | Arabic, Hebrew                    | Arabic, Hebrew, Persian                              |

__Bicameral scripts__ with various case folding rules:

```bash
mkdir -p data/cc-100
curl -fL https://data.statmt.org/cc-100/en.txt.xz | xz -d > data/cc-100/cc100_en.txt      # 82 GB - English
curl -fL https://data.statmt.org/cc-100/de.txt.xz | xz -d > data/cc-100/cc100_de.txt      # 18 GB - German
curl -fL https://data.statmt.org/cc-100/tr.txt.xz | xz -d > data/cc-100/cc100_tr.txt      # 5.4 GB - Turkish
curl -fL https://data.statmt.org/cc-100/ru.txt.xz | xz -d > data/cc-100/cc100_ru.txt      # 46 GB - Russian
curl -fL https://data.statmt.org/cc-100/uk.txt.xz | xz -d > data/cc-100/cc100_uk.txt      # 14 GB - Ukrainian
curl -fL https://data.statmt.org/cc-100/el.txt.xz | xz -d > data/cc-100/cc100_el.txt      # 7.4 GB - Greek
curl -fL https://data.statmt.org/cc-100/hy.txt.xz | xz -d > data/cc-100/cc100_hy.txt      # 776 MB - Armenian
curl -fL https://data.statmt.org/cc-100/ka.txt.xz | xz -d > data/cc-100/cc100_ka.txt      # 1.1 GB - Georgian
curl -fL https://data.statmt.org/cc-100/pl.txt.xz | xz -d > data/cc-100/cc100_pl.txt      # 12 GB - Polish
curl -fL https://data.statmt.org/cc-100/cs.txt.xz | xz -d > data/cc-100/cc100_cs.txt      # 4.4 GB - Czech
curl -fL https://data.statmt.org/cc-100/nl.txt.xz | xz -d > data/cc-100/cc100_nl.txt      # 7.9 GB - Dutch
curl -fL https://data.statmt.org/cc-100/fr.txt.xz | xz -d > data/cc-100/cc100_fr.txt      # 14 GB - French
curl -fL https://data.statmt.org/cc-100/es.txt.xz | xz -d > data/cc-100/cc100_es.txt      # 14 GB - Spanish
curl -fL https://data.statmt.org/cc-100/pt.txt.xz | xz -d > data/cc-100/cc100_pt.txt      # 13 GB - Portuguese
curl -fL https://data.statmt.org/cc-100/it.txt.xz | xz -d > data/cc-100/cc100_it.txt      # 7.8 GB - Italian
```

__Unicameral scripts__ without case folding, but with other normalization/segmentation challenges:

```bash
mkdir -p data/cc-100
curl -fL https://data.statmt.org/cc-100/ar.txt.xz | xz -d > data/cc-100/cc100_ar.txt      # 5.4 GB - Arabic (RTL)
curl -fL https://data.statmt.org/cc-100/he.txt.xz | xz -d > data/cc-100/cc100_he.txt      # 6.1 GB - Hebrew (RTL)
curl -fL https://data.statmt.org/cc-100/fa.txt.xz | xz -d > data/cc-100/cc100_fa.txt      # 20 GB - Persian (RTL)
curl -fL https://data.statmt.org/cc-100/hi.txt.xz | xz -d > data/cc-100/cc100_hi.txt      # 2.5 GB - Hindi (Devanagari)
curl -fL https://data.statmt.org/cc-100/bn.txt.xz | xz -d > data/cc-100/cc100_bn.txt      # 860 MB - Bengali
curl -fL https://data.statmt.org/cc-100/ta.txt.xz | xz -d > data/cc-100/cc100_ta.txt      # 1.3 GB - Tamil
curl -fL https://data.statmt.org/cc-100/te.txt.xz | xz -d > data/cc-100/cc100_te.txt      # 536 MB - Telugu
curl -fL https://data.statmt.org/cc-100/th.txt.xz | xz -d > data/cc-100/cc100_th.txt      # 8.7 GB - Thai (no spaces)
curl -fL https://data.statmt.org/cc-100/vi.txt.xz | xz -d > data/cc-100/cc100_vi.txt      # 28 GB - Vietnamese
curl -fL https://data.statmt.org/cc-100/zh-Hans.txt.xz | xz -d > data/cc-100/cc100_zh.txt # 14 GB - Chinese
curl -fL https://data.statmt.org/cc-100/ja.txt.xz | xz -d > data/cc-100/cc100_ja.txt      # 15 GB - Japanese
curl -fL https://data.statmt.org/cc-100/ko.txt.xz | xz -d > data/cc-100/cc100_ko.txt      # 14 GB - Korean (Jamo)
curl -fL https://data.statmt.org/cc-100/my.txt.xz | xz -d > data/cc-100/cc100_my.txt      # 46 MB - Myanmar
curl -fL https://data.statmt.org/cc-100/km.txt.xz | xz -d > data/cc-100/cc100_km.txt      # 153 MB - Khmer
curl -fL https://data.statmt.org/cc-100/am.txt.xz | xz -d > data/cc-100/cc100_am.txt      # 133 MB - Amharic (Ethiopic)
curl -fL https://data.statmt.org/cc-100/si.txt.xz | xz -d > data/cc-100/cc100_si.txt      # 452 MB - Sinhala
```

### Leipzig Corpora Collection

The [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/) provides pre-segmented sentences in 200+ languages.
Each tar.gz contains `*-sentences.txt` (tab-separated `id\tsentence`), `*-words.txt` (frequencies), and co-occurrence files.
Standard sizes: 10K, 30K, 100K, 300K, 1M sentences. Check for newer years at the download page.

__Bicameral scripts__ with various case folding rules:

```bash
mkdir -p data/leipzig-1m data/leipzig-300k
curl -fL https://downloads.wortschatz-leipzig.de/corpora/eng_wikipedia_2016_1M.tar.gz | tar -xzf - -O 'eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_en.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/deu_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'deu_wikipedia_2021_1M/deu_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_de.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tur_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'tur_wikipedia_2021_1M/tur_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_tr.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/rus_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'rus_wikipedia_2021_1M/rus_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_ru.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ukr_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ukr_wikipedia_2021_1M/ukr_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_uk.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ell_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ell_wikipedia_2021_1M/ell_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_el.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/hye_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'hye_wikipedia_2021_1M/hye_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_hy.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/kat_wikipedia_2021_300K.tar.gz | tar -xzf - -O 'kat_wikipedia_2021_300K/kat_wikipedia_2021_300K-sentences.txt' | cut -f2 > data/leipzig-300k/leipzig300K_ka.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/pol_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'pol_wikipedia_2021_1M/pol_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_pl.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ces_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ces_wikipedia_2021_1M/ces_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_cs.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/nld_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'nld_wikipedia_2021_1M/nld_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_nl.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/fra_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'fra_wikipedia_2021_1M/fra_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_fr.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/spa_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'spa_wikipedia_2021_1M/spa_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_es.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/por_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'por_wikipedia_2021_1M/por_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_pt.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ita_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ita_wikipedia_2021_1M/ita_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_it.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/lit_wikipedia_2021_300K.tar.gz | tar -xzf - -O 'lit_wikipedia_2021_300K/lit_wikipedia_2021_300K-sentences.txt' | cut -f2 > data/leipzig-300k/leipzig300K_lt.txt
```

__Unicameral scripts__ without case folding, but with other normalization/segmentation challenges:

```bash
mkdir -p data/leipzig-1m data/leipzig-300k data/leipzig-30k data/leipzig-10k
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ara_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ara_wikipedia_2021_1M/ara_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_ar.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/heb_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'heb_wikipedia_2021_1M/heb_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_he.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/fas_wikipedia_2014_1M.tar.gz | tar -xzf - -O 'fas_wikipedia_2014_1M/fas_wikipedia_2014_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_fa.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/hin_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'hin_wikipedia_2021_1M/hin_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_hi.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/ben_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'ben_wikipedia_2021_1M/ben_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_bn.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tam_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'tam_wikipedia_2021_1M/tam_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_ta.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tel_wikipedia_2021_300K.tar.gz | tar -xzf - -O 'tel_wikipedia_2021_300K/tel_wikipedia_2021_300K-sentences.txt' | cut -f2 > data/leipzig-300k/leipzig300K_te.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/tha_wikipedia_2021_10K.tar.gz | tar -xzf - -O 'tha_wikipedia_2021_10K/tha_wikipedia_2021_10K-sentences.txt' | cut -f2 > data/leipzig-10k/leipzig10K_th.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/vie_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'vie_wikipedia_2021_1M/vie_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_vi.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/zho_wikipedia_2018_1M.tar.gz | tar -xzf - -O 'zho_wikipedia_2018_1M/zho_wikipedia_2018_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_zh.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/jpn_wikipedia_2018_1M.tar.gz | tar -xzf - -O 'jpn_wikipedia_2018_1M/jpn_wikipedia_2018_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_ja.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/kor_wikipedia_2021_1M.tar.gz | tar -xzf - -O 'kor_wikipedia_2021_1M/kor_wikipedia_2021_1M-sentences.txt' | cut -f2 > data/leipzig-1m/leipzig1M_ko.txt
curl -fL https://downloads.wortschatz-leipzig.de/corpora/amh_wikipedia_2021_30K.tar.gz | tar -xzf - -O 'amh_wikipedia_2021_30K/amh_wikipedia_2021_30K-sentences.txt' | cut -f2 > data/leipzig-30k/leipzig30K_am.txt
```

To produce a mixed dataset with rows in all languages:

```bash
mkdir -p data/leipzig-1gb
cat data/leipzig-*/*.txt | shuf | head -c 1G > data/leipzig-1gb/leipzig1GB.txt
```

### DNA Corpus

For bioinformatics workloads, I use the following datasets with increasing string lengths:

```bash
mkdir -p data/acgt
curl -fL -o data/acgt/acgt_100.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_100.txt?download=true'
curl -fL -o data/acgt/acgt_1k.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_1k.txt?download=true'
curl -fL -o data/acgt/acgt_10k.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_10k.txt?download=true'
curl -fL -o data/acgt/acgt_100k.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_100k.txt?download=true'
curl -fL -o data/acgt/acgt_1m.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_1m.txt?download=true'
curl -fL -o data/acgt/acgt_10m.txt 'https://huggingface.co/datasets/ashvardanian/StringWars/resolve/main/acgt_10m.txt?download=true'
```

### Unicode Testing Data

The Unicode Project comes with many adversarial string examples for case folding, grapheme segmentation, word, and sentence breaks.
The following script, downloads all of them and composes them ino a single randomly shuffled document:

```bash
source_urls=(
    https://www.unicode.org/Public/UCD/latest/ucd/NormalizationTest.txt
    https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt
    https://www.unicode.org/Public/UCD/latest/ucd/SpecialCasing.txt
    https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/GraphemeBreakTest.txt
    https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/WordBreakTest.txt
    https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/SentenceBreakTest.txt
    https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/LineBreakTest.txt
    https://www.unicode.org/Public/emoji/latest/emoji-test.txt
    https://www.unicode.org/Public/security/latest/confusables.txt
)

for url in "${source_urls[@]}"; do
    curl -fL --create-dirs -o "unicode/${url##*/}" "$url"
done

# Concatenate every file and shuffle all lines into one adversarial document. The shuffle is
# reproducible: a seeded AES-CTR keystream feeds `shuf --random-source`, so the same STRINGWARS_SEED
# always yields the same ordering (default 42, matching the benchmark harnesses).
seed="${STRINGWARS_SEED:-42}"
seeded_random() { openssl enc -aes-256-ctr -pass "pass:$1" -nosalt </dev/zero 2>/dev/null; }
cat unicode/*.txt | shuf --random-source=<(seeded_random "$seed") > unicode_tests.txt
```

## Deep Profiling

In case you are profiling the some of the internal kernels of mentioned libraries, here are a few example commands to get around.
Such as using `ncu` for NVIDIA GPUs to evaluate the register usage and occupancy of the CUDA kernels used in StringZilla's Levenshtein distance calculation:

```bash
/usr/local/cuda/bin/ncu \
  --metrics launch__registers_per_thread,launch__occupancy_per_block_size,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum \
  --target-processes all \
  --kernel-name "levenshtein_on_each_cuda_thread" \
  --launch-skip 5 \
  --launch-count 1 \
  bash -c 'STRINGWARS_DATASET=data/acgt/acgt_100.txt STRINGWARS_BATCH_PER_CORE=65536 STRINGWARS_TOKENS=lines STRINGWARS_FILTER="uniform/stringzillas::LevenshteinDistances\(1xGPU\)" cargo bench --features "cuda bench_similarities" --bench bench_similarities --jobs 1'
```

Using `perf` on Linux to analyze the CPU-side performance of SIMD-accelerated substring search:

```bash
perf record -e cpu-clock -g graph,0x400000 -o perf.data -- cargo bench --features "bench_similarities" --bench bench_similarities --jobs 1
perf report -i perf.data
```

## Contributing

After cloning, enable the repository's git hooks once:

```bash
git config core.hooksPath .githooks
```

The `pre-commit` hook rejects banner marks (three or more consecutive `-` or `=`) in
Python and Rust sources. Use plain text for section titles, and `# region:` /
`# endregion:` at module scope for foldable sections.

It also runs `ruff check` and `ruff format --check` on staged Python (the project
requires Python 3.13). Fix issues with `ruff check --fix . && ruff format .` rather
than silencing them with `# noqa`.
