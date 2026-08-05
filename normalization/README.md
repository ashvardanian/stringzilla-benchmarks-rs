# Case Folding & Normalization Benchmarks

Benchmarks for Unicode case-insensitive operations and normalization — case folding, case-insensitive comparison and substring search, and NFC/NFD/NFKC/NFKD normalization — across different languages and hardware platforms.

## Case Folding

Measured on the full Leipzig corpora (`STRINGWARS_TOKENS=file`), single-threaded.
`Standard` is `std::to_lowercase` and `str.casefold()`, `StringZilla` is `utf8_uncased_fold`.

### Intel Xeon4 Sapphire Rapids

| Language     | Standard 🦀 | StringZilla 🦀 |      | Standard 🐍 | StringZilla 🐍 |      |
| ------------ | ---------: | ------------: | ---: | ---------: | ------------: | ---: |
| Arabic 🇸🇦     |    88 MB/s |     2.25 GB/s |  26x |   188 MB/s |      302 MB/s |   2x |
| Armenian 🇦🇲   |    72 MB/s |      373 MB/s |   5x |   209 MB/s |      180 MB/s |   1x |
| Bengali 🇧🇩    |    99 MB/s |     2.42 GB/s |  24x |   303 MB/s |      380 MB/s |   1x |
| Chinese 🇨🇳    |    96 MB/s |      414 MB/s |   4x |   239 MB/s |      190 MB/s |   1x |
| Czech 🇨🇿      |    96 MB/s |     1.41 GB/s |  15x |   125 MB/s |      228 MB/s |   2x |
| Dutch 🇳🇱      |   136 MB/s |     4.71 GB/s |  35x |   303 MB/s |      331 MB/s |   1x |
| English 🇬🇧    |   135 MB/s |     5.07 GB/s |  37x |   361 MB/s |      393 MB/s |   1x |
| Farsi 🇮🇷      |    84 MB/s |     1.15 GB/s |  14x |   219 MB/s |      271 MB/s |   1x |
| French 🇫🇷     |   125 MB/s |     1.71 GB/s |  14x |   122 MB/s |      238 MB/s |   2x |
| German 🇩🇪     |   133 MB/s |     2.05 GB/s |  15x |   124 MB/s |      281 MB/s |   2x |
| Greek 🇬🇷      |    69 MB/s |     1.26 GB/s |  18x |   194 MB/s |      275 MB/s |   1x |
| Hebrew 🇮🇱     |    74 MB/s |     2.37 GB/s |  32x |   180 MB/s |      260 MB/s |   1x |
| Hindi 🇮🇳      |    98 MB/s |     2.45 GB/s |  25x |   291 MB/s |      367 MB/s |   1x |
| Italian 🇮🇹    |   140 MB/s |     3.07 GB/s |  22x |   152 MB/s |      343 MB/s |   2x |
| Japanese 🇯🇵   |    97 MB/s |     1.17 GB/s |  12x |   242 MB/s |      267 MB/s |   1x |
| Korean 🇰🇷     |   149 MB/s |     2.24 GB/s |  15x |   241 MB/s |      286 MB/s |   1x |
| Polish 🇵🇱     |   117 MB/s |     1.12 GB/s |  10x |   110 MB/s |      195 MB/s |   2x |
| Portuguese 🇧🇷 |   133 MB/s |     2.30 GB/s |  17x |   114 MB/s |      265 MB/s |   2x |
| Russian 🇷🇺    |    69 MB/s |     1.31 GB/s |  19x |   199 MB/s |      288 MB/s |   1x |
| Spanish 🇪🇸    |   130 MB/s |     2.17 GB/s |  17x |   109 MB/s |      280 MB/s |   3x |
| Tamil 🇮🇳      |   113 MB/s |     2.40 GB/s |  21x |   319 MB/s |      394 MB/s |   1x |
| Turkish 🇹🇷    |   106 MB/s |     1.12 GB/s |  11x |   124 MB/s |      228 MB/s |   2x |
| Ukrainian 🇺🇦  |    69 MB/s |     1.23 GB/s |  18x |   203 MB/s |      283 MB/s |   1x |
| Vietnamese 🇻🇳 |    86 MB/s |     1.25 GB/s |  15x |   155 MB/s |      255 MB/s |   2x |

> Measured June 17, 2026.

### AMD Zen5 Turin

| Language     | Standard 🦀 | StringZilla 🦀 |      | Standard 🐍 | StringZilla 🐍 |      |
| ------------ | ---------: | ------------: | ---: | ---------: | ------------: | ---: |
| English 🇬🇧    |   482 MB/s |     7.53 GB/s |  16x |   257 MB/s |     3.14 GB/s |  12x |
| German 🇩🇪     |   432 MB/s |     2.59 GB/s |   6x |   260 MB/s |     1.81 GB/s |   7x |
| Russian 🇷🇺    |   217 MB/s |     2.20 GB/s |  10x |   470 MB/s |     1.56 GB/s |   3x |
| French 🇫🇷     |   346 MB/s |     1.84 GB/s |   5x |   274 MB/s |     1.37 GB/s |   5x |
| Greek 🇬🇷      |   220 MB/s |     1.00 GB/s |   5x |   431 MB/s |      779 MB/s |   2x |
| Armenian 🇦🇲   |   223 MB/s |      908 MB/s |   4x |   470 MB/s |      746 MB/s |   2x |
| Vietnamese 🇻🇳 |   265 MB/s |      352 MB/s |   1x |   340 MB/s |      291 MB/s |   1x |
| Arabic 🇸🇦     |   232 MB/s |     1004 MB/s |   4x |   467 MB/s |     1.80 GB/s |   4x |
| Bengali 🇧🇩    |   314 MB/s |     6.17 GB/s |  20x |   694 MB/s |     2.91 GB/s |   4x |
| Chinese 🇨🇳    |   325 MB/s |     1.21 GB/s |   4x |   697 MB/s |      886 MB/s |   1x |
| Czech 🇨🇿      |   322 MB/s |      827 MB/s |   3x |   292 MB/s |      688 MB/s |   2x |
| Dutch 🇳🇱      |   471 MB/s |     4.73 GB/s |  10x |   262 MB/s |     2.97 GB/s |  11x |
| Farsi 🇮🇷      |   235 MB/s |      858 MB/s |   4x |   475 MB/s |     1.42 GB/s |   3x |
| Georgian 🇬🇪   |   294 MB/s |      192 MB/s |   1x |   689 MB/s |      488 MB/s |   1x |
| Hebrew 🇮🇱     |   233 MB/s |     1.01 GB/s |   4x |   473 MB/s |     1.86 GB/s |   4x |
| Italian 🇮🇹    |   439 MB/s |     2.29 GB/s |   5x |   268 MB/s |     1.93 GB/s |   7x |
| Japanese 🇯🇵   |   330 MB/s |     3.51 GB/s |  11x |   726 MB/s |     2.00 GB/s |   3x |
| Korean 🇰🇷     |   314 MB/s |      861 MB/s |   3x |   623 MB/s |     2.80 GB/s |   4x |
| Lithuanian 🇱🇹 |   352 MB/s |      864 MB/s |   2x |   274 MB/s |      728 MB/s |   3x |
| Polish 🇵🇱     |   364 MB/s |      939 MB/s |   3x |   277 MB/s |      786 MB/s |   3x |
| Portuguese 🇧🇷 |   395 MB/s |     2.38 GB/s |   6x |   270 MB/s |     1.79 GB/s |   7x |
| Spanish 🇪🇸    |   414 MB/s |     2.38 GB/s |   6x |   272 MB/s |     1.80 GB/s |   7x |
| Tamil 🇮🇳      |   306 MB/s |     6.05 GB/s |  20x |   712 MB/s |     3.03 GB/s |   4x |
| Turkish 🇹🇷    |   326 MB/s |      852 MB/s |   3x |   284 MB/s |      706 MB/s |   2x |
| Ukrainian 🇺🇦  |   217 MB/s |     2.09 GB/s |  10x |   476 MB/s |     1.58 GB/s |   3x |


To rerun the benchmarks for all languages:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bench bench_normalization --features bench_normalization
bin=$(find target/release/deps -name 'bench_normalization-*' -executable -type f | head -1)

for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold" "$bin"
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-fold/" uv run normalization/bench.py
done
```

## Case-Insensitive Substring Search

### Intel Xeon4 Sapphire Rapids

| Language     | Standard 🦀 | StringZilla 🦀 |      | Standard 🐍 | StringZilla 🐍 |      |
| ------------ | ---------: | ------------: | ---: | ---------: | ------------: | ---: |
| Arabic 🇸🇦     |   103 MB/s |     7.24 GB/s |  70x |  3.01 GB/s |    14.78 GB/s |   5x |
| Armenian 🇦🇲   |   135 MB/s |      272 MB/s |   2x |  2.07 GB/s |      860 MB/s |   0x |
| Bengali 🇧🇩    |   191 MB/s |     6.97 GB/s |  37x |  4.51 GB/s |    21.19 GB/s |   5x |
| Chinese 🇨🇳    |   104 MB/s |     8.72 GB/s |  84x |  5.40 GB/s |    13.94 GB/s |   3x |
| Czech 🇨🇿      |    40 MB/s |     5.33 GB/s | 132x |  1.38 GB/s |     6.36 GB/s |   5x |
| Dutch 🇳🇱      |    41 MB/s |     4.33 GB/s | 107x |   860 MB/s |     7.99 GB/s |   9x |
| English 🇬🇧    |    43 MB/s |     4.91 GB/s | 115x |   770 MB/s |     5.61 GB/s |   7x |
| Farsi 🇮🇷      |   127 MB/s |     6.63 GB/s |  52x |  2.36 GB/s |    10.70 GB/s |   5x |
| French 🇫🇷     |    62 MB/s |     5.36 GB/s |  86x |  1.10 GB/s |     6.83 GB/s |   6x |
| Georgian 🇬🇪   |   190 MB/s |     1.03 GB/s |   5x |  3.20 GB/s |      620 MB/s |   0x |
| German 🇩🇪     |    47 MB/s |     4.47 GB/s |  95x |   900 MB/s |     6.08 GB/s |   7x |
| Greek 🇬🇷      |    56 MB/s |     1.66 GB/s |  30x |  1.38 GB/s |     2.48 GB/s |   2x |
| Hebrew 🇮🇱     |    77 MB/s |     6.86 GB/s |  89x |  2.92 GB/s |    15.72 GB/s |   5x |
| Italian 🇮🇹    |    62 MB/s |     5.03 GB/s |  81x |   970 MB/s |     8.87 GB/s |   9x |
| Japanese 🇯🇵   |   106 MB/s |     9.41 GB/s |  89x |  4.88 GB/s |    13.17 GB/s |   3x |
| Korean 🇰🇷     |   154 MB/s |     9.94 GB/s |  65x |  4.59 GB/s |    20.05 GB/s |   4x |
| Polish 🇵🇱     |    42 MB/s |     4.43 GB/s | 105x |  1.29 GB/s |     8.02 GB/s |   6x |
| Portuguese 🇧🇷 |    41 MB/s |     4.93 GB/s | 121x |  1.10 GB/s |     8.12 GB/s |   7x |
| Russian 🇷🇺    |    60 MB/s |     3.54 GB/s |  59x |  2.30 GB/s |     5.70 GB/s |   2x |
| Spanish 🇪🇸    |    64 MB/s |     4.88 GB/s |  76x |  1.02 GB/s |     6.33 GB/s |   6x |
| Tamil 🇮🇳      |   116 MB/s |     6.98 GB/s |  60x |  5.81 GB/s |    23.11 GB/s |   4x |
| Turkish 🇹🇷    |    62 MB/s |     4.12 GB/s |  66x |  1.49 GB/s |     5.25 GB/s |   4x |
| Ukrainian 🇺🇦  |    97 MB/s |     2.97 GB/s |  31x |  2.26 GB/s |     5.35 GB/s |   2x |
| Vietnamese 🇻🇳 |    76 MB/s |     5.06 GB/s |  67x |  1.07 GB/s |     1.12 GB/s |   1x |

> Measured June 17, 2026.

To rerun the benchmarks for all languages:

```bash
for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=words STRINGWARS_FILTER="case-insensitive-find" STRINGWARS_UNIQUE=1 "$bin"
done
```

---

See [README.md](../README.md) for dataset information and replication instructions.
