# Case Folding & Normalization Benchmarks

Benchmarks for Unicode case-insensitive operations and normalization — case folding, case-insensitive comparison and substring search, and NFC/NFD/NFKC/NFKD normalization — across different languages and hardware platforms.

## Case Folding

Measured single-threaded on a 512 KB prefix of each Leipzig corpus, as one `file`-mode token.
The budget is set by `pyunormalize`, which sustains ~53 MChar/s to 800 K characters and then
collapses by more than four orders of magnitude.
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
| Georgian 🇬🇪   |          — |             — |    — |          — |             — |    — |
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
| Hindi 🇮🇳      |          — |             — |    — |          — |             — |    — |
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

### Apple M5 Pro

| Language     | Standard 🦀 | StringZilla 🦀 |       | Standard 🐍 | PyICU 🐍   | StringZilla 🐍 |       |
| ------------ | ---------: | ------------: | ----: | ---------: | --------: | ------------: | ----: |
| Arabic 🇸🇦     |   319 MB/s |     2.36 GB/s |  7.4x |   915 MB/s |  455 MB/s |     2.33 GB/s |  2.5x |
| Armenian 🇦🇲   |   309 MB/s |      946 MB/s |  3.1x |   933 MB/s |  447 MB/s |      913 MB/s |  1.0x |
| Bengali 🇧🇩    |   441 MB/s |     6.75 GB/s | 15.3x |  1.49 GB/s | 1.42 GB/s |     6.67 GB/s |  4.5x |
| Chinese 🇨🇳    |   462 MB/s |      540 MB/s |  1.2x |  1.37 GB/s |  700 MB/s |      601 MB/s |  0.4x |
| Czech 🇨🇿      |   334 MB/s |     1.40 GB/s |  4.2x |   496 MB/s |  561 MB/s |     1.36 GB/s |  2.7x |
| Dutch 🇳🇱      |   709 MB/s |    15.72 GB/s | 22.2x |   497 MB/s |  501 MB/s |    15.26 GB/s | 30.7x |
| English 🇬🇧    |   713 MB/s |    15.98 GB/s | 22.4x |   561 MB/s |  447 MB/s |    15.95 GB/s | 28.4x |
| Farsi 🇮🇷      |   309 MB/s |     2.65 GB/s |  8.6x |   787 MB/s |  533 MB/s |     2.71 GB/s |  3.4x |
| French 🇫🇷     |   549 MB/s |     1.43 GB/s |  2.6x |   466 MB/s |  509 MB/s |     1.38 GB/s |  3.0x |
| Georgian 🇬🇪   |   445 MB/s |      812 MB/s |  1.8x |  1.08 GB/s |  411 MB/s |      749 MB/s |  0.7x |
| German 🇩🇪     |   655 MB/s |     2.35 GB/s |  3.6x |   570 MB/s |  380 MB/s |     2.26 GB/s |  4.0x |
| Greek 🇬🇷      |   312 MB/s |     1.74 GB/s |  5.6x |   681 MB/s |  205 MB/s |     1.57 GB/s |  2.3x |
| Hebrew 🇮🇱     |   323 MB/s |     2.85 GB/s |  8.8x |   719 MB/s |  410 MB/s |     2.85 GB/s |  4.0x |
| Hindi 🇮🇳      |   419 MB/s |     7.33 GB/s | 17.5x |  1.15 GB/s | 1.26 GB/s |     7.24 GB/s |  6.3x |
| Italian 🇮🇹    |   686 MB/s |     4.18 GB/s |  6.1x |   564 MB/s |  503 MB/s |     4.07 GB/s |  7.2x |
| Japanese 🇯🇵   |   444 MB/s |     1.80 GB/s |  4.1x |  1.42 GB/s |  785 MB/s |     1.92 GB/s |  1.4x |
| Korean 🇰🇷     |   410 MB/s |     1.86 GB/s |  4.5x |  1.08 GB/s | 1.13 GB/s |     1.75 GB/s |  1.6x |
| Polish 🇵🇱     |   516 MB/s |     1.39 GB/s |  2.7x |   474 MB/s |  495 MB/s |     1.34 GB/s |  2.8x |
| Portuguese 🇧🇷 |   610 MB/s |     1.81 GB/s |  3.0x |   462 MB/s |  487 MB/s |     1.74 GB/s |  3.8x |
| Russian 🇷🇺    |   313 MB/s |     1.12 GB/s |  3.6x |   957 MB/s |  266 MB/s |     1.02 GB/s |  1.1x |
| Spanish 🇪🇸    |   641 MB/s |     2.05 GB/s |  3.2x |   456 MB/s |  495 MB/s |     1.94 GB/s |  4.3x |
| Tamil 🇮🇳      |   453 MB/s |     6.27 GB/s | 13.8x |  1.10 GB/s |  744 MB/s |     6.24 GB/s |  5.7x |
| Turkish 🇹🇷    |   420 MB/s |     1.27 GB/s |  3.0x |   613 MB/s |  499 MB/s |     1.23 GB/s |  2.0x |
| Ukrainian 🇺🇦  |   307 MB/s |     1.13 GB/s |  3.7x |   758 MB/s |  303 MB/s |     1.04 GB/s |  1.4x |
| Vietnamese 🇻🇳 |   347 MB/s |     1.44 GB/s |  4.1x |   563 MB/s |  397 MB/s |     1.39 GB/s |  2.5x |

> Measured July 29, 2026.

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

| Language     | Standard 🦀 | StringZilla 🦀 |        | Standard 🐍 | StringZilla 🐍 |      |
| ------------ | ---------: | ------------: | -----: | ---------: | ------------: | ---: |
| Arabic 🇸🇦     |   103 MB/s |     7.24 GB/s |  70.3x |  3.01 GB/s |    14.78 GB/s | 4.9x |
| Armenian 🇦🇲   |   135 MB/s |      272 MB/s |   2.0x |  2.07 GB/s |      860 MB/s | 0.4x |
| Bengali 🇧🇩    |   191 MB/s |     6.97 GB/s |  36.5x |  4.51 GB/s |    21.19 GB/s | 4.7x |
| Chinese 🇨🇳    |   104 MB/s |     8.72 GB/s |  83.8x |  5.40 GB/s |    13.94 GB/s | 2.6x |
| Czech 🇨🇿      |    40 MB/s |     5.33 GB/s | 133.2x |  1.38 GB/s |     6.36 GB/s | 4.6x |
| Dutch 🇳🇱      |    41 MB/s |     4.33 GB/s | 105.6x |   860 MB/s |     7.99 GB/s | 9.3x |
| English 🇬🇧    |    43 MB/s |     4.91 GB/s | 114.2x |   770 MB/s |     5.61 GB/s | 7.3x |
| Farsi 🇮🇷      |   127 MB/s |     6.63 GB/s |  52.2x |  2.36 GB/s |    10.70 GB/s | 4.5x |
| French 🇫🇷     |    62 MB/s |     5.36 GB/s |  86.5x |  1.10 GB/s |     6.83 GB/s | 6.2x |
| Georgian 🇬🇪   |   190 MB/s |     1.03 GB/s |   5.4x |  3.20 GB/s |      620 MB/s | 0.2x |
| German 🇩🇪     |    47 MB/s |     4.47 GB/s |  95.1x |   900 MB/s |     6.08 GB/s | 6.8x |
| Greek 🇬🇷      |    56 MB/s |     1.66 GB/s |  29.6x |  1.38 GB/s |     2.48 GB/s | 1.8x |
| Hebrew 🇮🇱     |    77 MB/s |     6.86 GB/s |  89.1x |  2.92 GB/s |    15.72 GB/s | 5.4x |
| Hindi 🇮🇳      |          — |             — |      — |          — |             — |    — |
| Italian 🇮🇹    |    62 MB/s |     5.03 GB/s |  81.1x |   970 MB/s |     8.87 GB/s | 9.1x |
| Japanese 🇯🇵   |   106 MB/s |     9.41 GB/s |  88.8x |  4.88 GB/s |    13.17 GB/s | 2.7x |
| Korean 🇰🇷     |   154 MB/s |     9.94 GB/s |  64.5x |  4.59 GB/s |    20.05 GB/s | 4.4x |
| Polish 🇵🇱     |    42 MB/s |     4.43 GB/s | 105.5x |  1.29 GB/s |     8.02 GB/s | 6.2x |
| Portuguese 🇧🇷 |    41 MB/s |     4.93 GB/s | 120.2x |  1.10 GB/s |     8.12 GB/s | 7.4x |
| Russian 🇷🇺    |    60 MB/s |     3.54 GB/s |  59.0x |  2.30 GB/s |     5.70 GB/s | 2.5x |
| Spanish 🇪🇸    |    64 MB/s |     4.88 GB/s |  76.2x |  1.02 GB/s |     6.33 GB/s | 6.2x |
| Tamil 🇮🇳      |   116 MB/s |     6.98 GB/s |  60.2x |  5.81 GB/s |    23.11 GB/s | 4.0x |
| Turkish 🇹🇷    |    62 MB/s |     4.12 GB/s |  66.5x |  1.49 GB/s |     5.25 GB/s | 3.5x |
| Ukrainian 🇺🇦  |    97 MB/s |     2.97 GB/s |  30.6x |  2.26 GB/s |     5.35 GB/s | 2.4x |
| Vietnamese 🇻🇳 |    76 MB/s |     5.06 GB/s |  66.6x |  1.07 GB/s |     1.12 GB/s | 1.0x |

> Measured June 17, 2026. `Standard` is `pcre2::pre-jit` 🦀 / `regex.search<fullcase>` 🐍 — PCRE2/regex with full Unicode case folding, the fair baseline against StringZilla's own full case folding.

### Apple M5 Pro

| Language     | memchr+ICU 🦀 | pcre2 no-jit 🦀 | pcre2 jit-on-fly 🦀 | pcre2 pre-jit 🦀 | StringZilla 🦀 |         | casefold.find 🐍 | PyICU 🐍  | regex 🐍    | StringZilla 🐍 |      |
| ------------ | -----------: | -------------: | -----------------: | --------------: | ------------: | ------: | --------------: | -------: | ---------: | ------------: | ---: |
| Arabic 🇸🇦     |     296 MB/s |         9 MB/s |             9 MB/s |          9 MB/s |    11.21 GB/s | 1245.6x |        690 MB/s | 132 MB/s |  3.24 GB/s |    11.18 GB/s | 3.5x |
| Armenian 🇦🇲   |     304 MB/s |        52 MB/s |            63 MB/s |         62 MB/s |     1.15 GB/s |   18.5x |        700 MB/s | 178 MB/s |  2.76 GB/s |     1.08 GB/s | 0.4x |
| Bengali 🇧🇩    |     463 MB/s |        22 MB/s |            24 MB/s |         24 MB/s |    24.35 GB/s | 1014.6x |        957 MB/s | 235 MB/s |  5.21 GB/s |    29.92 GB/s | 5.7x |
| Chinese 🇨🇳    |     417 MB/s |        10 MB/s |            10 MB/s |         10 MB/s |    12.50 GB/s | 1250.0x |       1.19 GB/s | 141 MB/s |  5.54 GB/s |    14.90 GB/s | 2.7x |
| Czech 🇨🇿      |     167 MB/s |        24 MB/s |            25 MB/s |         25 MB/s |     4.37 GB/s |  174.8x |        408 MB/s | 102 MB/s |  1.68 GB/s |     2.90 GB/s | 1.7x |
| Dutch 🇳🇱      |     171 MB/s |         9 MB/s |             9 MB/s |          9 MB/s |     5.70 GB/s |  633.3x |        341 MB/s | 113 MB/s |  1.16 GB/s |     5.22 GB/s | 4.5x |
| English 🇬🇧    |     157 MB/s |        19 MB/s |            19 MB/s |         19 MB/s |     6.90 GB/s |  363.2x |        392 MB/s |  98 MB/s |  1.20 GB/s |     6.10 GB/s | 5.1x |
| Farsi 🇮🇷      |     333 MB/s |         8 MB/s |             8 MB/s |          8 MB/s |    22.51 GB/s | 2813.8x |        591 MB/s | 147 MB/s |  2.71 GB/s |    14.77 GB/s | 5.5x |
| French 🇫🇷     |     163 MB/s |        29 MB/s |            30 MB/s |         30 MB/s |     7.06 GB/s |  235.3x |        369 MB/s | 105 MB/s |  1.12 GB/s |     4.30 GB/s | 3.8x |
| Georgian 🇬🇪   |     301 MB/s |       104 MB/s |           137 MB/s |        137 MB/s |     3.56 GB/s |   26.0x |        899 MB/s | 269 MB/s |  3.31 GB/s |     3.03 GB/s | 0.9x |
| German 🇩🇪     |     163 MB/s |        15 MB/s |            16 MB/s |         16 MB/s |     7.52 GB/s |  470.0x |        417 MB/s | 103 MB/s |  1.02 GB/s |     5.71 GB/s | 5.6x |
| Greek 🇬🇷      |     204 MB/s |        42 MB/s |            48 MB/s |         47 MB/s |     1.73 GB/s |   36.8x |        554 MB/s | 151 MB/s |  2.51 GB/s |     2.03 GB/s | 0.8x |
| Hebrew 🇮🇱     |     280 MB/s |        24 MB/s |            26 MB/s |         27 MB/s |    16.93 GB/s |  627.0x |        626 MB/s | 171 MB/s |  3.04 GB/s |    10.84 GB/s | 3.6x |
| Hindi 🇮🇳      |     446 MB/s |         7 MB/s |             7 MB/s |          7 MB/s |    16.04 GB/s | 2291.4x |       1.07 GB/s | 271 MB/s |  4.92 GB/s |    13.52 GB/s | 2.7x |
| Italian 🇮🇹    |     162 MB/s |        43 MB/s |            44 MB/s |         45 MB/s |     5.62 GB/s |  124.9x |        439 MB/s | 109 MB/s |  1.12 GB/s |     6.49 GB/s | 5.8x |
| Japanese 🇯🇵   |     422 MB/s |        12 MB/s |            12 MB/s |         12 MB/s |    11.03 GB/s |  919.2x |       1.26 GB/s | 164 MB/s | 10.17 GB/s |     9.63 GB/s | 0.9x |
| Korean 🇰🇷     |     323 MB/s |        79 MB/s |            98 MB/s |         98 MB/s |    30.21 GB/s |  308.3x |        853 MB/s |  96 MB/s |  5.42 GB/s |    30.05 GB/s | 5.5x |
| Polish 🇵🇱     |     159 MB/s |        63 MB/s |            67 MB/s |         66 MB/s |     5.11 GB/s |   77.4x |        382 MB/s | 105 MB/s |  1.29 GB/s |     4.59 GB/s | 3.6x |
| Portuguese 🇧🇷 |     166 MB/s |        46 MB/s |            48 MB/s |         48 MB/s |     4.65 GB/s |   96.9x |        354 MB/s | 108 MB/s |  1.35 GB/s |     5.45 GB/s | 4.0x |
| Russian 🇷🇺    |     229 MB/s |        54 MB/s |            64 MB/s |         64 MB/s |     3.79 GB/s |   59.2x |        749 MB/s | 173 MB/s |  2.92 GB/s |     2.81 GB/s | 1.0x |
| Spanish 🇪🇸    |     161 MB/s |        20 MB/s |            21 MB/s |         21 MB/s |     5.30 GB/s |  252.4x |        357 MB/s | 105 MB/s |  1.15 GB/s |     4.91 GB/s | 4.3x |
| Tamil 🇮🇳      |     399 MB/s |        52 MB/s |            60 MB/s |         61 MB/s |    19.83 GB/s |  325.1x |       1.07 GB/s | 278 MB/s |  5.28 GB/s |    21.25 GB/s | 4.0x |
| Turkish 🇹🇷    |     155 MB/s |        14 MB/s |            14 MB/s |         14 MB/s |     2.74 GB/s |  195.7x |        392 MB/s | 111 MB/s |  1.65 GB/s |     2.99 GB/s | 1.8x |
| Ukrainian 🇺🇦  |     245 MB/s |        17 MB/s |            18 MB/s |         18 MB/s |     2.65 GB/s |  147.2x |        610 MB/s | 176 MB/s |  2.90 GB/s |     3.22 GB/s | 1.1x |
| Vietnamese 🇻🇳 |     185 MB/s |         4 MB/s |             4 MB/s |          4 MB/s |     1.99 GB/s |  497.5x |        443 MB/s |  94 MB/s |  1.54 GB/s |     1.94 GB/s | 1.3x |

> Measured July 29, 2026.

To rerun the benchmarks for all languages:

```bash
for f in leipzig*.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  STRINGWARS_DATASET="$f" STRINGWARS_TOKENS=file STRINGWARS_FILTER="case-insensitive-find" "$bin"
done
```

## Unicode Normalization

Normalizing each Leipzig corpus into NFC, single-threaded.
Rust compares `stringzilla::utf8_norm` against the `unicode-normalization` crate and ICU4X's `ComposingNormalizer`;
Python compares `stringzilla.utf8_norm` against `unicodedata.normalize`, PyICU's `Normalizer2`, and
[`pyunormalize`](https://github.com/mlodel/pyunormalize) — a pure-Python UAX #15 implementation that ships its own
Unicode tables, included as the reference for what the algorithm costs with no native code behind it.
NFD/NFKC/NFKD were measured too and follow the same ordering; only NFC is tabulated here.

### Apple M5 Pro

| Language     | unicode-norm 🦀 | ICU4X 🦀       | StringZilla 🦀  | unicodedata 🐍 | PyICU 🐍       | pyunormalize 🐍 | StringZilla 🐍  |
| ------------ | -------------: | ------------: | -------------: | ------------: | ------------: | -------------: | -------------: |
| Arabic 🇸🇦     |       160 MB/s |      506 MB/s |  __1.17 GB/s__ | __2.55 GB/s__ |     1.00 GB/s |              — |      1.21 GB/s |
| Armenian 🇦🇲   |       177 MB/s |      508 MB/s |  __1.12 GB/s__ |       84 MB/s |     1.03 GB/s |              — |  __1.15 GB/s__ |
| Bengali 🇧🇩    |       240 MB/s |      374 MB/s |   __424 MB/s__ |       97 MB/s |  __829 MB/s__ |              — |       406 MB/s |
| Chinese 🇨🇳    |       262 MB/s |      757 MB/s |  __1.16 GB/s__ | __4.24 GB/s__ |     1.69 GB/s |        77 MB/s |      1.17 GB/s |
| Czech 🇨🇿      |       124 MB/s |     3.87 GB/s | __15.39 GB/s__ |     1.94 GB/s |     1.04 GB/s |        55 MB/s | __12.42 GB/s__ |
| Dutch 🇳🇱      |       138 MB/s |     4.01 GB/s | __38.54 GB/s__ |     1.79 GB/s |      943 MB/s |        52 MB/s | __22.80 GB/s__ |
| English 🇬🇧    |       134 MB/s |     3.82 GB/s | __21.99 GB/s__ |     1.87 GB/s |      934 MB/s |        47 MB/s | __16.02 GB/s__ |
| Farsi 🇮🇷      |       174 MB/s |      507 MB/s |  __1.06 GB/s__ |       71 MB/s | __1.07 GB/s__ |              — |      1.07 GB/s |
| French 🇫🇷     |       134 MB/s |     3.70 GB/s | __13.59 GB/s__ |     1.81 GB/s |      969 MB/s |        53 MB/s | __10.73 GB/s__ |
| Georgian 🇬🇪   |       256 MB/s |      685 MB/s |  __1.05 GB/s__ |       96 MB/s | __1.57 GB/s__ |              — |      1.06 GB/s |
| German 🇩🇪     |       136 MB/s |     3.77 GB/s | __16.59 GB/s__ |     1.78 GB/s |      945 MB/s |        53 MB/s | __12.78 GB/s__ |
| Greek 🇬🇷      |       155 MB/s |      534 MB/s |  __1.28 GB/s__ | __2.85 GB/s__ |     1.01 GB/s |        64 MB/s |      1.27 GB/s |
| Hebrew 🇮🇱     |       168 MB/s |      509 MB/s |  __1.09 GB/s__ |       88 MB/s |      970 MB/s |              — |  __1.11 GB/s__ |
| Hindi 🇮🇳      |       240 MB/s |      502 MB/s |   __766 MB/s__ |       97 MB/s | __1.50 GB/s__ |              — |       761 MB/s |
| Italian 🇮🇹    |       137 MB/s |     3.98 GB/s | __29.26 GB/s__ |     1.76 GB/s |      953 MB/s |        54 MB/s | __20.20 GB/s__ |
| Japanese 🇯🇵   |       225 MB/s |      734 MB/s |  __1.02 GB/s__ | __4.37 GB/s__ |     1.73 GB/s |        85 MB/s |      1.04 GB/s |
| Korean 🇰🇷     |       134 MB/s |      598 MB/s |   __864 MB/s__ | __4.25 GB/s__ |     1.41 GB/s |        74 MB/s |       795 MB/s |
| Polish 🇵🇱     |       129 MB/s |     3.68 GB/s | __11.81 GB/s__ |      123 MB/s |      979 MB/s |              — | __10.05 GB/s__ |
| Portuguese 🇧🇷 |       129 MB/s |     3.98 GB/s | __20.09 GB/s__ |     1.81 GB/s |      965 MB/s |        54 MB/s | __14.24 GB/s__ |
| Russian 🇷🇺    |       171 MB/s |      508 MB/s | __11.58 GB/s__ |       73 MB/s |     1.03 GB/s |              — |  __9.83 GB/s__ |
| Spanish 🇪🇸    |       134 MB/s |     3.99 GB/s | __21.68 GB/s__ |     1.79 GB/s |      971 MB/s |        53 MB/s | __14.92 GB/s__ |
| Tamil 🇮🇳      |       219 MB/s |      367 MB/s |   __552 MB/s__ |       86 MB/s |  __968 MB/s__ |              — |       533 MB/s |
| Turkish 🇹🇷    |       120 MB/s |     3.76 GB/s | __13.58 GB/s__ |     1.91 GB/s |     1.02 GB/s |        55 MB/s | __10.67 GB/s__ |
| Ukrainian 🇺🇦  |       168 MB/s |      540 MB/s |  __6.25 GB/s__ |       72 MB/s |     1.07 GB/s |              — |  __5.66 GB/s__ |
| Vietnamese 🇻🇳 |       111 MB/s | __1.50 GB/s__ |       861 MB/s |       75 MB/s |  __916 MB/s__ |              — |       817 MB/s |

> Measured July 29, 2026.

---

See [README.md](../README.md) for dataset information and replication instructions.
