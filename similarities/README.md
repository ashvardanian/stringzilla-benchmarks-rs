# Similarity Scoring Benchmarks

Benchmarks for string similarity and alignment algorithms across Rust and Python implementations, including CPU and GPU variants.

## Overview

Edit Distance calculation is a common component of Search Engines, Data Cleaning, and Natural Language Processing, as well as in Bioinformatics.
It's a computationally expensive operation, generally implemented using dynamic programming, with a quadratic time complexity upper bound.
For biological sequences, the Needleman-Wunsch and Smith-Waterman algorithms are more appropriate, as they allow overriding the default substitution costs.
Each of those has two flavors - with linear and affine gap penalties, also known as the "Gotoh" variation.

Performance is measured in MCUPS (Million Cell Updates Per Second).
Both harnesses (`bench.rs` and `bench.py`) score an all-pairs cross-product: a `side x side` matrix of disjoint query and candidate batches, where each axis is `round(sqrt(STRINGWARS_BATCH_PER_CORE * cores))`.
A CPU core and a GPU streaming-multiprocessor (SM) each count as one core, so `<1xSPR>` runs one core, `<16xSPR>` all sixteen, and `<H100>` feeds the device a batch sized across the H100's 132 SMs.
MCUPS is the aggregate cell-update count over the whole matrix divided by the wall-clock time.

Datasets:

- __acgt_100 / acgt_1k__ — uniform synthetic DNA, lines of ~100 and ~1,000 bytes (~100 bytes is also the canonical protein-domain length).
- __XLSum words / lines__ — highly non-uniform multilingual text: ~5-byte word tokens and ~3.2 KB article lines.

The synthetic-DNA and word runs use `STRINGWARS_BATCH_PER_CORE=16384` to saturate the GPU (`<1xSPR>` side 128; `<16xSPR>` side 512; `<H100>` side 1,471).
The ~3.2 KB XLSum-lines run uses `STRINGWARS_BATCH_PER_CORE=256` (`<H100>` side 184) because each line-vs-line pair carries ~1,000x more cells than a word pair, so a smaller batch already saturates the device and keeps the single-core runs tractable.
StringZilla scores every column with the same unary 32-class match/mismatch costs, and the `bio` / `biopython` baselines use the same unary match/mismatch scoring, so the cross-language comparison stays apples-to-apples.

## Levenshtein Distance

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                          |         ACGT 100B |            ACGT 1KB |   XLSum words |       XLSum lines |
| ------------------------------------------------ | ----------------: | ------------------: | ------------: | ----------------: |
| Rust                                             |                   |                     |               |                   |
| `bio::levenshtein<1xSPR>`                        |         337 MCUPS |           674 MCUPS |     184 MCUPS |         124 MCUPS |
| `rapidfuzz::levenshtein<Bytes,1xSPR>`            |       3,300 MCUPS |        12,390 MCUPS |   1,110 MCUPS |       9,990 MCUPS |
| `rapidfuzz::levenshtein<Chars,1xSPR>`            |       2,300 MCUPS |         8,990 MCUPS |     181 MCUPS |      11,070 MCUPS |
| `stringzillas::LevenshteinDistances<1xSPR>`      |  __15,680 MCUPS__ |        12,770 MCUPS |   3,360 MCUPS |       5,850 MCUPS |
| `stringzillas::LevenshteinDistances<16xSPR>`     |     127,860 MCUPS |   __141,800 MCUPS__ |  20,820 MCUPS |      36,780 MCUPS |
| `stringzillas::LevenshteinDistances<H100>`       |   5,980,110 MCUPS | __6,237,990 MCUPS__ | 139,850 MCUPS |      41,850 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<1xSPR>`  |   __8,640 MCUPS__ |         8,310 MCUPS |     188 MCUPS |       7,750 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<16xSPR>` |      67,870 MCUPS |   __100,460 MCUPS__ |   1,600 MCUPS |      43,260 MCUPS |
|                                                  |                   |                     |               |                   |
| Python                                           |                   |                     |               |                   |
| `rapidfuzz.Levenshtein.distance`                 |       2,670 MCUPS |        15,720 MCUPS |   24.94 MCUPS |      14,020 MCUPS |
| `Levenshtein.distance`                           |       2,520 MCUPS |        15,610 MCUPS |   22.10 MCUPS |      14,260 MCUPS |
| `jellyfish.levenshtein_distance`                 |      107.50 MCUPS |        130.44 MCUPS |   16.19 MCUPS |      173.60 MCUPS |
| `editdistance.eval`                              |       1,200 MCUPS |        377.29 MCUPS |   14.28 MCUPS |      545.27 MCUPS |
| `nltk.edit_distance`                             |        1.29 MCUPS |          0.98 MCUPS |    0.89 MCUPS |        0.90 MCUPS |
| `edlib.align`                                    |       1,480 MCUPS |         9,160 MCUPS |   11.40 MCUPS |               — ‖ |
| `polyleven.levenshtein`                          |       2,490 MCUPS |         5,100 MCUPS |   89.30 MCUPS |      11,510 MCUPS |
| `cudf.edit_distance<H100>` ‡                     |                 — |                   — |             — |                 — |
| `stringzillas.LevenshteinDistances<1xSPR>`       |  __14,250 MCUPS__ |        11,930 MCUPS |   2,090 MCUPS |       6,950 MCUPS |
| `stringzillas.LevenshteinDistances<16xSPR>`      | __176,770 MCUPS__ |       159,350 MCUPS |  11,290 MCUPS |      43,700 MCUPS |
| `stringzillas.LevenshteinDistances<H100>`        |   4,074,700 MCUPS | __6,022,000 MCUPS__ |  26,960 MCUPS |     265,820 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<1xSPR>`   |   __8,710 MCUPS__ |         7,460 MCUPS |  185.95 MCUPS |       6,970 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<16xSPR>`  | __109,800 MCUPS__ |       106,210 MCUPS |   1,530 MCUPS |      63,090 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<H100>`    |     232,970 MCUPS |        91,320 MCUPS |   7,870 MCUPS | __508,710 MCUPS__ |

> Measured June 19, 2026.

### Apple M5 Pro

| Library                                         |        ACGT 100B |          ACGT 1KB |      XLSum words |       XLSum lines |
| ----------------------------------------------- | ---------------: | ----------------: | ---------------: | ----------------: |
| Rust                                            |                  |                   |                  |                   |
| `bio::levenshtein<1xM5>`                        |        900 MCUPS |       1,567 MCUPS |        436 MCUPS |         217 MCUPS |
| `rapidfuzz::levenshtein<Bytes,1xM5>`            |      9,119 MCUPS |      24,876 MCUPS |      2,037 MCUPS |      21,589 MCUPS |
| `rapidfuzz::levenshtein<Chars,1xM5>`            |      8,664 MCUPS |      24,464 MCUPS |        468 MCUPS |      21,075 MCUPS |
| `stringzillas::LevenshteinDistances<1xM5>`      |     22,422 MCUPS |      54,250 MCUPS |      2,482 MCUPS |      14,073 MCUPS |
| `stringzillas::LevenshteinDistances<18xM5>`     |     49,922 MCUPS | __715,914 MCUPS__ |      2,537 MCUPS |     150,332 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<1xM5>`  |     22,392 MCUPS |      54,255 MCUPS |        711 MCUPS |      28,444 MCUPS |
| `stringzillas::LevenshteinDistancesUtf8<18xM5>` | __53,979 MCUPS__ |     694,590 MCUPS | __11,028 MCUPS__ | __316,631 MCUPS__ |
|                                                 |                  |                   |                  |                   |
| Python                                          |                  |                   |                  |                   |
| `rapidfuzz.Levenshtein.distance`                |      9,704 MCUPS |      29,245 MCUPS |        215 MCUPS |      26,394 MCUPS |
| `Levenshtein.distance`                          |      9,189 MCUPS |      29,192 MCUPS |        140 MCUPS |      26,423 MCUPS |
| `jellyfish.levenshtein_distance`                |        214 MCUPS |         252 MCUPS |         59 MCUPS |         432 MCUPS |
| `editdistance.eval`                             |      4,652 MCUPS |         814 MCUPS |         51 MCUPS |         780 MCUPS |
| `nltk.edit_distance`                            |       5.94 MCUPS |        4.64 MCUPS |       3.78 MCUPS |                 — |
| `edlib.align`                                   |      4,864 MCUPS |      24,552 MCUPS |         19 MCUPS |      26,314 MCUPS |
| `polyleven.levenshtein`                         |     10,828 MCUPS |      11,316 MCUPS |        461 MCUPS |      26,301 MCUPS |
| `stringzillas.LevenshteinDistances<1xM5>`       |     22,084 MCUPS |      54,549 MCUPS |  __1,396 MCUPS__ |      14,478 MCUPS |
| `stringzillas.LevenshteinDistances<18xM5>`      | __46,893 MCUPS__ | __723,480 MCUPS__ |        445 MCUPS |     181,045 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<1xM5>`   |     22,074 MCUPS |      54,532 MCUPS |        479 MCUPS |      27,126 MCUPS |
| `stringzillas.LevenshteinDistancesUTF8<18xM5>`  |     46,047 MCUPS |     696,989 MCUPS |        166 MCUPS | __399,501 MCUPS__ |

> Measured July 29, 2026.

## Needleman-Wunsch for Global Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                       |     ACGT 100B |          ACGT 1KB |  XLSum words |   XLSum lines |
| --------------------------------------------- | ------------: | ----------------: | -----------: | ------------: |
| Rust                                          |               |                   |              |               |
| `bio::pairwise::global<1xSPR>`                |      30 MCUPS |          47 MCUPS |     35 MCUPS |      32 MCUPS |
| `stringzillas::NeedlemanWunschScores<1xSPR>`  |   2,090 MCUPS |  __12,000 MCUPS__ |     46 MCUPS |     604 MCUPS |
| `stringzillas::NeedlemanWunschScores<16xSPR>` |  15,280 MCUPS |  __90,450 MCUPS__ |    328 MCUPS |   6,520 MCUPS |
| `stringzillas::NeedlemanWunschScores<H100>`   | 414,930 MCUPS | __701,760 MCUPS__ | 26,480 MCUPS | 234,810 MCUPS |
|                                               |               |                   |              |               |
| Python                                        |               |                   |              |               |
| `biopython.PairwiseAligner.global`            |  374.82 MCUPS |      444.36 MCUPS |  18.49 MCUPS |  751.30 MCUPS |
| `stringzillas.NeedlemanWunschScores<1xSPR>`   |   3,860 MCUPS |   __8,670 MCUPS__ |  43.69 MCUPS |  833.62 MCUPS |
| `stringzillas.NeedlemanWunschScores<16xSPR>`  |  42,390 MCUPS | __111,580 MCUPS__ | 369.22 MCUPS |   5,190 MCUPS |
| `stringzillas.NeedlemanWunschScores<H100>`    | 396,550 MCUPS | __700,900 MCUPS__ | 18,800 MCUPS | 203,490 MCUPS |

> Measured June 19, 2026.

### Apple M5 Pro

| Library                                      |        ACGT 100B |         ACGT 1KB |     XLSum words |     XLSum lines |
| -------------------------------------------- | ---------------: | ---------------: | --------------: | --------------: |
| Rust                                         |                  |                  |                 |                 |
| `bio::pairwise::global<1xM5>`                |        153 MCUPS |        157 MCUPS |       124 MCUPS |   __124 MCUPS__ |
| `stringzillas::NeedlemanWunschScores<1xM5>`  |        974 MCUPS |      1,071 MCUPS |       252 MCUPS |               — |
| `stringzillas::NeedlemanWunschScores<18xM5>` | __11,684 MCUPS__ | __15,581 MCUPS__ | __1,065 MCUPS__ |               — |
|                                              |                  |                  |                 |                 |
| Python                                       |                  |                  |                 |                 |
| `biopython.PairwiseAligner.global`           |        914 MCUPS |        842 MCUPS |        34 MCUPS | __1,894 MCUPS__ |
| `stringzillas.NeedlemanWunschScores<1xM5>`   |        823 MCUPS |      1,017 MCUPS |        24 MCUPS |               — |
| `stringzillas.NeedlemanWunschScores<18xM5>`  | __10,193 MCUPS__ | __14,331 MCUPS__ |   __210 MCUPS__ |               — |

> Measured July 29, 2026.

## Smith-Waterman for Local Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                     |     ACGT 100B |          ACGT 1KB |  XLSum words |   XLSum lines |
| ------------------------------------------- | ------------: | ----------------: | -----------: | ------------: |
| Rust                                        |               |                   |              |               |
| `bio::pairwise::local<1xSPR>`               |      51 MCUPS |          42 MCUPS |     34 MCUPS |      33 MCUPS |
| `stringzillas::SmithWatermanScores<1xSPR>`  |   1,740 MCUPS |   __8,710 MCUPS__ |     44 MCUPS |     569 MCUPS |
| `stringzillas::SmithWatermanScores<16xSPR>` |  14,560 MCUPS |  __76,170 MCUPS__ |    329 MCUPS |   5,880 MCUPS |
| `stringzillas::SmithWatermanScores<H100>`   | 339,880 MCUPS | __607,390 MCUPS__ | 23,010 MCUPS | 225,600 MCUPS |
|                                             |               |                   |              |               |
| Python                                      |               |                   |              |               |
| `biopython.PairwiseAligner.local`           |  247.40 MCUPS |      412.19 MCUPS |  19.80 MCUPS |  593.81 MCUPS |
| `stringzillas.SmithWatermanScores<1xSPR>`   |   3,310 MCUPS |   __8,500 MCUPS__ |  45.51 MCUPS |  804.90 MCUPS |
| `stringzillas.SmithWatermanScores<16xSPR>`  |  39,710 MCUPS | __102,950 MCUPS__ | 379.24 MCUPS |   5,860 MCUPS |
| `stringzillas.SmithWatermanScores<H100>`    | 329,370 MCUPS | __607,900 MCUPS__ | 16,710 MCUPS | 195,370 MCUPS |

> Measured June 19, 2026.

### Apple M5 Pro

| Library                                    |        ACGT 100B |         ACGT 1KB |   XLSum words |     XLSum lines |
| ------------------------------------------ | ---------------: | ---------------: | ------------: | --------------: |
| Rust                                       |                  |                  |               |                 |
| `bio::pairwise::local<1xM5>`               |        144 MCUPS |        147 MCUPS |     128 MCUPS |   __127 MCUPS__ |
| `stringzillas::SmithWatermanScores<1xM5>`  |        960 MCUPS |      1,048 MCUPS |     249 MCUPS |               — |
| `stringzillas::SmithWatermanScores<18xM5>` | __11,465 MCUPS__ | __14,810 MCUPS__ | __931 MCUPS__ |               — |
|                                            |                  |                  |               |                 |
| Python                                     |                  |                  |               |                 |
| `biopython.PairwiseAligner.local`          |        776 MCUPS |        782 MCUPS |      34 MCUPS | __1,905 MCUPS__ |
| `stringzillas.SmithWatermanScores<1xM5>`   |        843 MCUPS |      1,055 MCUPS |      24 MCUPS |               — |
| `stringzillas.SmithWatermanScores<18xM5>`  | __10,436 MCUPS__ | __15,984 MCUPS__ | __209 MCUPS__ |               — |

> Measured July 29, 2026.

## Needleman-Wunsch-Gotoh for Global Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                       |       ACGT 100B |          ACGT 1KB |  XLSum words |   XLSum lines |
| --------------------------------------------- | --------------: | ----------------: | -----------: | ------------: |
| Rust                                          |                 |                   |              |               |
| `bio::pairwise::global<1xSPR>`                |        51 MCUPS |          47 MCUPS |     40 MCUPS |      35 MCUPS |
| `stringzillas::NeedlemanWunschScores<1xSPR>`  |     2,650 MCUPS |   __2,660 MCUPS__ |     75 MCUPS |     213 MCUPS |
| `stringzillas::NeedlemanWunschScores<16xSPR>` |    17,760 MCUPS |  __33,300 MCUPS__ |    640 MCUPS |   2,110 MCUPS |
| `stringzillas::NeedlemanWunschScores<H100>`   |   226,980 MCUPS | __397,400 MCUPS__ | 15,940 MCUPS | 115,650 MCUPS |
|                                               |                 |                   |              |               |
| Python                                        |                 |                   |              |               |
| `biopython.PairwiseAligner.global`            |    218.68 MCUPS |      254.93 MCUPS |  18.89 MCUPS |  504.15 MCUPS |
| `stringzillas.NeedlemanWunschScores<1xSPR>`   | __3,000 MCUPS__ |       3,280 MCUPS |  43.99 MCUPS |  301.87 MCUPS |
| `stringzillas.NeedlemanWunschScores<16xSPR>`  |    34,260 MCUPS |  __60,840 MCUPS__ | 341.98 MCUPS |   2,560 MCUPS |
| `stringzillas.NeedlemanWunschScores<H100>`    |   211,610 MCUPS | __395,760 MCUPS__ | 16,180 MCUPS | 119,740 MCUPS |

> Measured June 19, 2026.

### Apple M5 Pro

| Library                                      |        ACGT 100B |         ACGT 1KB |   XLSum words |     XLSum lines |
| -------------------------------------------- | ---------------: | ---------------: | ------------: | --------------: |
| Rust                                         |                  |                  |               |                 |
| `bio::pairwise::global<1xM5>`                |        138 MCUPS |        142 MCUPS |     126 MCUPS |   __123 MCUPS__ |
| `stringzillas::NeedlemanWunschScores<1xM5>`  |        929 MCUPS |        986 MCUPS |     243 MCUPS |               — |
| `stringzillas::NeedlemanWunschScores<18xM5>` | __10,780 MCUPS__ | __14,184 MCUPS__ | __862 MCUPS__ |               — |
|                                              |                  |                  |               |                 |
| Python                                       |                  |                  |               |                 |
| `biopython.PairwiseAligner.global`           |        610 MCUPS |        700 MCUPS |      34 MCUPS | __1,631 MCUPS__ |
| `stringzillas.NeedlemanWunschScores<1xM5>`   |        902 MCUPS |      1,007 MCUPS |      43 MCUPS |               — |
| `stringzillas.NeedlemanWunschScores<18xM5>`  | __10,683 MCUPS__ | __15,219 MCUPS__ | __232 MCUPS__ |               — |

> Measured July 29, 2026.

## Smith-Waterman-Gotoh for Local Alignment

### Intel Xeon4 Sapphire Rapids & NVIDIA H100

| Library                                     |       ACGT 100B |          ACGT 1KB |  XLSum words |   XLSum lines |
| ------------------------------------------- | --------------: | ----------------: | -----------: | ------------: |
| Rust                                        |                 |                   |              |               |
| `bio::pairwise::local<1xSPR>`               |        48 MCUPS |          44 MCUPS |     33 MCUPS |      37 MCUPS |
| `stringzillas::SmithWatermanScores<1xSPR>`  | __2,890 MCUPS__ |       2,550 MCUPS |     73 MCUPS |     213 MCUPS |
| `stringzillas::SmithWatermanScores<16xSPR>` |    19,180 MCUPS |  __35,310 MCUPS__ |    614 MCUPS |   2,150 MCUPS |
| `stringzillas::SmithWatermanScores<H100>`   |   226,790 MCUPS | __364,790 MCUPS__ | 17,340 MCUPS | 117,690 MCUPS |
|                                             |                 |                   |              |               |
| Python                                      |                 |                   |              |               |
| `biopython.PairwiseAligner.local`           |    115.17 MCUPS |      165.71 MCUPS |  18.92 MCUPS |  342.68 MCUPS |
| `stringzillas.SmithWatermanScores<1xSPR>`   | __2,670 MCUPS__ |       2,640 MCUPS |  41.97 MCUPS |  317.48 MCUPS |
| `stringzillas.SmithWatermanScores<16xSPR>`  |    30,500 MCUPS |  __58,560 MCUPS__ | 370.97 MCUPS |   1,660 MCUPS |
| `stringzillas.SmithWatermanScores<H100>`    |   216,750 MCUPS | __365,470 MCUPS__ | 14,030 MCUPS | 121,680 MCUPS |

> Measured June 19, 2026.

### Apple M5 Pro

| Library                                    |        ACGT 100B |         ACGT 1KB |   XLSum words |   XLSum lines |
| ------------------------------------------ | ---------------: | ---------------: | ------------: | ------------: |
| Rust                                       |                  |                  |               |               |
| `bio::pairwise::local<1xM5>`               |        119 MCUPS |        124 MCUPS |     131 MCUPS | __117 MCUPS__ |
| `stringzillas::SmithWatermanScores<1xM5>`  |        886 MCUPS |        956 MCUPS |     232 MCUPS |             — |
| `stringzillas::SmithWatermanScores<18xM5>` | __10,565 MCUPS__ | __13,303 MCUPS__ | __887 MCUPS__ |             — |
|                                            |                  |                  |               |               |
| Python                                     |                  |                  |               |               |
| `biopython.PairwiseAligner.local`          |        387 MCUPS |        396 MCUPS |      33 MCUPS | __999 MCUPS__ |
| `stringzillas.SmithWatermanScores<1xM5>`   |        879 MCUPS |        975 MCUPS |      43 MCUPS |             — |
| `stringzillas.SmithWatermanScores<18xM5>`  | __10,421 MCUPS__ | __14,582 MCUPS__ | __260 MCUPS__ |             — |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
