# Sequence Operations Benchmarks

Benchmarks for string collection operations like sorting across Rust and Python implementations.

## Overview

Rust has several Dataframe libraries, DBMS and Search engines that heavily rely on string sorting and intersections.
Those operations mostly are implemented using conventional algorithms:

- Comparison-based Quicksort or Mergesort for sorting.
- Hash-based or Tree-based algorithms for intersections.

Assuming the compares can be accelerated with SIMD and so can be the hash functions, StringZilla could already provide a performance boost in such applications, but starting with v4 it also provides specialized algorithms for sorting and intersections.
Those are directly compatible with arbitrary string-comparable collection types with a support of an indexed access to the elements.

## String Sorting

Sorting short whitespace-delimited words from `xlsum.csv` on a single core, in plain __byte order__ and in __Unicode case-folded__ order (where `ß` sorts as `ss`).
The table fuses both index sorts and full sorts: `argsort` / `arg_sort` / `lexsort_to_indices` / `sort_indices` return only the index permutation, while `sort` / `sorted` / `sort_values` return the reordered sequence.
StringZilla's `argsort` writes that permutation into a caller-owned buffer — in Python a NumPy `out=` array — so it allocates nothing per call, and `Strs.sorted` hands back a reordered _view_ over the shared bytes instead of copying the strings.
Every engine builds its container outside the timed region and reuses its output buffer where the API allows.
StringZilla's sort is always stable, so each competitor is configured for a stable sort where it exposes one: NumPy and pandas use `kind="stable"`, Rust's `std` uses the stable `sort_by_key` / `sort_by`, and Polars keeps `maintain_order` on in Rust — its Python `Series.sort` has no stability flag, so that row runs Polars' default order.
For the case-folded column every folding sort shares one comparator, StringZilla's `sz_utf8_uncased_order`, so the gap measures the sort algorithm rather than differing Unicode tables; in Python that comparator is reached through `functools.cmp_to_key`, whose per-element proxy dispatch dominates the `list.sort` row.

### Intel Xeon4 Sapphire Rapids

| Library                        |              Byte Order |     Unicode Case-Folded |
| ------------------------------ | ----------------------: | ----------------------: |
| Rust                           |                         |                         |
| `stringzilla::argsort`         | __209.32 M compares/s__ |  __97.34 M compares/s__ |
| `polars::DataFrame::sort`      |     208.12 M compares/s |                       — |
| `polars::Series::sort`         |     205.21 M compares/s |                       — |
| `arrow::lexsort_to_indices`    |     122.58 M compares/s |                       — |
| `polars::Series::arg_sort`     |      58.06 M compares/s |                       — |
| `std::sort_by_key`             |      37.46 M compares/s |      24.72 M compares/s |
|                                |                         |                         |
| Python                         |                         |                         |
| `polars.Series.sort`           | __229.90 M compares/s__ |                       — |
| `stringzilla.Strs.argsort`     |     219.92 M compares/s | __105.76 M compares/s__ |
| `stringzilla.Strs.sorted`      |     178.71 M compares/s |      95.62 M compares/s |
| `pyarrow.compute.sort_indices` |      72.99 M compares/s |                       — |
| `pandas.Series.sort_values`    |      59.44 M compares/s |                       — |
| `list.sort`                    |      51.10 M compares/s |      10.21 M compares/s |
| `polars.Series.arg_sort`       |      31.47 M compares/s |                       — |
| `numpy.argsort`                |      21.65 M compares/s |                       — |

> Measured June 17, 2026, single-threaded (Polars pinned to one thread), sorting short words from `xlsum.csv`.

### Apple M5 Pro

| Library                        |              Byte Order |     Unicode Case-Folded |
| ------------------------------ | ----------------------: | ----------------------: |
| Rust                           |                         |                         |
| `stringzilla::argsort`         | __540.89 M compares/s__ | __232.68 M compares/s__ |
| `arrow::lexsort_to_indices`    |     328.57 M compares/s |                       — |
| `polars::Series::sort`         |     283.13 M compares/s |                       — |
| `polars::DataFrame::sort`      |     277.38 M compares/s |                       — |
| `polars::Series::arg_sort`     |     207.91 M compares/s |                       — |
| `std::sort_by_key`             |     116.47 M compares/s |      52.72 M compares/s |
|                                |                         |                         |
| Python                         |                         |                         |
| `stringzilla.Strs.argsort`     | __577.80 M compares/s__ | __303.35 M compares/s__ |
| `stringzilla.Strs.sorted`      |     537.43 M compares/s |     288.58 M compares/s |
| `polars.Series.sort`           |     281.42 M compares/s |                       — |
| `pyarrow.compute.sort_indices` |     128.68 M compares/s |                       — |
| `pandas.Series.sort_values`    |     117.46 M compares/s |                       — |
| `polars.Series.arg_sort`       |     114.46 M compares/s |                       — |
| `list.sort`                    |     107.48 M compares/s |      22.25 M compares/s |
| `numpy.argsort`                |      46.34 M compares/s |                       — |

> Measured July 29, 2026.

---

See the [top-level README](../README.md) for dataset information and replication instructions.
