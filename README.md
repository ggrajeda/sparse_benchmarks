# sparse_benchmarks

Benchmarks for large sparse-matrix operations in Python (scipy) and R (SparseArray).
Both harnesses build the same deterministic matrix from a shared configuration
(`configs/bench.json`) and time five operations under identical conditions.

The matrices are oriented for genomics: **rows = genes, cols = cells** (a fixed
~10,000 genes by a large, growing number of cells). Cells are therefore the huge
dimension and the *column* axis — the compressed axis for both scipy CSC and R
SVT. The headline multiply is a **left-multiply** (`x %*% M`), which is each
library's fast path and corresponds to a per-cell weighted sum of genes (a
signature score per cell):

| Operation | Meaning |
|-----------|---------|
| `vecmat` | `x %*% M` — one gene-weight vector → a score per cell |
| `vecmat_multi` | `X %*% M` — `N_VECS` gene signatures scored at once |
| `row_sums` | total per gene |
| `col_sums` | total per cell (library size) |
| `sum` | grand total |

## Running

```bash
scripts/run_benchmarks.sh --matrices medium
```

Additional options:

| Flag | Description |
|------|-------------|
| `--matrices small,medium,large` | Run a subset of configured matrices |
| `--python-only` | Skip the R harness |
| `--r-only` | Skip the Python harness |
| `--config PATH` | Use an alternative config file |

Results are written to `results/all_results.csv`. A summary table is printed
automatically at the end of the run. To reprint it from an existing CSV:

```bash
python3 scripts/summarize.py results/all_results.csv
```

## Cross-check

Both harnesses store the matrix column-native (scipy **CSC**, R **SVT**), so each
operation is an apples-to-apples comparison rather than row-major (CSR) vs
column-major. To confirm the two languages actually compute the same thing, each
harness writes a compact digest of every operation's result — `n`, `sum`,
`sumsq`, `min`, `max` — to `results/{python,r}_digests.csv`. These reductions are
order-independent, so they match despite numpy flattening row-major and R
column-major. When both harnesses run, `run_benchmarks.sh` invokes
`scripts/crosscheck.py` to compare them within a relative tolerance (float
reduction order differs across libraries, so exact equality is not expected):

```bash
python3 scripts/crosscheck.py results        # [--rtol R] [--atol A]
```

The `vecmat`/`vecmat_multi` operands are deterministic but non-constant (a
function of the gene index), shared across both languages — an all-ones operand
would make the cross-check unable to distinguish a transposed/permuted matrix
from a correct one.

## Results

Matrix: **large** — 10,000 × 100,000,000, nnz = 2,200,000,000, single thread.

```
Operation     scipy (s)               SparseArray (s)         Ratio (R/Python)
              median [IQR]            median [IQR]
------------------------------------------------------------------------------
vecmat        9.663 [0.079]           17.009 [0.282]          1.8x slower
vecmat_multi  47.260 [3.105]          293.280 [11.833]        6.2x slower
row_sums      6.450 [0.024]           19.243 [0.134]          3.0x slower
col_sums      4.623 [0.025]           21.013 [0.038]          4.5x slower
sum           6.456 [0.041]           17.259 [0.129]          2.7x slower
------------------------------------------------------------------------------
n = 5 repetitions per cell (after warmup runs). IQR = Q3 - Q1.
Matrix: large (10,000 x 100,000,000, nnz = 2,200,000,000), single thread.
```

Both libraries store the matrix column-native (scipy **CSC**, R **SVT**) and scan
columns the same way, so these timing differences come from the libraries'
kernels themselves and not from comparing mismatched storage layouts — an
apples-to-apples comparison. SparseArray stays within a small factor of scipy on the
fast-path left-multiply, but the gap widens at this scale — most notably on
`vecmat_multi` (6.2x), where SparseArray's batched left-multiply scales less
favorably than scipy's.
