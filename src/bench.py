#!/usr/bin/env python3
"""Python sparse-matrix benchmark harness.

Reads the shared configs/bench.json, deterministically builds each matrix in COO
(via the modular map documented in the config), converts to CSC once
(untimed), then times each operation `warmup`+`repetitions` times and appends the
results as tidy long-format rows to results/python_results.csv.

We store CSC (not CSR) so the layout matches R's SVT_SparseArray, which is
column-compressed: with both libraries column-native, every operation is an
apples-to-apples comparison rather than row-major-vs-column-major.

The matrix is byte-identical to what the R harness builds from the same config, so
absolute coordinates/values line up across languages. After timing, each harness
also writes a compact, order-independent digest of every operation's result
(results/python_digests.csv) so scripts/crosscheck.py can confirm R and Python
actually compute the same thing.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

# Number of dense weight vectors for the vecmat_multi (dense x sparse) benchmark --
# e.g. several gene signatures scored at once. The R harness must use the same
# value.
N_VECS = 16


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    p.add_argument("--config", type=Path,
                   default=repo_root / "configs" / "bench.json")
    p.add_argument("--matrices", type=str, default=None,
                   help="Comma-separated subset of matrix names to run "
                        "(default: all in config). Useful to skip 'large' on "
                        "memory-limited hosts.")
    return p.parse_args()


def set_threads(n):
    """Pin BLAS/OMP thread env vars (must run BEFORE numpy is imported, or it has
    no effect).

    We pin the threads for reproducibility and to make sure CPU time measures the
    true compute time: time_op measures CPU time (process_time), which sums across
    threads, so it only equals true compute time when single-threaded.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


# ---- parse args + config and set threads before importing numpy/scipy ----
ARGS = parse_args()
with open(ARGS.config) as fh:
    CONFIG = json.load(fh)
set_threads(int(CONFIG["run"]["threads"]))

import numpy as np                       # noqa: E402
import scipy                             # noqa: E402
from scipy import sparse                 # noqa: E402
from time import process_time            # noqa: E402


def generate_csc(rows, cols, nnz, stride, value_mod, chunk=50_000_000):
    """Build the deterministic matrix in COO and return it as CSC.

    k_i = (i * stride) % (rows*cols), row_i = k_i // cols, col_i = k_i % cols,
    v_i = 1 + (k_i % value_mod). All index arithmetic is done in exact int64;
    we chunk so that the largest intermediate (chunk*stride) cannot overflow.
    """
    mn = rows * cols
    g = math.gcd(stride, mn)
    if g != 1:
        raise ValueError(
            f"stride {stride} is not coprime to rows*cols ({mn}); gcd={g}. "
            "The k_i would collide and produce duplicate coordinates.")
    if nnz > mn:
        raise ValueError(f"nnz ({nnz}) exceeds rows*cols ({mn}).")
    # Overflow guard for the int64 arithmetic below: base < mn and the per-chunk
    # product is bounded by chunk*stride; keep the sum well under 2**63.
    if mn >= 2 ** 62:
        raise ValueError("rows*cols must stay below 2**62 for safe int64 math.")
    chunk = min(chunk, (2 ** 62) // stride)

    row = np.empty(nnz, dtype=np.int64)
    col = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)

    i0 = 0
    while i0 < nnz:
        n = min(chunk, nnz - i0)
        base = (i0 * stride) % mn          # Python int: exact, no overflow
        k = (base + np.arange(n, dtype=np.int64) * stride) % mn
        row[i0:i0 + n] = k // cols
        col[i0:i0 + n] = k % cols
        data[i0:i0 + n] = 1.0 + (k % value_mod)
        i0 += n

    coo = sparse.coo_matrix((data, (row, col)), shape=(rows, cols))
    csc = coo.tocsc()                      # untimed; sums duplicates (none here)
    # indptr holds cumulative nnz (up to `nnz`), so past 2**31 nnz scipy must use
    # 64-bit indices or the counts would overflow; fail loudly if it didn't.
    if nnz > 2 ** 31 - 1 and csc.indptr.dtype != np.int64:
        raise RuntimeError("CSC indptr is not int64 despite nnz > 2**31; "
                           "indices would silently truncate.")
    return csc


def make_operations(M, value_mod):
    """Return {name: zero-arg callable} for the operations we benchmark.

    Each callable returns a materialized result (no lazy views) so timing
    reflects real work.

    These are LEFT-multiplies (vecmat: dense @ sparse) -- the fast path for a
    column-compressed store, and the genomics operation of interest: with rows =
    genes and cols = cells, `x @ M` is a per-cell weighted sum of genes (a
    signature score per cell), and `X @ M` scores N_VECS signatures at once.

    The operands are deterministic but NON-constant (a function of the row / gene
    index and, for vecmat_multi, the signature index), built identically in the R
    harness. An all-ones operand would make `x @ M` reproduce column sums, so the
    cross-check could not tell a transposed/permuted matrix from a correct one;
    varying the operand gives the digest real discriminating power.
    """
    g0 = np.arange(M.shape[0], dtype=np.float64)             # 0-based row (gene) index
    x = 1.0 + (g0 % value_mod)                               # vecmat operand, length rows
    c0 = np.arange(N_VECS, dtype=np.float64)
    X = 1.0 + ((c0[:, None] + g0[None, :]) % value_mod)      # vecmat_multi operand, (N_VECS, rows)
    return {
        "vecmat":       lambda: x @ M,
        "vecmat_multi": lambda: X @ M,
        "row_sums":     lambda: np.asarray(M.sum(axis=1)).ravel(),
        "col_sums":     lambda: np.asarray(M.sum(axis=0)).ravel(),
        "sum":          lambda: M.sum(),
    }


def digest(result):
    """Compact, order-independent summary of an operation's result.

    All four reductions are invariant to element order, so they match across
    languages despite numpy flattening row-major and R column-major. Plain sum is
    order-sensitive in floating point, but only at the level of accumulated
    rounding, which the cross-check tolerates via a relative threshold.
    """
    a = np.asarray(result, dtype=np.float64).ravel()         # view when contiguous
    n = a.size
    return {
        "n": n,
        "sum": float(a.sum()),
        "sumsq": float(np.dot(a, a)),                         # no large temporary
        "min": float(a.min()) if n else 0.0,
        "max": float(a.max()) if n else 0.0,
    }


def time_op(fn, warmup, reps):
    # process_time() measures CPU time (user + system) of this process, not wall
    # clock. With threads pinned to 1 this tracks the kernel's actual compute and
    # excludes time the process spent descheduled, so it is robust to OS scheduling
    # and contention from other processes. The R harness uses proc.time() user+sys
    # to match, so both harnesses measure the same kind of time.
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = process_time()
        result = fn()
        times.append(process_time() - t0)
        del result
    return times


def main():
    run = CONFIG["run"]
    gen = CONFIG["generation"]
    threads = int(run["threads"])
    warmup = int(run["warmup"])
    reps = int(run["repetitions"])
    stride = int(gen["stride"])
    value_mod = int(gen["value_mod"])
    op_names = CONFIG["operations"]

    matrices = CONFIG["matrices"]
    if ARGS.matrices:
        wanted = {s.strip() for s in ARGS.matrices.split(",")}
        matrices = [m for m in matrices if m["name"] in wanted]
        if not matrices:
            sys.exit(f"No configured matrices match {sorted(wanted)}.")

    out_dir = Path(CONFIG["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "python_results.csv"
    digest_path = out_dir / "python_digests.csv"
    columns = CONFIG["output"]["columns"]
    digest_columns = ["language", "matrix_name", "operation",
                      "n", "sum", "sumsq", "min", "max"]

    import csv
    with open(out_path, "w", newline="") as fh, \
            open(digest_path, "w", newline="") as dfh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        dwriter = csv.DictWriter(dfh, fieldnames=digest_columns)
        dwriter.writeheader()

        for spec in matrices:
            name = spec["name"]
            rows, cols, nnz = int(spec["rows"]), int(spec["cols"]), int(spec["nnz"])
            print(f"[python] building '{name}' ({rows}x{cols}, nnz={nnz}) ...",
                  flush=True)
            M = generate_csc(rows, cols, nnz, stride, value_mod)
            ops = make_operations(M, value_mod)

            for op in op_names:
                if op not in ops:
                    print(f"[python]   skipping unknown op '{op}'", flush=True)
                    continue
                print(f"[python]   timing {op} ...", flush=True)
                for rep, t in enumerate(time_op(ops[op], warmup, reps), start=1):
                    writer.writerow({
                        "language": "python",
                        "library": "scipy",
                        "library_version": scipy.__version__,
                        "matrix_name": name,
                        "operation": op,
                        "rows": rows,
                        "cols": cols,
                        "nnz": nnz,
                        "threads": threads,
                        "rep": rep,
                        "time_sec": f"{t:.9f}",
                    })
                fh.flush()
                # One extra (untimed) eval to digest the result for the cross-check.
                d = digest(ops[op]())
                dwriter.writerow({
                    "language": "python", "matrix_name": name, "operation": op,
                    "n": d["n"],
                    "sum": f"{d['sum']:.17g}", "sumsq": f"{d['sumsq']:.17g}",
                    "min": f"{d['min']:.17g}", "max": f"{d['max']:.17g}",
                })
                dfh.flush()
            del M, ops

    print(f"[python] wrote {out_path}", flush=True)
    print(f"[python] wrote {digest_path}", flush=True)


if __name__ == "__main__":
    main()
