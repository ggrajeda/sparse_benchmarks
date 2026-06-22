#!/usr/bin/env python3
"""Python sparse-matrix benchmark harness.

Reads the shared configs/bench.json, deterministically builds each matrix in COO
(via the incremental modular map documented in the config), converts to CSR once
(untimed), then times each operation `warmup`+`repetitions` times and appends the
results as tidy long-format rows to results/python_results.csv.

The matrix is byte-identical to what the R harness builds from the same config, so
absolute coordinates/values line up across languages.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

# Number of dense columns for the SpMM (sparse x dense) benchmark. Kept small so
# the result is bounded in memory; the R harness must use the same value.
SPMM_COLS = 16


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
    """Pin BLAS/threading env BEFORE numpy is imported, else it has no effect."""
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
from time import perf_counter            # noqa: E402


def generate_csr(rows, cols, nnz, stride, value_mod, chunk=50_000_000):
    """Build the deterministic matrix in COO and return it as CSR.

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
    csr = coo.tocsr()                      # untimed; sums duplicates (none here)
    # Past 2**31 nnz scipy must use 64-bit indices; fail loudly if it didn't.
    if nnz > 2 ** 31 - 1 and csr.indptr.dtype != np.int64:
        raise RuntimeError("CSR indptr is not int64 despite nnz > 2**31; "
                           "indices would silently truncate.")
    return csr


def make_operations(M):
    """Return {name: zero-arg callable} for the operations we benchmark.

    Each callable returns a materialized result (no lazy views) so timing
    reflects real work.
    """
    x = np.ones(M.shape[1], dtype=np.float64)               # spmv operand
    X = np.ones((M.shape[1], SPMM_COLS), dtype=np.float64)   # spmm operand
    return {
        "spmv":      lambda: M @ x,
        "spmm":      lambda: M @ X,
        "row_sums":  lambda: np.asarray(M.sum(axis=1)).ravel(),
        "col_sums":  lambda: np.asarray(M.sum(axis=0)).ravel(),
        "sum":       lambda: M.sum(),
    }


def time_op(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        result = fn()
        times.append(perf_counter() - t0)
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
    columns = CONFIG["output"]["columns"]

    import csv
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()

        for spec in matrices:
            name = spec["name"]
            rows, cols, nnz = int(spec["rows"]), int(spec["cols"]), int(spec["nnz"])
            print(f"[python] building '{name}' ({rows}x{cols}, nnz={nnz}) ...",
                  flush=True)
            M = generate_csr(rows, cols, nnz, stride, value_mod)
            ops = make_operations(M)

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
            del M, ops

    print(f"[python] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
