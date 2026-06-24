#!/usr/bin/env python3
"""
Print a summary table (median + IQR per operation x library) from a tidy
benchmark CSV produced by run_benchmarks.sh.

Usage: python3 scripts/summarize.py [path/to/all_results.csv]
"""

import csv
import sys
from pathlib import Path
from statistics import median

def iqr(values):
    n = len(values)
    s = sorted(values)
    # Type-7 quantiles (same as R default): h = 1 + (n-1)*p
    def quantile(p):
        h = 1 + (n - 1) * p
        lo = int(h) - 1
        hi = min(lo + 1, n - 1)
        return s[lo] + (h - int(h)) * (s[hi] - s[lo])
    return quantile(0.75) - quantile(0.25)

def main():
    if len(sys.argv) >= 2:
        csv_path = Path(sys.argv[1])
    else:
        repo_root = Path(__file__).resolve().parent.parent
        csv_path  = repo_root / "results" / "all_results.csv"

    if not csv_path.exists():
        sys.exit(f"error: CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Group times by (operation, language, library). Also collect the distinct
    # matrices and thread counts present so the footer can describe the actual run
    # instead of hardcoding it.
    groups: dict[tuple, list[float]] = {}
    max_rep = 0
    matrices: dict[str, tuple] = {}   # name -> (rows, cols, nnz)
    threads_seen: set[int] = set()
    for row in rows:
        key = (row["operation"], row["language"], row["library"])
        groups.setdefault(key, []).append(float(row["time_sec"]))
        max_rep = max(max_rep, int(row["rep"]))
        matrices.setdefault(
            row["matrix_name"],
            (int(row["rows"]), int(row["cols"]), int(row["nnz"])))
        threads_seen.add(int(row["threads"]))

    ops_order = ["vecmat", "vecmat_multi", "row_sums", "col_sums", "sum"]

    # Identify the python and r library names from the data
    lib_py = next((lib for (_, lang, lib) in groups if lang == "python"), "python")
    lib_r  = next((lib for (_, lang, lib) in groups if lang == "r"),      "r")

    def cell(vals):
        if not vals:
            return "N/A"
        m = median(vals)
        i = iqr(vals)
        return f"{m:.3f} [{i:.3f}]"

    def ratio_str(r_med, py_med):
        if r_med is None or py_med is None:
            return "N/A"
        # Sub-millisecond medians round to 0 on small matrices; a ratio is then
        # meaningless (and 1/ratio would divide by zero).
        if r_med == 0 or py_med == 0:
            return "N/A (too fast to time)"
        ratio = r_med / py_med
        if ratio >= 1:
            return f"{ratio:.1f}x slower"
        else:
            return f"{1/ratio:.2f}x faster"

    W_OP = 12
    W_PY = 22
    W_R  = 22

    hdr1 = f"{'Operation':<{W_OP}}  {lib_py + ' (s)':<{W_PY}}  {lib_r + ' (s)':<{W_R}}  Ratio (R/Python)"
    hdr2 = f"{''.ljust(W_OP)}  {'median [IQR]':<{W_PY}}  {'median [IQR]':<{W_R}}"
    sep  = "-" * len(hdr1)

    print()
    print(hdr1)
    print(hdr2)
    print(sep)

    for op in ops_order:
        py_vals = groups.get((op, "python", lib_py), [])
        r_vals  = groups.get((op, "r",      lib_r),  [])
        py_cell = cell(py_vals)
        r_cell  = cell(r_vals)
        py_med  = median(py_vals) if py_vals else None
        r_med   = median(r_vals)  if r_vals  else None
        print(f"{op:<{W_OP}}  {py_cell:<{W_PY}}  {r_cell:<{W_R}}  {ratio_str(r_med, py_med)}")

    print(sep)
    print(f"n = {max_rep} repetitions per cell (after warmup runs). IQR = Q3 - Q1.")

    if threads_seen == {1}:
        thread_note = "single thread"
    else:
        thread_note = f"threads = {', '.join(str(t) for t in sorted(threads_seen))}"

    def describe(name, dims):
        rows_, cols_, nnz_ = dims
        return f"{name} ({rows_:,} x {cols_:,}, nnz = {nnz_:,})"

    if len(matrices) == 1:
        (name, dims), = matrices.items()
        print(f"Matrix: {describe(name, dims)}, {thread_note}.")
    else:
        # Multiple matrices in one CSV: the cells above pool their times, so call
        # that out rather than silently mislabeling the table as a single matrix.
        print(f"Matrices (times pooled across all of them), {thread_note}:")
        for name, dims in matrices.items():
            print(f"  - {describe(name, dims)}")
    print()

if __name__ == "__main__":
    main()
