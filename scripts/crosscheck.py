#!/usr/bin/env python3
"""
Cross-check that the R and Python harnesses computed the same results.

Each harness writes a compact digest of every operation's result
(results/python_digests.csv and results/r_digests.csv): n, sum, sumsq, min, max.
These reductions are order-independent, so they match across languages despite
numpy flattening row-major and R column-major.

This script compares the two digests per (matrix_name, operation) and reports
PASS/FAIL. Comparison uses a RELATIVE tolerance, not exact equality: summing
billions of float64 terms accumulates rounding that differs with reduction order
and library, so byte-exact agreement would give false failures. `n` must match
exactly. Exits non-zero if any operation disagrees (or is missing from one side).

Usage: python3 scripts/crosscheck.py [results_dir] [--rtol R] [--atol A]
"""

import argparse
import csv
import sys
from pathlib import Path

FIELDS = ("sum", "sumsq", "min", "max")


def load(path):
    """Return {(matrix_name, operation): {n, sum, sumsq, min, max}}."""
    with open(path, newline="") as f:
        out = {}
        for row in csv.DictReader(f):
            key = (row["matrix_name"], row["operation"])
            out[key] = {
                "n": int(row["n"]),
                **{fld: float(row[fld]) for fld in FIELDS},
            }
        return out


def close(a, b, rtol, atol):
    # Same form as numpy.isclose: |a - b| <= atol + rtol * |b|.
    return abs(a - b) <= atol + rtol * abs(b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent
    p.add_argument("results_dir", nargs="?", type=Path,
                   default=repo_root / "results")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance (default 1e-6)")
    p.add_argument("--atol", type=float, default=1e-9,
                   help="absolute tolerance (default 1e-9)")
    args = p.parse_args()

    py_path = args.results_dir / "python_digests.csv"
    r_path = args.results_dir / "r_digests.csv"
    missing = [str(q) for q in (py_path, r_path) if not q.exists()]
    if missing:
        sys.exit("error: digest file(s) not found: " + ", ".join(missing) +
                 "\n(run both the Python and R harnesses first)")

    py = load(py_path)
    r = load(r_path)

    keys = sorted(set(py) | set(r))
    width = max((len(f"{m}/{o}") for m, o in keys), default=20)

    print()
    print(f"Cross-check (rtol={args.rtol:g}, atol={args.atol:g})")
    print(f"{'matrix/op':<{width}}  result  detail")
    print("-" * (width + 40))

    failures = 0
    for key in keys:
        label = f"{key[0]}/{key[1]}"
        if key not in py or key not in r:
            side = "R" if key not in py else "Python"
            print(f"{label:<{width}}  MISSING from {side}")
            failures += 1
            continue
        dp, dr = py[key], r[key]
        problems = []
        if dp["n"] != dr["n"]:
            problems.append(f"n: py={dp['n']} r={dr['n']}")
        for fld in FIELDS:
            if not close(dp[fld], dr[fld], args.rtol, args.atol):
                rel = abs(dp[fld] - dr[fld]) / (abs(dr[fld]) or 1.0)
                problems.append(f"{fld}: py={dp[fld]:.6g} r={dr[fld]:.6g} "
                                f"(rel={rel:.2e})")
        if problems:
            print(f"{label:<{width}}  FAIL    " + "; ".join(problems))
            failures += 1
        else:
            print(f"{label:<{width}}  PASS")

    print("-" * (width + 40))
    if failures:
        print(f"{failures} mismatch(es) found.")
        sys.exit(1)
    print(f"All {len(keys)} operation(s) agree across R and Python.")


if __name__ == "__main__":
    main()
