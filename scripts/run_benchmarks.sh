#!/usr/bin/env bash
#
# Run the Python and R sparse-matrix benchmarks and merge their outputs.
#
# Reads the shared config (configs/bench.json), pins thread counts in the
# environment BEFORE launching either interpreter (so we measure sparse kernels,
# not multithreaded BLAS), runs both harnesses, then concatenates their tidy CSVs
# into results/all_results.csv for comparison.
#
# Any extra arguments are forwarded verbatim to both harnesses, e.g.:
#   scripts/run_benchmarks.sh --matrices small,medium
#   scripts/run_benchmarks.sh --config /path/to/other.json
#
# Flags consumed by this script (must come before forwarded args):
#   --python-only   run only the Python harness
#   --r-only        run only the R harness
#   --config PATH   config file (also forwarded to the harnesses)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${REPO_ROOT}/configs/bench.json"
RUN_PYTHON=1
RUN_R=1

# Peek at flags we care about without consuming the rest (those go to harnesses).
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-only) RUN_PYTHON=1; RUN_R=0; shift ;;
    --r-only)      RUN_PYTHON=0; RUN_R=1; shift ;;
    --config)      CONFIG="$2"; FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
    *)             FORWARD_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "error: config not found: ${CONFIG}" >&2
  exit 1
fi

# Pull thread count and output dir from the config (python3 is a hard dep of the
# Python harness anyway, so it's safe to rely on here for parsing).
THREADS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"]["threads"])' "${CONFIG}")"
OUT_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"]["dir"])' "${CONFIG}")"
# Resolve OUT_DIR relative to repo root if it isn't absolute.
[[ "${OUT_DIR}" = /* ]] || OUT_DIR="${REPO_ROOT}/${OUT_DIR}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
echo "[run] threads pinned to ${THREADS}; output dir ${OUT_DIR}"

mkdir -p "${OUT_DIR}"
PY_CSV="${OUT_DIR}/python_results.csv"
R_CSV="${OUT_DIR}/r_results.csv"
ALL_CSV="${OUT_DIR}/all_results.csv"

if [[ "${RUN_PYTHON}" -eq 1 ]]; then
  echo "[run] === Python ==="
  python3 "${REPO_ROOT}/src/bench.py" "${FORWARD_ARGS[@]}"
fi

if [[ "${RUN_R}" -eq 1 ]]; then
  echo "[run] === R ==="
  if command -v Rscript >/dev/null 2>&1; then
    Rscript "${REPO_ROOT}/src/bench.R" "${FORWARD_ARGS[@]}"
  else
    echo "[run] warning: Rscript not found on PATH; skipping R benchmark" >&2
    RUN_R=0
  fi
fi

# Merge: take the header from the first CSV present, then append data rows from
# each. Both harnesses write the same column schema, so this is a plain stack.
echo "[run] merging results into ${ALL_CSV}"
: > "${ALL_CSV}"
header_written=0
for csv in "${PY_CSV}" "${R_CSV}"; do
  [[ -f "${csv}" ]] || continue
  if [[ "${header_written}" -eq 0 ]]; then
    head -n 1 "${csv}" > "${ALL_CSV}"
    header_written=1
  fi
  tail -n +2 "${csv}" >> "${ALL_CSV}"
done

if [[ "${header_written}" -eq 0 ]]; then
  echo "[run] warning: no result CSVs found to merge" >&2
else
  rows="$(($(wc -l < "${ALL_CSV}") - 1))"
  echo "[run] done: ${rows} data rows in ${ALL_CSV}"
fi
