#!/usr/bin/env Rscript
# R sparse-matrix benchmark harness.
#
# Reads the shared configs/bench.json, deterministically builds each matrix as a
# COO_SparseArray (via the incremental modular map documented in the config),
# converts it once (untimed) to an SVT_SparseArray for benchmarking, then times
# each operation `warmup`+`repetitions` times and appends the results as tidy
# long-format rows to results/r_results.csv.
#
# The logical matrix is identical to what the Python harness builds from the same
# config (R uses 1-based coordinates, Python 0-based, but the entries match).
#
# NOTE on threads: SparseArray's kernels are largely single-threaded C, but %*%
# may hit BLAS. R cannot reliably change BLAS thread count after startup, so the
# runner sets OMP/OPENBLAS/MKL_NUM_THREADS in the environment before launching R.
# We only record the configured value here.

suppressPackageStartupMessages({
  library(jsonlite)        # parse the shared config
  library(bit64)           # exact 64-bit integer arithmetic for the generator
  library(SparseArray)     # COO_SparseArray / SVT_SparseArray
})

# Dense columns for the SpMM (sparse x dense) benchmark. Must match Python's
# SPMM_COLS so the two harnesses do the same work.
SPMM_COLS <- 16L

# Avoid scientific notation so nnz (> 2^31) is written as a plain integer string.
options(scipen = 999)

# ---- minimal CLI parsing: --config <path> and --matrices a,b,c ----
parse_cli <- function(args) {
  out <- list(config = NULL, matrices = NULL)
  i <- 1L
  while (i <= length(args)) {
    if (args[i] == "--config" && i < length(args)) {
      out$config <- args[i + 1L]; i <- i + 2L
    } else if (args[i] == "--matrices" && i < length(args)) {
      out$matrices <- strsplit(args[i + 1L], ",")[[1]]; i <- i + 2L
    } else {
      i <- i + 1L
    }
  }
  out
}

# Euclid's algorithm on doubles. Inputs are integer-valued and < 2^53, so the
# modulo stays exact; avoids needing bit64 just for the coprimality check.
gcd_dbl <- function(a, b) {
  a <- abs(a); b <- abs(b)
  while (b > 0) { t <- b; b <- a %% b; a <- t }
  a
}

# Build the deterministic matrix as a COO_SparseArray.
#
# k_i = (i * stride) %% (rows*cols), row_i = k_i %/% cols, col_i = k_i %% cols,
# v_i = 1 + (k_i %% value_mod). All index arithmetic is in bit64::integer64 so it
# stays exact past 2^53 (plain doubles would silently corrupt). Coordinates are
# converted to 1-based int for SparseArray. Generation is chunked to bound the
# size of the integer64 temporaries.
generate_coo <- function(rows, cols, nnz, stride, value_mod, chunk = 50e6) {
  mn_dbl <- rows * cols
  g <- gcd_dbl(stride, mn_dbl)
  if (g != 1) {
    stop(sprintf("stride %.0f is not coprime to rows*cols (%.0f); gcd=%.0f. ",
                 stride, mn_dbl, g),
         "The k_i would collide and produce duplicate coordinates.")
  }
  if (nnz > mn_dbl) stop(sprintf("nnz (%.0f) exceeds rows*cols (%.0f).", nnz, mn_dbl))

  nnz <- as.double(nnz)
  chunk <- as.double(chunk)
  stride64    <- as.integer64(stride)
  cols64      <- as.integer64(cols)
  value_mod64 <- as.integer64(value_mod)
  mn64        <- as.integer64(rows) * as.integer64(cols)

  # Preallocate; nzcoo is an integer (row, col) matrix as required by the
  # COO_SparseArray constructor. These are long vectors when nnz > 2^31.
  nzcoo  <- matrix(NA_integer_, nrow = nnz, ncol = 2L)
  nzdata <- double(nnz)

  i0 <- 0
  while (i0 < nnz) {
    n  <- min(chunk, nnz - i0)
    # Global indices i0 .. i0+n-1 as integer64 (0-based).
    i  <- as.integer64(i0) + as.integer64(seq.int(0, n - 1))
    k  <- (i * stride64) %% mn64            # exact: i*stride < 2^63
    # Coordinates: 0-based -> 1-based. k %/% cols < rows < 2^31, so as.integer ok.
    ridx <- as.integer(k %/% cols64) + 1L
    cidx <- as.integer(k %% cols64) + 1L
    vals <- as.double(k %% value_mod64) + 1

    sel <- (i0 + 1):(i0 + n)               # 1-based destination slice
    nzcoo[sel, 1L] <- ridx
    nzcoo[sel, 2L] <- cidx
    nzdata[sel]    <- vals
    i0 <- i0 + n
  }

  # dim must be an integer vector; rows/cols are < 2^31 so as.integer is safe.
  COO_SparseArray(dim = as.integer(c(rows, cols)), nzcoo = nzcoo, nzdata = nzdata)
}

# Operations to benchmark, as zero-arg closures over the SVT matrix M. Each
# returns a realized result so timing reflects real work.
make_operations <- function(M) {
  ncol_M <- ncol(M)
  x <- matrix(1, nrow = ncol_M, ncol = 1L)          # spmv operand (dense)
  X <- matrix(1, nrow = ncol_M, ncol = SPMM_COLS)    # spmm operand (dense)
  list(
    spmv     = function() M %*% x,
    spmm     = function() M %*% X,
    row_sums = function() rowSums(M),
    col_sums = function() colSums(M),
    sum      = function() sum(M)
  )
}

time_op <- function(fn, warmup, reps) {
  for (i in seq_len(warmup)) invisible(fn())
  times <- numeric(reps)
  for (r in seq_len(reps)) {
    t0 <- Sys.time()
    res <- fn()
    times[r] <- as.numeric(Sys.time() - t0, units = "secs")
    rm(res)
  }
  times
}

main <- function() {
  cli <- parse_cli(commandArgs(trailingOnly = TRUE))
  repo_root <- normalizePath(file.path(dirname(sub("^--file=", "",
                  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
  config_path <- if (!is.null(cli$config)) cli$config else
                   file.path(repo_root, "configs", "bench.json")
  config <- fromJSON(config_path)

  run       <- config$run
  gen       <- config$generation
  threads   <- as.integer(run$threads)
  warmup    <- as.integer(run$warmup)
  reps      <- as.integer(run$repetitions)
  stride    <- as.double(gen$stride)
  value_mod <- as.double(gen$value_mod)
  op_names  <- config$operations

  matrices <- config$matrices            # data.frame: name, rows, cols, nnz
  if (!is.null(cli$matrices)) {
    matrices <- matrices[matrices$name %in% cli$matrices, , drop = FALSE]
    if (nrow(matrices) == 0L) stop("No configured matrices match the requested subset.")
  }

  out_dir <- config$output$dir
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  out_path <- file.path(out_dir, "r_results.csv")
  columns  <- config$output$columns
  lib_version <- as.character(packageVersion("SparseArray"))

  results <- vector("list", 0L)

  for (mi in seq_len(nrow(matrices))) {
    name <- matrices$name[mi]
    rows <- as.double(matrices$rows[mi])
    cols <- as.double(matrices$cols[mi])
    nnz  <- as.double(matrices$nnz[mi])
    cat(sprintf("[r] building '%s' (%.0fx%.0f, nnz=%.0f) ...\n", name, rows, cols, nnz))

    coo <- generate_coo(rows, cols, nnz, stride, value_mod)
    M <- as(coo, "SVT_SparseArray")        # untimed conversion for benchmarking
    rm(coo)
    ops <- make_operations(M)

    for (op in op_names) {
      if (is.null(ops[[op]])) {
        cat(sprintf("[r]   skipping unknown op '%s'\n", op)); next
      }
      cat(sprintf("[r]   timing %s ...\n", op))
      times <- time_op(ops[[op]], warmup, reps)
      results[[length(results) + 1L]] <- data.frame(
        language        = "r",
        library         = "SparseArray",
        library_version = lib_version,
        matrix_name     = name,
        operation       = op,
        rows            = rows,
        cols            = cols,
        nnz             = nnz,
        threads         = threads,
        rep             = seq_len(reps),
        time_sec        = times,
        stringsAsFactors = FALSE
      )
    }
    rm(M, ops); gc(verbose = FALSE)
  }

  out <- do.call(rbind, results)[, columns, drop = FALSE]
  # time_sec with full precision; everything else plain (scipen disables sci notation).
  out$time_sec <- sprintf("%.9f", out$time_sec)
  write.csv(out, out_path, row.names = FALSE, quote = FALSE)
  cat(sprintf("[r] wrote %s\n", out_path))
}

main()
