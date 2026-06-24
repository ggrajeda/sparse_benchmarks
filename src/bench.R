#!/usr/bin/env Rscript
# R sparse-matrix benchmark harness.
#
# Reads the shared configs/bench.json, deterministically builds each matrix as an
# SVT_SparseArray (via vectorized generation from the modular map in the config),
# then times each operation `warmup`+`repetitions` times and appends the results
# as tidy long-format rows to results/r_results.csv.
#
# The logical matrix is identical to what the Python harness builds from the same
# config (R uses 1-based coordinates, Python 0-based, but the entries match).
# SVT_SparseArray is column-compressed; the Python harness stores CSC to match, so
# both libraries are column-native and every operation is an apples-to-apples
# comparison. After timing, this harness also writes a compact, order-independent
# digest of every operation's result (results/r_digests.csv); scripts/crosscheck.py
# compares it against Python's to confirm both compute the same thing.
#
# NOTE on threads: the runner pins OMP/OPENBLAS/MKL_NUM_THREADS in the environment
# before launching R -- for reproducibility and to make sure CPU time measures the
# true compute time -- and we record the configured value here. Pinning to 1 is
# required by the timing: time_op measures CPU time (proc.time user+sys), which
# sums across threads, so it only equals true compute time when single-threaded.

suppressPackageStartupMessages({
  library(jsonlite)        # parse the shared config
  library(bit64)           # exact 64-bit integer arithmetic for the generator
  library(SparseArray)     # SVT_SparseArray
})

# Number of dense weight vectors for the vecmat_multi (dense x sparse) benchmark --
# e.g. several gene signatures scored at once. Must match Python's N_VECS so the
# two harnesses do the same work.
N_VECS <- 16L

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
  while (b > 0) { tt <- b; b <- a %% b; a <- tt }
  a
}

# Modular inverse of a mod m via the extended Euclidean algorithm. Both a and m
# are integer-valued doubles in (0, 2^31); result is a double x with
# (a * x) %% m == 1. Requires gcd(a, m) == 1 (guaranteed here: gcd(stride, cols)
# divides gcd(stride, rows*cols) == 1).
modinv_dbl <- function(a, m) {
  a <- a %% m
  m0 <- m; x0 <- 0; x1 <- 1
  while (a > 1) {
    q  <- a %/% m
    tt <- m;  m  <- a %% m; a  <- tt
    tt <- x0; x0 <- x1 - q * x0; x1 <- tt
  }
  if (x1 < 0) x1 <- x1 + m0
  x1
}

# Build the deterministic matrix as an SVT_SparseArray. We avoid the COO
# intermediate (whose nzcoo matrix requires nrow == nnz, overflowing R's 32-bit
# matrix dimension limit when nnz > 2^31 - 1).
#
# We process the columns in BLOCKS, generating each block's entries directly from
# the modular map rather than scanning all i. The key identity: col_i =
# (i*stride) %% cols, so the smallest i landing in column j is
# i_start(j) = (j * stride_inv) %% cols (stride_inv = stride^{-1} mod cols), and
# the rest are i_start(j) + cols, + 2*cols, ... < nnz. For a block of columns we
# build i_start and the per-column counts vectorized, expand to per-entry vectors,
# compute k = (i*stride) %% (rows*cols), off = k %/% cols, val = k %% value_mod + 1,
# then sort by (col, off) and slice into one leaf list(values, offsets) per column.
#
# Blocking serves two purposes: it bounds peak memory, and -- crucially -- it
# keeps each block's entry count under 2^31 so order(method="radix") works (R's
# radix sort does NOT support long vectors, which a single global sort of all
# nnz > 2^31 entries would require).
generate_svt <- function(rows, cols, nnz, stride, value_mod) {
  mn_dbl <- rows * cols
  g <- gcd_dbl(stride, mn_dbl)
  if (g != 1) {
    stop(sprintf("stride %.0f is not coprime to rows*cols (%.0f); gcd=%.0f. ",
                 stride, mn_dbl, g),
         "The k_i would collide and produce duplicate coordinates.")
  }
  if (nnz > mn_dbl) stop(sprintf("nnz (%.0f) exceeds rows*cols (%.0f).", nnz, mn_dbl))

  nnz  <- as.double(nnz)
  cols <- as.double(cols)

  stride64    <- as.integer64(stride)
  cols64      <- as.integer64(cols)
  value_mod64 <- as.integer64(value_mod)
  mn64        <- as.integer64(rows) * as.integer64(cols)
  stride_inv  <- modinv_dbl(stride %% cols, cols)

  # Columns per block: keep block entries (~ block_w * nnz/cols) under ~1e9, well
  # below the 2^31 long-vector limit for order().
  per_col <- nnz / cols
  block_w <- max(1, floor(1e9 / max(per_col, 1)))

  SVT <- vector("list", as.integer(cols))

  j_lo <- 0
  while (j_lo < cols) {
    j_hi    <- min(j_lo + block_w, cols)           # exclusive upper column bound
    J       <- seq.int(j_lo, j_hi - 1)             # 0-based columns in this block
    i_start <- (J * stride_inv) %% cols            # first i in each column (exact < 2^53)
    keep    <- i_start < nnz                       # drop columns with no entries
    if (any(keep)) {
      J       <- as.integer(J[keep])
      i_start <- i_start[keep]
      n_in    <- floor((nnz - i_start - 1) / cols) + 1     # entries per column
      col_e   <- rep(J, n_in)                              # 0-based column per entry
      i_e     <- rep(i_start, n_in) + (sequence(n_in) - 1) * cols
      k       <- (as.integer64(i_e) * stride64) %% mn64
      off_e   <- as.integer(k %/% cols64)                  # 0-based row offsets
      val_e   <- as.double(k %% value_mod64) + 1
      rm(i_e, k)

      # Sort by (col, off); col_e is already ascending, so this offset-sorts
      # within each column. Block has < 2^31 entries, so radix is fine.
      ord   <- order(col_e, off_e, method = "radix")
      col_s <- col_e[ord]; off_s <- off_e[ord]; val_s <- val_e[ord]
      rm(col_e, off_e, val_e, ord)

      # Slice each column's contiguous run into a leaf list(values, offsets).
      n_ent  <- length(col_s)
      starts <- c(1L, which(col_s[-1L] != col_s[-n_ent]) + 1L)
      ends   <- c(starts[-1L] - 1L, n_ent)
      grp    <- col_s[starts]                              # 0-based column id of each run
      for (gi in seq_along(starts)) {
        rng <- starts[gi]:ends[gi]
        SVT[[grp[gi] + 1L]] <- list(val_s[rng], off_s[rng])
      }
    }
    j_lo <- j_hi
  }

  # dim must be an integer vector; rows/cols are < 2^31 so as.integer is safe.
  SparseArray:::new_SVT_SparseArray(
    dim   = as.integer(c(rows, cols)),
    type  = "double",
    SVT   = SVT,
    check = FALSE
  )
}

# Operations to benchmark, as zero-arg closures over the SVT matrix M. Each
# returns a realized result so timing reflects real work.
#
# These are LEFT-multiplies (vecmat: dense %*% sparse) -- the fast path for a
# column-compressed store, and the genomics operation of interest: with rows =
# genes and cols = cells, `x %*% M` is a per-cell weighted sum of genes (a
# signature score per cell), and `X %*% M` scores N_VECS signatures at once.
#
# The operands are deterministic but NON-constant (a function of the 0-based row
# / gene index and, for vecmat_multi, the signature index), built identically in
# the Python harness. An all-ones operand would make x %*% M reproduce column
# sums, so the cross-check could not distinguish a transposed/permuted matrix
# from a correct one; varying the operand gives the digest real discriminating
# power.
make_operations <- function(M, value_mod) {
  nrow_M <- nrow(M)
  g0 <- as.double(0:(nrow_M - 1L))                   # 0-based row (gene) index
  x  <- matrix(1 + (g0 %% value_mod), nrow = 1L)     # vecmat operand: 1 x nrow
  c0 <- as.double(0:(N_VECS - 1L))
  X  <- 1 + (outer(c0, g0, "+") %% value_mod)        # vecmat_multi operand: N_VECS x nrow
  list(
    vecmat       = function() x %*% M,
    vecmat_multi = function() X %*% M,
    row_sums     = function() rowSums(M),
    col_sums     = function() colSums(M),
    sum          = function() sum(M)
  )
}

# Compact, order-independent summary of an operation's result. The reductions are
# invariant to element order, so they match Python despite numpy flattening
# row-major and R column-major. sumsq uses (t)crossprod (BLAS) so no large
# temporary is materialized: the vecmat results are short and wide (1 x cols or
# N_VECS x cols), so we contract along the SHORT dimension via tcrossprod.
digest <- function(result) {
  n <- length(result)
  d <- dim(result)
  if (!is.null(d)) {
    sumsq <- if (d[1L] <= d[2L]) sum(diag(tcrossprod(result)))
             else                sum(diag(crossprod(result)))
  } else {
    sumsq <- sum(as.numeric(crossprod(as.numeric(result))))  # vector -> 1x1
  }
  list(
    n     = n,
    sum   = sum(result),
    sumsq = sumsq,
    min   = if (n > 0) min(result) else 0,
    max   = if (n > 0) max(result) else 0
  )
}

time_op <- function(fn, warmup, reps) {
  for (i in seq_len(warmup)) invisible(fn())
  times <- numeric(reps)
  for (r in seq_len(reps)) {
    # proc.time() user.self + sys.self is the CPU time (user + system) of this
    # process, not wall clock. With threads pinned to 1 this tracks the kernel's
    # actual compute and excludes time the process spent descheduled, so it is
    # robust to OS scheduling and contention from other processes. This matches
    # Python's process_time() so the two harnesses measure the same kind of time.
    t0 <- proc.time()
    res <- fn()
    dt <- proc.time() - t0
    times[r] <- dt[["user.self"]] + dt[["sys.self"]]
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
  digest_path <- file.path(out_dir, "r_digests.csv")
  columns  <- config$output$columns
  lib_version <- as.character(packageVersion("SparseArray"))

  results <- vector("list", 0L)
  digests <- vector("list", 0L)

  for (mi in seq_len(nrow(matrices))) {
    name <- matrices$name[mi]
    rows <- as.double(matrices$rows[mi])
    cols <- as.double(matrices$cols[mi])
    nnz  <- as.double(matrices$nnz[mi])
    cat(sprintf("[r] building '%s' (%.0fx%.0f, nnz=%.0f) ...\n", name, rows, cols, nnz))

    M <- generate_svt(rows, cols, nnz, stride, value_mod)
    ops <- make_operations(M, value_mod)

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
      # One extra (untimed) eval to digest the result for the cross-check.
      d <- digest(ops[[op]]())
      digests[[length(digests) + 1L]] <- data.frame(
        language    = "r",
        matrix_name = name,
        operation   = op,
        n           = d$n,
        sum         = d$sum,
        sumsq       = d$sumsq,
        min         = d$min,
        max         = d$max,
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

  dout <- do.call(rbind, digests)
  # n as a plain integer string (can exceed 2^31 for vecmat_multi); floats at full precision.
  dout$n     <- sprintf("%.0f", as.double(dout$n))
  for (col in c("sum", "sumsq", "min", "max")) dout[[col]] <- sprintf("%.17g", dout[[col]])
  write.csv(dout, digest_path, row.names = FALSE, quote = FALSE)
  cat(sprintf("[r] wrote %s\n", digest_path))
}

main()
