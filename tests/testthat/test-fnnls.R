# =============================================================================
# test-fnnls.R  –  testthat suite for fnnls()
# Covers: basic NNLS, input validation, lower_bound, sum_to_constant
# =============================================================================
library(testthat)
library(nnsolve)

# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------
make_nnls_problem <- function(k, D, seed = 42) {
  set.seed(seed)
  H   <- matrix(rnorm(k * D), nrow = k, ncol = D)
  x   <- rnorm(D)
  XtX <- H %*% t(H) + diag(1e-8, k)
  Xty <- as.vector(H %*% x)
  list(XtX = XtX, Xty = Xty)
}

# =============================================================================
# 1.  CORE NNLS (original tests)
# =============================================================================

test_that("fnnls returns non-negative solution", {
  p <- make_nnls_problem(10, 100)
  w <- fnnls(p$XtX, p$Xty)
  expect_true(all(w >= 0))
})

test_that("fnnls returns vector of correct length", {
  k <- 10
  p <- make_nnls_problem(k, 100)
  w <- fnnls(p$XtX, p$Xty)
  expect_length(w, k)
})

test_that("fnnls solution satisfies KKT conditions", {
  p <- make_nnls_problem(10, 100)
  w <- fnnls(p$XtX, p$Xty)
  g <- p$Xty - p$XtX %*% w          # gradient at solution
  expect_true(all(w >= 0))
  expect_true(all(g[w == 0]  <= 1e-6))
  expect_true(all(abs(g[w > 0]) <= 1e-6))
})

test_that("fnnls handles all-zero solution", {
  k   <- 5
  XtX <- diag(k)
  Xty <- rep(-1, k)
  w   <- fnnls(XtX, Xty)
  expect_true(all(w == 0))
})

test_that("fnnls handles unconstrained solution that is already non-negative", {
  k   <- 5
  XtX <- diag(k)
  Xty <- c(1, 2, 3, 4, 5)
  w   <- fnnls(XtX, Xty)
  expect_equal(w, Xty, tolerance = 1e-10)
})

test_that("fnnls handles k = 1", {
  XtX <- matrix(4.0, 1, 1)
  Xty <- 2.0
  w   <- fnnls(XtX, Xty)
  expect_length(w, 1)
  expect_true(w >= 0)
  expect_equal(w, 0.5, tolerance = 1e-10)
})

test_that("fnnls handles large k", {
  p <- make_nnls_problem(k = 100, D = 1000)
  w <- fnnls(p$XtX, p$Xty)
  expect_length(w, 100)
  expect_true(all(w >= 0))
})

test_that("fnnls is consistent across multiple runs", {
  p  <- make_nnls_problem(10, 100)
  w1 <- fnnls(p$XtX, p$Xty)
  w2 <- fnnls(p$XtX, p$Xty)
  expect_equal(w1, w2)
})

# =============================================================================
# 2.  INPUT VALIDATION (original tests)
# =============================================================================

test_that("fnnls input validation — XtX not a matrix", {
  expect_error(fnnls(c(1, 2, 3), c(1, 2, 3)), "XtX must be a matrix")
})

test_that("fnnls input validation — Xty not numeric", {
  XtX <- diag(3)
  expect_error(fnnls(XtX, c("a", "b", "c")), "Xty must be numeric")
})

test_that("fnnls input validation — XtX not square", {
  expect_error(fnnls(matrix(1:6, 2, 3), c(1, 2)), "XtX must be square")
})

test_that("fnnls input validation — dimension mismatch", {
  expect_error(fnnls(diag(3), c(1, 2)), "XtX and Xty dimensions must match")
})

# =============================================================================
# 3.  LOWER BOUND
# =============================================================================

test_that("lower_bound: all solution entries >= lb", {
  p <- make_nnls_problem(10, 100)
  lb <- 0.05
  w  <- fnnls(p$XtX, p$Xty, lower_bound = TRUE, lb = lb)
  expect_true(all(w >= lb - 1e-9))
})

test_that("lower_bound: solution length is unchanged", {
  k  <- 10
  p  <- make_nnls_problem(k, 100)
  w  <- fnnls(p$XtX, p$Xty, lower_bound = TRUE, lb = 0.1)
  expect_length(w, k)
})

test_that("lower_bound: lb = 0 matches plain NNLS", {
  p  <- make_nnls_problem(10, 100)
  w0 <- fnnls(p$XtX, p$Xty)
  w1 <- fnnls(p$XtX, p$Xty, lower_bound = TRUE, lb = 0.0)
  expect_equal(w0, w1, tolerance = 1e-8)
})

test_that("lower_bound: solution shifts correctly relative to standard NNLS", {
  # With a diagonal system, the closed-form shifted solution is known exactly.
  k   <- 5
  XtX <- diag(k)
  Xty <- c(1, 2, 3, 4, 5)
  lb  <- 0.5
  w   <- fnnls(XtX, Xty, lower_bound = TRUE, lb = lb)
  # unconstrained optimum is Xty; all > lb, so solution should equal Xty
  expect_equal(w, Xty, tolerance = 1e-8)
  expect_true(all(w >= lb - 1e-9))
})

test_that("lower_bound: forces entries that would be 0 up to lb", {
  # Xty is negative -> plain NNLS gives 0; with lb > 0 should give lb
  k   <- 5
  XtX <- diag(k)
  Xty <- rep(-1, k)
  lb  <- 0.2
  w   <- fnnls(XtX, Xty, lower_bound = TRUE, lb = lb)
  expect_true(all(w >= lb - 1e-9))
})

test_that("lower_bound: negative lb triggers error", {
  p <- make_nnls_problem(5, 50)
  expect_error(
    fnnls(p$XtX, p$Xty, lower_bound = TRUE, lb = -0.1),
    "Lower bound cannot be negative"
  )
})

test_that("lower_bound: works with k = 1", {
  XtX <- matrix(4.0, 1, 1)
  Xty <- 2.0
  w   <- fnnls(XtX, Xty, lower_bound = TRUE, lb = 0.1)
  expect_length(w, 1)
  expect_true(w >= 0.1 - 1e-9)
  expect_equal(w, 0.5, tolerance = 1e-8)   # optimum (0.5) > lb (0.1)
})

# =============================================================================
# 4.  SUM-TO-CONSTANT
# =============================================================================

test_that("sum_to_constant: entries sum to constant (default = 1)", {
  p <- make_nnls_problem(10, 100)
  w <- fnnls(p$XtX, p$Xty, sum_to_constant = TRUE)
  expect_equal(sum(w), 1.0, tolerance = 1e-6)
})

test_that("sum_to_constant: entries sum to specified constant", {
  p        <- make_nnls_problem(10, 100)
  const    <- 3.5
  w        <- fnnls(p$XtX, p$Xty, sum_to_constant = TRUE, constant = const)
  expect_equal(sum(w), const, tolerance = 1e-6)
})

test_that("sum_to_constant: all entries remain non-negative", {
  p <- make_nnls_problem(10, 100)
  w <- fnnls(p$XtX, p$Xty, sum_to_constant = TRUE, constant = 1.0)
  expect_true(all(w >= -1e-9))
})

test_that("sum_to_constant: solution has correct length", {
  k <- 10
  p <- make_nnls_problem(k, 100)
  w <- fnnls(p$XtX, p$Xty, sum_to_constant = TRUE)
  expect_length(w, k)
})

test_that("sum_to_constant: non-positive constant triggers error", {
  p <- make_nnls_problem(5, 50)
  expect_error(
    fnnls(p$XtX, p$Xty, sum_to_constant = TRUE, constant = 0),
    "Non-negative entries cannot sum to a non-positive constant"
  )
  expect_error(
    fnnls(p$XtX, p$Xty, sum_to_constant = TRUE, constant = -1),
    "Non-negative entries cannot sum to a non-positive constant"
  )
})

test_that("sum_to_constant: k = 1 trivially sums to constant", {
  XtX   <- matrix(4.0, 1, 1)
  Xty   <- 2.0
  const <- 2.0
  w     <- fnnls(XtX, Xty, sum_to_constant = TRUE, constant = const)
  expect_length(w, 1)
  expect_equal(sum(w), const, tolerance = 1e-6)
})

test_that("sum_to_constant: consistent across multiple runs", {
  p  <- make_nnls_problem(10, 100)
  w1 <- fnnls(p$XtX, p$Xty, sum_to_constant = TRUE)
  w2 <- fnnls(p$XtX, p$Xty, sum_to_constant = TRUE)
  expect_equal(w1, w2)
})

# =============================================================================
# 5.  LOWER BOUND + SUM-TO-CONSTANT (combined)
# =============================================================================

test_that("combined: all entries >= lb AND sum to constant", {
  p     <- make_nnls_problem(10, 100)
  lb    <- 0.02
  const <- 1.0
  w     <- fnnls(p$XtX, p$Xty,
                 sum_to_constant = TRUE, constant = const,
                 lower_bound     = TRUE,  lb       = lb)
  expect_true(all(w >= lb - 1e-9))
  expect_equal(sum(w), const, tolerance = 1e-6)
})

test_that("combined: solution has correct length", {
  k  <- 10
  p  <- make_nnls_problem(k, 100)
  w  <- fnnls(p$XtX, p$Xty,
              sum_to_constant = TRUE, lower_bound = TRUE, lb = 0.01)
  expect_length(w, k)
})

test_that("combined: negative lb still triggers error", {
  p <- make_nnls_problem(5, 50)
  expect_error(
    fnnls(p$XtX, p$Xty,
          sum_to_constant = TRUE, constant = 1,
          lower_bound     = TRUE, lb       = -0.1),
    "Lower bound cannot be negative"
  )
})

test_that("combined: non-positive constant still triggers error", {
  p <- make_nnls_problem(5, 50)
  expect_error(
    fnnls(p$XtX, p$Xty,
          sum_to_constant = TRUE, constant = -1,
          lower_bound     = TRUE, lb       = 0.0),
    "Non-negative entries cannot sum to a non-positive constant"
  )
})