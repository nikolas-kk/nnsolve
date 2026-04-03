fnnls_regs <- function(Y, X, tol = 1e-6, max_iter = 1000, sum_to_constant = FALSE, constant = 1.0, lower_bound = FALSE, lb = 0.0, parallel = FALSE, ncores = -1) {
  if (!is.matrix(Y)) stop("Y must be a matrix")
  if (!is.matrix(X)) stop("X must be a matrix")
  if (nrow(X) != nrow(Y)) stop("Incompatible Y and X matrices")
  if (tol < 0) stop("Tolerance must be small positive number")
  if (max_iter < 1) stop("Maximum iterations must be a positive integer")

  B <- fnnls_regs_cpp(Y, X, tol, max_iter, sum_to_constant, constant, lower_bound, lb, parallel, ncores)
  mse <- Rfast::colmeans((Y - X %*% B)^2)

  if (!is.null(colnames(X))) {
    rownames(B) <- colnames(X)
  }
  if (!is.null(colnames(Y))) {
    colnames(B) <- colnames(Y)
    names(mse) <- colnames(Y)
  }

  list(B = B, mse = mse)
}
