fnnls_reg <- function(y, X, tol = 1e-6, max_iter = 1000, sum_to_constant = FALSE, constant = 1, lower_bound = FALSE, lb = 0) {
  if (!is.matrix(X)) stop("X must be a matrix")
  if (!is.numeric(y)) stop("y must be numeric")
  if (tol < 0) stop("Tolerance must be small positive number")
  b <- fnnls_reg_cpp(y, X, tol, max_iter, sum_to_constant, constant, lower_bound, lb)
  mse <- mean((y - X %*% b)^2)
  list(b = b, mse = mse)
}
