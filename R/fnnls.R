fnnls <- function(XtX, Xty, tol = 1e-6, max_iter = 1000, sum_to_constant = FALSE, constant = 1, lower_bound = FALSE, lb = 0) {
  if (!is.matrix(XtX)) stop("XtX must be a matrix")
  if (!is.numeric(Xty)) stop("Xty must be numeric")
  if (nrow(XtX) != ncol(XtX)) stop("XtX must be square")
  if (nrow(XtX) != length(Xty)) stop("XtX and Xty dimensions must match")
  if (tol < 0) stop("Tolerance must be small positive number")
  fnnls_cpp(XtX, Xty, tol, max_iter, sum_to_constant, constant, lower_bound, lb)
}
