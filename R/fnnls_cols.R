#' Fast Non-Negative Least Squares for Multiple Outputs
#'
#' Solves the NNLS problem min ||Y - XB||_F^2 subject to B >= 0 using the Fast Non-Negative Least Squares algorithm of Bro & de Jong (1997).
#'
#' @param Y A numeric matrix of dimensions n x m.
#' @param X A numeric matrix of dimensions n x p.
#' @param tol The convergence tolerance, default is 1e-6.
#' @param max_iter The maximum number of iterations, default is 1000.
#' @param sum_to_constant If TRUE all entries in each column of B sum to 'constant'. Default is FALSE.
#' @param constant If sum_to_constant is TRUE, all entries in each column sum to this number. The default value is 1.
#' @param lower_bound If TRUE all entries bounded below by 'lb', otherwise they are nonnegative. The default value is FALSE.
#' @param lb If lower_bound is TRUE all entries are bounded below by 'lb'. The default value is 0.
#' @param parallel If TRUE, the columns of B are computed in parallel. The default value is FALSE.
#' @param ncores If parallel is TRUE, this many cores are used in the parallel computations. Must be positive integer. The default value is -1 (use all available cores).
#'
#' @return A list with two elements:
#' \itemize{
#'   \item \code{B}: A non-negative numeric matrix of dimensions p x m with the estimated coefficients.
#'   \item \code{mse}: A numeric vector of length m with the mean squared error for each output column.
#' }
#'
#' @examples
#' n <- 50
#' p <- 10
#' m <- 3
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' Y <- matrix(runif(n * m, min = 0, max = 10), nrow = n, ncol = m)
#' result <- fnnls_cols(Y, X, tol = 1e-8, max_iter = 1000)
#' result$B
#' result$mse
#'
#' @references
#' Bro, Rasmus & Jong, Sijmen. (1997). A Fast Non-negativity-constrained Least Squares Algorithm.
#' Journal of Chemometrics. 11. 393-401. 10.1002/(SICI)1099-128X(199709/10)11:53.0.CO;2-L.
#'
#' @export
fnnls_cols <- function(Y, X, tol = 1e-6, max_iter = 1000, sum_to_constant = FALSE, constant = 1.0, lower_bound = FALSE, lb = 0.0, parallel = FALSE, ncores = -1) {
  if (!is.matrix(Y)) stop("Y must be a matrix")
  if (!is.matrix(X)) stop("X must be a matrix")
  if (nrow(X) != nrow(Y)) stop("Incompatible Y and X matrices")
  if (tol < 0) stop("Tolerance must be small positive number")
  if (max_iter < 1) stop("Maximum iterations must be a positive integer")

  B <- fnnls_cols_cpp(Y, X, tol, max_iter, sum_to_constant, constant, lower_bound, lb, parallel, ncores)
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
