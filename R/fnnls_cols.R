#' Solves the NNLS problem min ||Y - XB||^2 subject to B >= 0 using the Fast Non-Negative Least Squares algorithm of Bro & de Jong (1997).
#'
#' @param Y The matrix of target variables.
#' @param X The matrix of predictors.
#' @param tol The convergence tolerance, default is 1e-6.
#' @param max_iter The maximum number of iterations, default is 1000.
#' @param sum_to_constant If TRUE all entries sum to 'constant', Default is FALSE.
#' @param constant If sum_to_constant is TRUE, all entries sum to this number. The default value is 1.
#' @param lower_bound If TRUE all entries bounded below by 'lb', otherwise they are nonnegative. The default value is FALSE.
#' @param lb If lower_bound is TRUE all entries are bounded below by 'lb'. The default value is 0.
#' @param parallel If TRUE, the columns of B are computed in parallel. The default value is FALSE.
#' @param ncores IF parallel is TRUE, ncores are used in the parallel computations.Must be positive integer. The default value is -1.
#'
#' @return A non-negative numeric matrix with the estimated coefficients.
#'
#' @examples
#' n <- 50
#' p <- 10
#' m <- 3
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' Y <- matrix(runif(n * m, min = 0, max = 10), nrow = n, ncol = m)
#' B <- fnnls_cols(Y, X, tol = 1e-8, max_iter = 1000)
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

  fnnls_cols_cpp(Y, X, tol, max_iter, sum_to_constant, constant, lower_bound, lb, parallel, ncores)
}
