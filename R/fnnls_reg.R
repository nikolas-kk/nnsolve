#' Fast Non-Negative Least Squares Regression
#'
#' Solves the NNLS problem min ||y - Xb||^2 subject to b >= 0 using the Fast Non-Negative Least Squares algorithm of Bro & de Jong (1997).
#'
#' @param y A numeric vector of length n.
#' @param X A numeric matrix of dimensions n x k.
#' @param tol The convergence tolerance, default is 1e-6.
#' @param max_iter The maximum number of iterations, default is 1000.
#' @param sum_to_constant If TRUE all entries sum to 'constant', Default is FALSE.
#' @param constant If sum_to_constant is TRUE, all entries sum to this number. The default value is 1.
#' @param lower_bound If TRUE all entries bounded below by 'lb', otherwise they are nonnegative. The default value is FALSE.
#' @param lb If lower_bound is TRUE all entries are bounded below by 'lb'. The default value is 0.
#'
#' @return A list with two elements:
#' \itemize{
#'   \item \code{b}: A non-negative numeric vector of length k with the estimated coefficients.
#'   \item \code{mse}: The mean squared error of the fitted model.
#' }
#'
#' @examples
#' n <- 100
#' k <- 10
#' X <- matrix(rnorm(n * k), nrow = n, ncol = k)
#' true_b <- abs(rnorm(k))
#' y <- X %*% true_b + rnorm(n, sd = 0.1)
#' result <- fnnls_reg(y, X)
#' result$b
#' result$mse
#'
#' @references
#' Bro, Rasmus & Jong, Sijmen. (1997). A Fast Non-negativity-constrained Least Squares Algorithm.
#' Journal of Chemometrics. 11. 393-401. 10.1002/(SICI)1099-128X(199709/10)11:53.0.CO;2-L.
#'
#' @export
fnnls_reg <- function(y, X, tol = 1e-6, max_iter = 1000, sum_to_constant = FALSE, constant = 1, lower_bound = FALSE, lb = 0) {
  if (!is.matrix(X)) stop("X must be a matrix")
  if (!is.numeric(y)) stop("y must be numeric")
  if (tol < 0) stop("Tolerance must be small positive number")
  b <- fnnls_reg_cpp(y, X, tol, max_iter, sum_to_constant, constant, lower_bound, lb)
  mse <- mean((y - X %*% b)^2)
  list(b = b, mse = mse)
}
