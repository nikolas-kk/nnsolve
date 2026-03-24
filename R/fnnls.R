#' Fast Non-Negative Least Squares
#'
#' Solves the NNLS problem min ||y - Xw||^2 subject to w >= 0 using the Fast Non-Negative Least Squares algorithm of Bro & de Jong (1997).
#'
#' @param XtX A symmetric positive definite matrix of dimensions k x k.
#' @param Xty A numeric vector of length k.
#' @param tol The convergence tolerance, default is 1e-6.
#' @param max_iter The maximum number of iterations, default is 1000.
#' @param sum_to_constant If TRUE all entries sum to 'constant', Default is FALSE.
#' @param constant If sum_to_constant is TRUE, all entries sum to this number. The default value is 1.
#' @param lower_bound If TRUE all entries bounded below by 'lb', otherwise they are nonnegative. The default value is FALSE.
#' @param lb If lower_bound is TRUE all entries are bounded below by 'lb'. The default value is 0.
#'
#' @return A non-negative numeric vector of length k with the estimated coefficients.
#'
#' @examples
#' k <- 10
#' D <- 100
#' H <- matrix(rnorm(k * D), nrow = k, ncol = D)
#' x <- rnorm(D)
#' XtX <- H %*% t(H) + diag(1e-8, k)
#' Xty <- as.vector(H %*% x)
#' w <- fnnls(XtX, Xty)
#'
#' @references
#' Bro, Rasmus & Jong, Sijmen. (1997). A Fast Non-negativity-constrained Least Squares Algorithm.
#' Journal of Chemometrics. 11. 393-401. 10.1002/(SICI)1099-128X(199709/10)11:53.0.CO;2-L.
#'
#' @export
fnnls <- function(XtX, Xty, tol = 1e-6, max_iter = 1000, sum_to_constant = FALSE, constant = 1, lower_bound = FALSE, lb = 0) {
  if (!is.matrix(XtX)) stop("XtX must be a matrix")
  if (!is.numeric(Xty)) stop("Xty must be numeric")
  if (nrow(XtX) != ncol(XtX)) stop("XtX must be square")
  if (nrow(XtX) != length(Xty)) stop("XtX and Xty dimensions must match")
  if (tol < 0) stop("Tolerance must be small positive number")
  fnnls_cpp(XtX, Xty, tol, max_iter, sum_to_constant, constant, lower_bound, lb)
}
