#' Fast Non-Negative Least Squares
#'
#' Solves the NNLS problem min ||Xty - XtX * w||^2 subject to w >= 0
#' using the Fast Non-Negative Least Squares algorithm of Bro & de Jong (1997).
#'
#' @param XtX symmetric positive definite matrix of dimensions k x k
#' @param Xty numeric vector of length k
#' @param tol convergence tolerance. Default 1e-6
#' @param max_iter maximum number of iterations. Default 1000
#' @param sum_to_constant if TRUE all entries sum to 'constant'.Default FALSE
#' @param constant if sum_to_constant is TRUE, all entries sum to this number. Default 1
#' @param lower_bound if TRUE all entries bounded below by 'lb', otherwise they are nonnegative. Default FALSE
#' @param lb if lower_bound is TRUE all entries are bounded below by 'lb'. Default 0
#' 
#' @return non-negative numeric vector of length k
#'
#' @examples
#' k <- 10
#' D <- 100
#' H   <- matrix(rnorm(k * D), nrow = k, ncol = D)
#' x   <- rnorm(D)
#' XtX <- H %*% t(H) + diag(1e-8, k)
#' Xty <- as.vector(H %*% x)
#' w   <- fnnls(XtX, Xty)
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
  if(tol < 0) stop("Tolerance must be small positive number")
  fnnls_cpp(XtX, Xty, tol, max_iter,sum_to_constant,constant,lower_bound ,lb)
}
