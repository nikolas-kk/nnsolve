//[[Rcpp::depends(RcppEigen)]]

#include <nnsolve/fnnls_core.h>
#include <nnsolve/fnnls.h>
#include <RcppEigen.h>


//[[Rcpp::export]]

Eigen::VectorXd fnnls_cpp(const Eigen::MatrixXd &XtX, const Eigen::VectorXd &Xty,
                   const double tol, const int max_iter,
                   const bool sum_to_constant,
                   const double constant, const bool lower_bound,
                   const double lb) {
  if (lower_bound && lb < 0)
    Rcpp::stop("Lower bound cannot be negative");

  if (sum_to_constant && constant <= 0)
    Rcpp::stop("Non-negative entries cannot sum to a non-positive constant");

  Eigen::MatrixXd XtX_mod = XtX;
  Eigen::VectorXd Xty_mod = Xty;
  const double scale = XtX.trace();
  XtX_mod /= scale;
  Xty_mod /= scale;

  if (!lower_bound && !sum_to_constant)
    return fnnls_core(XtX, Xty, tol, max_iter);

  if (sum_to_constant) {
    XtX_mod.array() += 100.0;
    Xty_mod.array() += 100.0 * constant;
  }

  if (lower_bound) {
    Xty_mod.noalias() -= XtX_mod.rowwise().sum() * lb;
  }

  Eigen::VectorXd w = fnnls_core(XtX_mod, Xty_mod, tol, max_iter);

  if (lower_bound)
    w.array() += lb;

  if (sum_to_constant)
    w.array() = w.array() / (w.sum() / constant);
  return w;
}

