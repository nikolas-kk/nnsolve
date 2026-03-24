//[[Rcpp::depends(RcppEigen)]]

#include "fnnls.h"
#include "fnnls_core.h"
#include <RcppEigen.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

//[[Rcpp::export]]
VectorXd fnnls_cpp(const MatrixXd &XtX, const VectorXd &Xty,
                   const double tol = 1e-6, const int max_iter = 1000,
                   const bool sum_to_constant = false,
                   const double constant = 1.0, const bool lower_bound = false,
                   const double lb = 0.0) {
  if (lower_bound && lb < 0)
    Rcpp::stop("Lower bound cannot be negative");

  if (sum_to_constant && constant <= 0)
    Rcpp::stop("Non-negative entries cannot sum to a non-positive constant");

  MatrixXd XtX_mod = XtX;
  VectorXd Xty_mod = Xty;
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

  VectorXd w = fnnls_core(XtX_mod, Xty_mod, tol, max_iter);

  if (lower_bound)
    w.array() += lb;

  if (sum_to_constant)
    w.array() = w.array() / (w.sum() / constant);
  return w;
}
