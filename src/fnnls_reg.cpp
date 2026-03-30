//[[Rcpp::depends(RcppEigen)]]

#include "fnnls.h"
#include <RcppEigen.h>


//[[Rcpp::export]]
Eigen::VectorXd fnnls_reg_cpp(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                               const double tol = 1e-6, const int max_iter = 1000,
                               const bool sum_to_constant = false,
                               const double constant = 1.0,
                               const bool lower_bound = false, const double lb = 0.0){

  const int X_cols = X.cols();
  const Eigen::MatrixXd Xt = X.transpose();
  const Eigen::MatrixXd Xty = Xt * y;

  Eigen::MatrixXd XtX(X_cols, X_cols);
  XtX.setZero();
  XtX.selfadjointView<Eigen::Upper>().rankUpdate(Xt);
  XtX = XtX.selfadjointView<Eigen::Upper>();

  const Eigen::VectorXd b = fnnls_cpp(XtX, Xty, tol, max_iter,
            sum_to_constant, constant, lower_bound, lb);

  return b;
}
