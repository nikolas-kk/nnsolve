//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(openmp)]]

#include "fnnls.h"
#include <RcppEigen.h>
#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::export]]

Eigen::MatrixXd fnnls_cols_cpp(const Eigen::MatrixXd &Y, const Eigen::MatrixXd &X,
                        const double tol = 1e-6, const int max_iter = 1000,
                        const bool sum_to_constant = false,
                        const double constant = 1.0,
                        const bool lower_bound = false, const double lb = 0.0,
                        const bool parallel = false, const int ncores = -1) {
#ifdef _OPENMP
  if (parallel && ncores > 0)
    omp_set_num_threads(ncores);
#endif

  const int X_cols = X.cols(), Y_cols = Y.cols();
  const Eigen::MatrixXd Xt = X.transpose();
  const Eigen::MatrixXd XtY = Xt * Y;

  Eigen::MatrixXd XtX(X_cols, X_cols);
  XtX.setZero();
  XtX.selfadjointView<Eigen::Upper>().rankUpdate(Xt);
  XtX = XtX.selfadjointView<Eigen::Upper>();

  Eigen::MatrixXd B(X_cols, Y_cols);

#ifdef _OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (int target = 0; target < Y_cols; ++target) {
    B.col(target) = fnnls_cpp(XtX, XtY.col(target), tol, max_iter,
                              sum_to_constant, constant, lower_bound, lb);
  }

  return B;
}
