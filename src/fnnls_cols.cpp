//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(openmp)]]

#include "fnnls.h"
#include <RcppEigen.h>
#ifdef __OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;

//[[Rcpp::export]]

MatrixXd fnnls_cols_cpp(const MatrixXd &Y, const MatrixXd &X,
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
  const MatrixXd Xt = X.transpose();

  const MatrixXd XtX(X_cols, X_cols);
  XtX.setZero();
  XtX.selfadjointView<Eigen::Upper>().rankUpdate(X.transpose());
  XtX = XtX.selfadjointView<Eigen::Upper>();

  MatrixXd B(X_cols, Y_cols);

#ifdef _OPENMP
#pragma omp parallel for if (parallel)
#endif
  for (int target = 0; target < Y_cols; ++target) {

    B.col(target) = fnnls_cpp(XtX, Xt * Y.col(target), tol, max_iter,
                              sum_to_constant, constant, lower_bound, lb);
  }

  return B;
}
