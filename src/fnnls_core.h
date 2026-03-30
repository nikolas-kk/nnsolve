#pragma once

#include <RcppEigen.h>

Eigen::VectorXd fnnls_core(const Eigen::MatrixXd &XtX,
                           const Eigen::VectorXd &Xty, const double tol,
                           const int max_iter);
