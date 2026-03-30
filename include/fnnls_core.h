#pragma once

#include <Eigen/Dense>

Eigen::VectorXd fnnls_core(const Eigen::MatrixXd &XtX,
                           const Eigen::VectorXd &Xty, const double tol,
                           const int max_iter);
