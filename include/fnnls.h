#pragma once

#include <Eigen/Dense>

Eigen::VectorXd fnnls_cpp(const Eigen::MatrixXd &XtX, const Eigen::VectorXd &Xty,
                       const double tol = 1e-6, const int max_iter = 1000,
                       const bool sum_to_constant = false,
                       const double constant = 1.0,
                       const bool lower_bound = false, const double lb = 0.0);
