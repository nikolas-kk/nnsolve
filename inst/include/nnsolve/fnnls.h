#pragma once

#include "types.h"
#include <Eigen/Dense>

Vec fnnls_cpp(const Mat &XtX, const Vec &Xty, const double tol = 1e-6,
              const int max_iter = 1000, const bool sum_to_constant = false,
              const double constant = 1.0, const bool lower_bound = false,
              const double lb = 0.0);

