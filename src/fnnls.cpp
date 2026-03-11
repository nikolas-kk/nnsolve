//[[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <cfloat>
#include <vector>

Eigen::VectorXd fnnls_core(const Eigen::MatrixXd &XtX,
                           const Eigen::VectorXd &Xty, const double tol,
                           const int max_iter) {
  const int k = Xty.rows();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(k);
  Eigen::VectorXd negative_gradient = Xty;
  std::vector<bool> is_active(k, true);
  std::vector<int> passive, updated_passive;
  passive.reserve(k);
  updated_passive.reserve(k);
  int outer_iter = 0;
  while (outer_iter < max_iter) {
    ++outer_iter;
    bool optimal = true;
    for (int i = 0; i < k; ++i) {
      if (is_active[i] && negative_gradient(i) > tol) {
        passive.push_back(i);
        is_active[i] = false;
        optimal = false;
      }
    }
    if (optimal)
      return w;
    int inner_iter = 0;
    while (inner_iter < max_iter) {
      ++inner_iter;
      Eigen::VectorXd s_p = XtX(passive, passive).llt().solve(Xty(passive));
      bool feasible_sp = true;
      for (double entry : s_p) {
        if (entry <= tol) {
          feasible_sp = false;
          break;
        }
      }
      if (feasible_sp) {
        w(passive) = s_p;
        break;
      }
      double min_a = 2.0;
      Eigen::VectorXd w_p = w(passive);
      for (Eigen::Index r = 0; r < s_p.size(); ++r) {
        if (s_p(r) <= tol) {
          double a = w_p(r) / (w_p(r) - s_p(r));
          min_a = (a < min_a) ? a : min_a;
        }
      }
      updated_passive.clear();
      if (min_a <= tol) {
        for (int r = 0; r < (int)s_p.size(); ++r) {
          if (s_p(r) > tol) {
            updated_passive.push_back(passive[r]);
            w(passive[r]) = s_p(r);
          } else {
            is_active[passive[r]] = true;
            w(passive[r]) = 0.0;
          }
        }
      } else {
        w(passive) = w_p + min_a * (s_p - w_p);
        for (int r : passive) {
          if (w(r) > tol) {
            updated_passive.push_back(r);
          } else {
            is_active[r] = true;
            w(r) = 0.0;
          }
        }
      }
      std::swap(passive, updated_passive);
    }
    negative_gradient = Xty;
    negative_gradient.noalias() -= XtX * w;
  }
  return w;
}

//[[Rcpp::export]]
Eigen::VectorXd fnnls_cpp(const Eigen::MatrixXd &XtX, const Eigen::VectorXd &Xty,
                       const double tol = 1e-6, const int max_iter = 1000,
                       const bool sum_to_constant = false,
                       const double constant = 1.0,
                       const bool lower_bound = false, const double lb = 0.0) {
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

  if(sum_to_constant)
    w.array() = w.array() / (w.sum()/ constant);
  return w;
}
