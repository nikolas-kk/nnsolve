//[[Rcpp::depends(RcppEigen)]]

#include "fnnls_core.h"
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

VectorXd fnnls_core(const MatrixXd &XtX, const VectorXd &Xty, const double tol,
                    const int max_iter) {
  const int k = Xty.rows();
  VectorXd w = VectorXd::Zero(k);
  VectorXd negative_gradient = Xty;
  vector<bool> is_active(k, true);
  vector<int> passive, updated_passive;
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
      VectorXd s_p = XtX(passive, passive).llt().solve(Xty(passive));
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
      VectorXd w_p = w(passive);
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
