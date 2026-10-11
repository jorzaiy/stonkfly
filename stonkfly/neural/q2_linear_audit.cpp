// Independent double-precision Arnoldi helper for Q2 stability audit.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" int q2_arnoldi(
    int n, const int64_t* ptr, const int32_t* post, const double* weight,
    const double* start, int requested_k, double* h, double* last_basis,
    double* tail_norm) {
  if (n <= 0 || requested_k <= 0) return -1;
  const int kmax = std::min(requested_k, n);
  std::fill(h, h + (int64_t)requested_k * requested_k, 0.0);
  std::vector<double> q((int64_t)n * (kmax + 1), 0.0), y(n, 0.0);
  double norm = 0.0;
  for (int i = 0; i < n; ++i) norm = std::hypot(norm, start[i]);
  if (!(norm > 0.0) || !std::isfinite(norm)) return -2;
  for (int i = 0; i < n; ++i) q[i] = start[i] / norm;
  int actual = kmax;
  for (int j = 0; j < kmax; ++j) {
    std::fill(y.begin(), y.end(), 0.0);
    const double* qj = q.data() + (int64_t)j * n;
    for (int source = 0; source < n; ++source)
      for (int64_t e = ptr[source]; e < ptr[source + 1]; ++e)
        y[post[e]] += weight[e] * qj[source];
    // Two-pass modified Gram-Schmidt.
    for (int pass = 0; pass < 2; ++pass) {
      for (int i = 0; i <= j; ++i) {
        const double* qi = q.data() + (int64_t)i * n;
        double dot = 0.0;
        for (int r = 0; r < n; ++r) dot += qi[r] * y[r];
        h[(int64_t)i * requested_k + j] += dot;
        for (int r = 0; r < n; ++r) y[r] -= dot * qi[r];
      }
    }
    double beta = 0.0;
    for (double value : y) beta = std::hypot(beta, value);
    if (j + 1 < requested_k) h[(int64_t)(j + 1) * requested_k + j] = beta;
    if (j == kmax - 1) {
      *tail_norm = beta;
      std::copy(qj, qj + n, last_basis);
    }
    if (!(beta > 1e-14) || !std::isfinite(beta)) {
      actual = j + 1; *tail_norm = beta;
      std::copy(qj, qj + n, last_basis); break;
    }
    if (j + 1 < kmax) {
      double* qnext = q.data() + (int64_t)(j + 1) * n;
      for (int r = 0; r < n; ++r) qnext[r] = y[r] / beta;
    }
  }
  return actual;
}
