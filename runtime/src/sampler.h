// Port of lpp.sample._sample (temperature + top-p support, inverse CDF in vocabulary order) and of
// classifier-free guidance. Sequential float32 arithmetic; PyTorch's vectorised softmax/cumsum
// can round differently in the last bit, which is why parity is reported per sequence.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <algorithm>
#include <vector>

namespace lpp {

struct SampleConfig {
  float temperature = 1.f;
  float top_p = 0.95f;
  float cfg_scale = 1.f;
};

// guided = uncond + s * (cond - uncond), elementwise
inline void guide(const float* cond, const float* uncond, float s, float* out, int n) {
  for (int i = 0; i < n; ++i) out[i] = uncond[i] + s * (cond[i] - uncond[i]);
}

// logits may contain -inf (grammar mask). Returns the sampled index.
inline int sample_top_p(const float* logits, int n, const SampleConfig& c, float u) {
  if (c.temperature <= 0) return static_cast<int>(std::max_element(logits, logits + n) - logits);
  float mx = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) mx = std::max(mx, logits[i] / c.temperature);
  std::vector<float> p(n);
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    p[i] = std::exp(logits[i] / c.temperature - mx);
    sum += p[i];
  }
  for (auto& v : p) v /= sum;
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
  std::vector<float> w(n, 0.f);                          // kept mass, back in vocabulary order
  float cum = 0, kept = 0;
  for (int j = 0; j < n; ++j) {
    const float v = p[idx[j]];
    if (cum < c.top_p) w[idx[j]] = v;                    // (cum - sp) < top_p: always keeps the top token
    cum += v;
  }
  for (int i = 0; i < n; ++i) kept += w[i];
  const float uu = std::min(u, 1.f - 1e-7f);
  float acc = 0;
  int last = -1;
  for (int i = 0; i < n; ++i) {                          // searchsorted(cdf, u), left side, vocabulary order
    if (w[i] <= 0.f) continue;
    acc += w[i];
    last = i;
    if (acc / kept >= uu) return i;
  }
  return last;
}

}  // namespace lpp
