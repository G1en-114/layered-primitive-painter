#include "raster.h"

#include <algorithm>
#include <cmath>

namespace lpp {

void paint(const Protocol& proto, const Prim& p, int S, float* P, float* T) {
  if (p.alpha <= 0.f) return;
  const float scale = static_cast<float>(proto.canvas_size) / static_cast<float>(S);
  const float c = std::cos(p.theta), s = std::sin(p.theta);
  const float ix = 1.f / (p.sx * p.sx), iy = 1.f / (p.sy * p.sy);
  const float A = ix * c * c + iy * s * s, B = (ix - iy) * c * s, C = ix * s * s + iy * c * c;
  const float cut2 = proto.cutoff_sigma * proto.cutoff_sigma;
  const float g_cut = std::exp(-0.5f * std::pow(cut2, p.beta));
  // the support d2 <= cut2 lies inside a circle of radius cutoff * max sigma
  const float rad = proto.cutoff_sigma * std::max(p.sx, p.sy);
  const int x0 = std::max(0, static_cast<int>(std::floor((p.x - rad) / scale - 0.5f)));
  const int x1 = std::min(S - 1, static_cast<int>(std::ceil((p.x + rad) / scale - 0.5f)));
  const int y0 = std::max(0, static_cast<int>(std::floor((p.y - rad) / scale - 0.5f)));
  const int y1 = std::min(S - 1, static_cast<int>(std::ceil((p.y + rad) / scale - 0.5f)));
  const size_t plane = size_t(S) * S;
  for (int yi = y0; yi <= y1; ++yi) {
    const float dy = (yi + 0.5f) * scale - p.y;
    for (int xi = x0; xi <= x1; ++xi) {
      const float dx = (xi + 0.5f) * scale - p.x;
      const float d2 = A * dx * dx + 2.f * B * dx * dy + C * dy * dy;
      if (d2 >= cut2) continue;
      const float g = std::exp(-0.5f * std::pow(std::max(d2, 1e-12f), p.beta));
      const float a = std::min(p.alpha * (g - g_cut) / (1.f - g_cut), proto.max_alpha);
      if (a <= 0.f) continue;
      const size_t o = size_t(yi) * S + xi;
      P[o] = (1.f - a) * P[o] + a * p.r;
      P[plane + o] = (1.f - a) * P[plane + o] + a * p.g;
      P[2 * plane + o] = (1.f - a) * P[2 * plane + o] + a * p.b;
      T[o] *= (1.f - a);
    }
  }
}

void compose(const Protocol& proto, const Layered& st, float* out) {
  const size_t plane = size_t(st.S) * st.S;
  for (int ch = 0; ch < 3; ++ch) {
    for (size_t o = 0; o < plane; ++o) {
      float v = proto.base_color[ch];
      for (int k = 0; k < st.L; ++k) v = st.P[(size_t(k) * 3 + ch) * plane + o] + st.T[size_t(k) * plane + o] * v;
      out[ch * plane + o] = v;
    }
  }
  for (int k = 0; k < st.L; ++k)
    for (size_t o = 0; o < plane; ++o) out[(3 + k) * plane + o] = 1.f - st.T[size_t(k) * plane + o];
}

std::vector<unsigned char> render_rgb8(const Protocol& proto, const std::vector<Prim>& prims, int S) {
  const size_t plane = size_t(S) * S;
  std::vector<float> P(3 * plane, 0.f), T(plane, 1.f);
  for (const auto& p : prims) paint(proto, p, S, P.data(), T.data());
  std::vector<unsigned char> img(3 * plane);
  for (size_t o = 0; o < plane; ++o)
    for (int ch = 0; ch < 3; ++ch) {
      const float v = P[ch * plane + o] + T[o] * proto.base_color[ch];
      img[o * 3 + ch] = static_cast<unsigned char>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
    }
  return img;
}

}  // namespace lpp
