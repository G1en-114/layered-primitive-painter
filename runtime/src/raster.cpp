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

void paint_line(const Protocol& proto, const Line& l, int S, float* P, float* T) {
  if (l.alpha <= 0.f) return;
  constexpr int K = 16;
  const float scale = static_cast<float>(proto.canvas_size) / static_cast<float>(S);
  float px[K + 1], py[K + 1];
  for (int i = 0; i <= K; ++i) {
    const float t = static_cast<float>(i) / K, u = 1.f - t;
    px[i] = u * u * l.x0 + 2.f * u * t * l.x1 + t * t * l.x2;
    py[i] = u * u * l.y0 + 2.f * u * t * l.y1 + t * t * l.y2;
  }
  const float half = l.width / 2.f, pad = half + 1.f;
  const float bx0 = *std::min_element(px, px + K + 1) - pad, bx1 = *std::max_element(px, px + K + 1) + pad;
  const float by0 = *std::min_element(py, py + K + 1) - pad, by1 = *std::max_element(py, py + K + 1) + pad;
  const int x0 = std::max(0, static_cast<int>(std::floor(bx0 / scale - 0.5f)));
  const int x1 = std::min(S - 1, static_cast<int>(std::ceil(bx1 / scale - 0.5f)));
  const int y0 = std::max(0, static_cast<int>(std::floor(by0 / scale - 0.5f)));
  const int y1 = std::min(S - 1, static_cast<int>(std::ceil(by1 / scale - 0.5f)));
  const float a_max = std::min(l.alpha, proto.max_alpha);
  const size_t plane = size_t(S) * S;
  for (int yi = y0; yi <= y1; ++yi) {
    const float qy = (yi + 0.5f) * scale;
    for (int xi = x0; xi <= x1; ++xi) {
      const float qx = (xi + 0.5f) * scale;
      float d2 = 1e30f;
      for (int i = 0; i < K; ++i) {
        const float ex = px[i + 1] - px[i], ey = py[i + 1] - py[i];
        const float tt = std::clamp(((qx - px[i]) * ex + (qy - py[i]) * ey) / std::max(ex * ex + ey * ey, 1e-8f), 0.f, 1.f);
        const float dx = qx - px[i] - tt * ex, dy = qy - py[i] - tt * ey;
        d2 = std::min(d2, dx * dx + dy * dy);
      }
      const float cov = std::clamp(half - std::sqrt(d2 + 1e-8f) + 0.5f, 0.f, 1.f);
      const float a = std::min(cov * a_max, proto.max_alpha);
      if (a <= 0.f) continue;
      const size_t o = size_t(yi) * S + xi;
      P[o] = (1.f - a) * P[o] + a * l.r;
      P[plane + o] = (1.f - a) * P[plane + o] + a * l.g;
      P[2 * plane + o] = (1.f - a) * P[2 * plane + o] + a * l.b;
      T[o] *= (1.f - a);
    }
  }
}

std::vector<unsigned char> render_strokes_rgb8(const Protocol& proto, const std::vector<Stroke>& strokes, int S) {
  const size_t plane = size_t(S) * S;
  std::vector<float> img(3 * plane);
  for (int ch = 0; ch < 3; ++ch) std::fill(img.begin() + ch * plane, img.begin() + (ch + 1) * plane, proto.base_color[ch]);
  for (int k = 0; k < proto.num_layers(); ++k) {
    std::vector<float> P(3 * plane, 0.f), T(plane, 1.f);
    bool any = false;
    for (int pass = 0; pass < 2; ++pass)                 // Gaussians, then lines
      for (const auto& s : strokes) {
        if (s.layer != k || s.line != (pass == 1)) continue;
        any = true;
        if (s.line) paint_line(proto, proto.decode_line(s.tokens), S, P.data(), T.data());
        else paint(proto, proto.decode(s.tokens), S, P.data(), T.data());
      }
    if (!any) continue;
    for (int ch = 0; ch < 3; ++ch)
      for (size_t o = 0; o < plane; ++o) img[ch * plane + o] = P[ch * plane + o] + T[o] * img[ch * plane + o];
  }
  std::vector<unsigned char> out(3 * plane);
  for (size_t o = 0; o < plane; ++o)
    for (int ch = 0; ch < 3; ++ch)
      out[o * 3 + ch] = static_cast<unsigned char>(std::lround(std::clamp(img[ch * plane + o], 0.f, 1.f) * 255.f));
  return out;
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
