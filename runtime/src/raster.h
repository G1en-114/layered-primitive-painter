// CPU reference rasterizer for generalized Gaussians (lpp.render.reference), composited with
// premultiplied "over" in sRGB. Painting primitives one at a time onto (P, T) is algebraically
// the same as the reference's log-space group reduction followed by over().
#pragma once
#include <vector>

#include "protocol.h"

namespace lpp {

struct Layered {                       // per-layer premultiplied state at one resolution
  int L = 0, S = 0;
  std::vector<float> P;                // [L][3][S*S]
  std::vector<float> T;                // [L][S*S]
  Layered(int layers, int size) : L(layers), S(size), P(size_t(layers) * 3 * size * size, 0.f),
                                  T(size_t(layers) * size * size, 1.f) {}
};

// Paint one primitive into (P, T) planes of size S (P: 3 planes of S*S).
void paint(const Protocol& proto, const Prim& p, int S, float* P, float* T);

// Canvas for the canvas encoder: composite RGB over the base colour + coverage 1 - T_k per layer,
// channel-major [3 + L][S][S] (lpp.canvas.compose).
void compose(const Protocol& proto, const Layered& st, float* out);

// Paint one Bezier line (protocol v3) into (P, T): hard edge, one-pixel ramp (lpp.render.bezier, 16 pieces).
void paint_line(const Protocol& proto, const Line& l, int S, float* P, float* T);

// Final image: primitives in paint order over the base colour -> RGB8 [S][S][3].
std::vector<unsigned char> render_rgb8(const Protocol& proto, const std::vector<Prim>& prims, int S);

// Final image of a stroke sequence as protocol v3 composites it (lpp.render.strokes): layers back to front;
// inside a layer its Gaussians, then its lines, each in sequence order. tokens: one row per stroke.
struct Stroke { int layer; bool line; const int64_t* tokens; };
std::vector<unsigned char> render_strokes_rgb8(const Protocol& proto, const std::vector<Stroke>& strokes, int S);

}  // namespace lpp
