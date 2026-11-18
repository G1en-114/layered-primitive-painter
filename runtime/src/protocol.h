// Protocol, codec tables and model config, read from an export directory (tools/export_onnx.py):
// manifest.json (config, graph names) and codec.json (token -> parameter lookup tables).
#pragma once
#include <array>
#include <map>
#include <string>
#include <vector>

namespace lpp {

struct Prim {               // one decoded primitive (canvas pixels, radians, sRGB)
  float x, y, sx, sy, theta, beta, r, g, b, alpha;
  float bend = 0.f;         // protocol v3 (lines only)
};

struct Line {               // protocol v3 LINE stroke: quadratic Bezier P0 -> P2, control P1 (lpp.codec.decode_lines)
  float x0, y0, x1, y1, x2, y2, width, r, g, b, alpha;
};

struct Protocol {
  // codec
  std::vector<std::string> attrs;                       // protocol attribute order
  std::map<std::string, std::vector<float>> table;      // scalar attributes
  std::vector<std::array<float, 3>> color;              // sRGB per colour code
  std::vector<int64_t> vocab;                           // vocabulary size per attribute
  int canvas_size = 256;
  std::array<float, 3> base_color{1, 1, 1};
  float max_alpha = 0.99f, cutoff_sigma = 3.f;
  // grammar
  std::vector<std::string> events;
  std::vector<int> layer_cap;
  int max_text_positions = 32;
  int kind_bos = 1, kind_control = 2, kind_primitive = 3, no_attr = -1;
  // model config
  int n_layers = 0, n_heads = 0, n_kv_heads = 0, d_model = 0, text_dim = 0, max_positions = 0;
  bool canvas = false;
  int canvas_k = 32, canvas_cfg_size = 64;
  bool draft = false;   // model trained in the draft order (config "order": "draft"): chained REFINE allowed
  // strokes: PRIMITIVE (Gaussian) and, from protocol v3, LINE (Bezier line)
  int prim_ev = 0, line_ev = -1;                        // line_ev = -1: no LINE event (v2)
  int beta_attr = -1, bend_attr = -1, bend_mid = 0;     // slots a stroke type does not use, and their token
  float beta_max = 8.f;
  bool is_stroke(int64_t e) const { return e == prim_ev || (line_ev >= 0 && e == line_ev); }

  int event_id(const std::string& name) const;
  int num_layers() const { return static_cast<int>(layer_cap.size()); }
  int head_dim() const { return d_model / n_heads; }
  Prim decode(const int64_t* tokens) const;            // tokens in protocol attribute order
  Line decode_line(const int64_t* tokens) const;       // a LINE record's tokens
  // canvas feedback (lpp.codec.Codec.decode_canvas): a line becomes a hard Gaussian over its extent
  Prim decode_canvas(const int64_t* tokens, bool line) const;

  static Protocol load(const std::string& export_dir);
};

}  // namespace lpp
