#include "protocol.h"

#include <fstream>
#include <stdexcept>

#include "json.hpp"

namespace lpp {

using nlohmann::json;

static json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  return json::parse(f);
}

int Protocol::event_id(const std::string& name) const {
  for (size_t i = 0; i < events.size(); ++i)
    if (events[i] == name) return static_cast<int>(i);
  throw std::runtime_error("unknown event " + name);
}

Prim Protocol::decode(const int64_t* tok) const {
  Prim p{};
  for (size_t i = 0; i < attrs.size(); ++i) {
    const auto& n = attrs[i];
    const int64_t t = tok[i];
    if (n == "color") {
      const auto& c = color.at(t);
      p.r = c[0], p.g = c[1], p.b = c[2];
      continue;
    }
    const float v = table.at(n).at(t);
    if (n == "x") p.x = v;
    else if (n == "y") p.y = v;
    else if (n == "sigma_x") p.sx = v;
    else if (n == "sigma_y") p.sy = v;
    else if (n == "theta") p.theta = v;
    else if (n == "beta") p.beta = v;
    else if (n == "alpha") p.alpha = v;
    else throw std::runtime_error("unsupported attribute " + n);
  }
  return p;
}

Protocol Protocol::load(const std::string& dir) {
  Protocol P;
  const json c = read_json(dir + "/codec.json");
  P.attrs = c["attributes"].get<std::vector<std::string>>();
  for (const auto& n : P.attrs) {
    if (n == "color") {
      for (const auto& rgb : c["tables"]["color"]) P.color.push_back({rgb[0].get<float>(), rgb[1].get<float>(), rgb[2].get<float>()});
      P.vocab.push_back(static_cast<int64_t>(P.color.size()));
    } else {
      P.table[n] = c["tables"][n].get<std::vector<float>>();
      P.vocab.push_back(static_cast<int64_t>(P.table[n].size()));
    }
  }
  P.canvas_size = c["canvas_size"];
  for (int i = 0; i < 3; ++i) P.base_color[i] = c["base_color"][i];
  P.max_alpha = c["max_alpha"];
  P.cutoff_sigma = c["cutoff_sigma"];
  P.events = c["events"].get<std::vector<std::string>>();
  for (const auto& l : c["layers"]) P.layer_cap.push_back(l["max_primitives"]);
  P.max_text_positions = c["max_text_positions"];
  P.kind_bos = c["kinds"]["BOS"], P.kind_control = c["kinds"]["CONTROL"], P.kind_primitive = c["kinds"]["PRIMITIVE"];
  P.no_attr = c["no_attr"];

  const json m = read_json(dir + "/manifest.json");
  const json& cfg = m["config"];
  P.n_layers = cfg["n_layers"], P.n_heads = cfg["n_heads"], P.n_kv_heads = cfg["n_kv_heads"];
  P.d_model = cfg["d_model"], P.text_dim = cfg["text_dim"], P.max_positions = cfg["max_positions"];
  P.canvas = cfg["canvas"], P.canvas_k = cfg["canvas_k"], P.canvas_cfg_size = cfg["canvas_size"];
  P.draft = cfg.value("order", std::string("layer")) == "draft";   // older manifests: layer order
  return P;
}

}  // namespace lpp
