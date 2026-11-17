#include "grammar.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace lpp {

Grammar::Grammar(const Protocol& p, std::set<int> require, bool allow_special, bool draft)
    : p_(p), require_(std::move(require)), allow_special_(allow_special), draft_(draft) {
  prim_ = p.event_id("PRIMITIVE");
  eos_ = p.event_id("EOS");
  for (int k = 0; k < p.num_layers(); ++k) {
    layer_ev_.push_back(p.event_id("LAYER_" + std::to_string(k)));
    redraw_ev_.push_back(p.event_id("REDRAW_" + std::to_string(k)));
    refine_ev_.push_back(p.event_id("REFINE_" + std::to_string(k)));
  }
}

std::vector<int> Grammar::allowed_events() const {
  std::vector<int> out;
  if (done_) return out;
  auto seen = [&](int k) { auto it = seen_.find(k); return it == seen_.end() ? 0 : it->second; };
  if (current_ >= 0 && p_.layer_cap[current_] - seen(current_) > 0) out.push_back(prim_);
  if (current_ >= 0 && count_ == 0) return out;           // a marker must be followed by a primitive
  bool req_ok = true;
  for (int r : require_)
    if (seen(r) <= 0) req_ok = false;
  if (!special_) {
    for (int k = 0; k < p_.num_layers(); ++k) {
      bool skips_required = false;
      for (int r : require_)
        if (r < k && seen(r) == 0) skips_required = true;
      if (k > current_ && !seen_.count(k) && !skips_required) out.push_back(layer_ev_[k]);
      if (allow_special_ && !seen_.count(k)) out.push_back(redraw_ev_[k]);
      else if ((allow_special_ || (draft_ && req_ok)) && seen_.count(k) && k <= current_ &&
               p_.layer_cap[k] - seen_.at(k) > 0)
        out.push_back(refine_ev_[k]);
    }
  } else if (refining_ && draft_) {
    for (int k = current_ + 1; k < p_.num_layers(); ++k)
      if (seen(k) > 0 && p_.layer_cap[k] - seen(k) > 0) out.push_back(refine_ev_[k]);
  }
  if (current_ >= 0 && req_ok) out.push_back(eos_);
  if (out.empty()) out.push_back(eos_);
  std::sort(out.begin(), out.end());
  return out;
}

void Grammar::step(int event) {
  auto allowed = allowed_events();
  if (std::find(allowed.begin(), allowed.end(), event) == allowed.end())
    throw std::runtime_error("event " + p_.events[event] + " not allowed");
  if (event == prim_) {
    ++count_;
    ++seen_[current_];
  } else if (event == eos_) {
    done_ = true;
  } else {
    const std::string& n = p_.events[event];
    current_ = std::stoi(n.substr(n.find('_') + 1));
    count_ = 0;
    if (n.rfind("REDRAW", 0) == 0 || n.rfind("REFINE", 0) == 0) special_ = true;
    if (n.rfind("REFINE", 0) == 0) refining_ = true;
    seen_.emplace(current_, 0);
  }
}

}  // namespace lpp
