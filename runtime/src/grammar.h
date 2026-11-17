// Port of lpp.sequence.Grammar (decoding use: no REDRAW/REFINE from BOS, optional EOS rules;
// draft = the model's draft order: refining may start once the required layers exist, and
// REFINE may follow REFINE with k strictly increasing).
#pragma once
#include <map>
#include <set>
#include <vector>

#include "protocol.h"

namespace lpp {

class Grammar {
 public:
  Grammar(const Protocol& p, std::set<int> require_before_eos, bool allow_special = false, bool draft = false);
  std::vector<int> allowed_events() const;      // sorted event ids
  void step(int event);                          // throws if not allowed
  void force_done() { done_ = true; }            // forced end when out of records
  bool done() const { return done_; }
  int current() const { return current_; }

 private:
  const Protocol& p_;
  std::set<int> require_;
  bool allow_special_, draft_;
  bool done_ = false;
  int current_ = -1, count_ = 0;
  bool special_ = false, refining_ = false;   // refining_: the special marker was a REFINE
  std::map<int, int> seen_;
  int prim_, eos_;
  std::vector<int> layer_ev_, redraw_ev_, refine_ev_;
};

}  // namespace lpp
