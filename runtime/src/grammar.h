// Port of lpp.sequence.Grammar (decoding use: no REDRAW/REFINE from BOS, optional EOS rules).
#pragma once
#include <map>
#include <set>
#include <vector>

#include "protocol.h"

namespace lpp {

class Grammar {
 public:
  Grammar(const Protocol& p, std::set<int> require_before_eos, bool allow_special = false);
  std::vector<int> allowed_events() const;      // sorted event ids
  void step(int event);                          // throws if not allowed
  void force_done() { done_ = true; }            // forced end when out of records
  bool done() const { return done_; }
  int current() const { return current_; }

 private:
  const Protocol& p_;
  std::set<int> require_;
  bool allow_special_;
  bool done_ = false;
  int current_ = -1, count_ = 0;
  bool special_ = false;
  std::map<int, int> seen_;
  int prim_, eos_;
  std::vector<int> layer_ev_, redraw_ev_, refine_ev_;
};

}  // namespace lpp
