// Random walks through the C++ Grammar, for the cross-check against lpp.sequence.Grammar
// (tests/test_runtime_grammar.py). Usage: lpp_grammar_walk EXPORT_DIR SEED DRAFT(0/1) [REQUIRED_LAYER...]
// Prints one line per step: the allowed event ids, a tab, the event taken (uniform among the allowed).
#include <cstdio>
#include <random>
#include <set>
#include <string>

#include "grammar.h"
#include "protocol.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s EXPORT_DIR SEED DRAFT [REQUIRED_LAYER...]\n", argv[0]);
    return 2;
  }
  const lpp::Protocol P = lpp::Protocol::load(argv[1]);
  std::mt19937 rng(static_cast<unsigned>(std::stoul(argv[2])));
  const bool draft = std::stoi(argv[3]) != 0;
  std::set<int> require;
  for (int i = 4; i < argc; ++i) require.insert(std::stoi(argv[i]));
  lpp::Grammar g(P, require, false, draft);
  const int prim = P.event_id("PRIMITIVE");
  for (int step = 0; step < 4000 && !g.done(); ++step) {
    auto allowed = g.allowed_events();
    // favour primitives so walks get deep into layers and refines instead of ending at once
    int ev = allowed[std::uniform_int_distribution<size_t>(0, allowed.size() - 1)(rng)];
    for (int e : allowed)
      if (e == prim && std::uniform_real_distribution<double>(0, 1)(rng) < 0.8) ev = prim;
    for (size_t i = 0; i < allowed.size(); ++i) std::printf(i ? ",%d" : "%d", allowed[i]);
    std::printf("\t%d\n", ev);
    g.step(ev);
  }
  return 0;
}
