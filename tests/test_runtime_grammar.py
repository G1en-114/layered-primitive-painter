"""C++ Grammar (runtime/src/grammar.cpp) == lpp.sequence.Grammar, on random walks.

The C++ tool walks its grammar with random choices and prints the allowed events at every
step; the Python grammar replays the same choices and must allow exactly the same events.
Needs the built runtime (runtime/build/lpp_grammar_walk) and one ONNX export for the protocol.
"""
import json
import subprocess
from pathlib import Path

import pytest

from lpp.protocol import _parse
from lpp.sequence import Grammar

ROOT = Path(__file__).resolve().parents[1]
WALK = ROOT / "runtime/build/lpp_grammar_walk"
EXPORTS = sorted(p.parent for p in (ROOT / "out/onnx").glob("*/manifest.json"))


@pytest.mark.skipif(not WALK.exists() or not EXPORTS, reason="runtime not built or no ONNX export")
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("require", [(), (0, 2)])
def test_cpp_grammar_matches_python(draft, require):
    export = EXPORTS[0]
    proto = _parse(json.loads((export / "manifest.json").read_text())["protocol"])
    steps = refines_chained = 0
    for seed in range(150):
        out = subprocess.run([str(WALK), str(export), str(seed), str(int(draft)), *map(str, require)],
                             capture_output=True, text=True, check=True).stdout
        g = Grammar(proto, require_before_eos=frozenset(require), allow_special=False, draft=draft)
        g.bos()
        refines = 0
        for line in out.splitlines():
            allowed, taken = line.split("\t")
            assert [int(a) for a in allowed.split(",")] == g.allowed_events(), (seed, steps)
            g.step(int(taken))
            refines += proto.events[int(taken)].startswith("REFINE")
            steps += 1
        refines_chained += refines > 1
    if draft:
        assert refines_chained > 0          # the walks did exercise chained REFINE
