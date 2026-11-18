"""C++ runtime with protocol v3 (LINE strokes): its sequences are grammatical and its PNGs equal
lpp.render.strokes.render_strokes on the records it wrote (per layer: Gaussians, then lines).
Needs the built runtime and a v3 ONNX export under out/onnx (e.g. a tiny random model)."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from lpp.codec import Codec
from lpp.protocol import _parse
from lpp.render.strokes import render_strokes
from lpp.sequence import DrawSeq, validate

ROOT = Path(__file__).resolve().parents[1]
PAINT = ROOT / "runtime/build/lpp_paint"
V3 = [p.parent for p in sorted((ROOT / "out/onnx").glob("*/manifest.json"))
      if json.loads(p.read_text())["protocol"].get("version") == 3]


@pytest.mark.skipif(not PAINT.exists() or not V3, reason="runtime not built or no v3 ONNX export")
def test_cpp_v3_sequences_and_renders_match_python(tmp_path):
    export = V3[0]
    man = json.loads((export / "manifest.json").read_text())
    proto = _parse(man["protocol"])
    B, Lt, D, steps = 3, 4, man["config"]["text_dim"], 160
    rng = np.random.default_rng(0)
    np.save(tmp_path / "text.npy", rng.standard_normal((B, Lt, D)).astype(np.float32))
    np.save(tmp_path / "tv.npy", np.ones((B, Lt), bool))
    np.save(tmp_path / "u.npy", rng.random((steps, 1 + len(proto.attr_names), B)).astype(np.float32))
    subprocess.run([str(PAINT), "--onnx", str(export), "--text", str(tmp_path / "text.npy"), "--text-valid",
                    str(tmp_path / "tv.npy"), "--uniforms", str(tmp_path / "u.npy"), "--out", str(tmp_path / "o"),
                    "--max-records", str(steps)], check=True, capture_output=True)
    rec, lens = np.load(tmp_path / "o/records.npy"), np.load(tmp_path / "o/lengths.npy")
    codec = Codec.load(proto=proto, device="cuda" if torch.cuda.is_available() else "cpu")
    line = proto.event_id("LINE")
    n_lines = 0
    for b in range(B):
        r = rec[b, : lens[b]]
        validate(DrawSeq(torch.from_numpy(r[:, 0]), torch.from_numpy(r[:, 1]), torch.from_numpy(r[:, 3:]),
                         torch.from_numpy(r[:, 2])), proto, draft=man["config"].get("order") == "draft")
        st = r[:, 0] == 3
        n_lines += int((r[st, 1] == line).sum())
        img = render_strokes(torch.from_numpy(r[st, 3:]), torch.from_numpy(r[st, 2]), torch.from_numpy(r[st, 1] == line), codec)[0]
        py = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.int16)
        cpp = np.array(Image.open(tmp_path / f"o/sample_{b}.png")).astype(np.int16)
        assert np.abs(py - cpp).max() <= 2, b
    assert n_lines > 0
