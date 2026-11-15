import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from conftest import random_prims
from lpp.render.reference import render, visible_mass

spec = importlib.util.spec_from_file_location("prune_sequences", Path(__file__).resolve().parents[1] /
                                            "tools/prune_sequences.py")
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)


def test_chunked_visibility_matches_dense_suffix(proto):
    small = replace(proto, canvas_size=24)
    prims = random_prims(2, 23)
    prims.xy /= 256 / 24
    prims.sigma /= 256 / 24
    expected = visible_mass(prims, proto=small)
    assert torch.allclose(prune.visible_masses(prims, small, chunk=7), expected, atol=2e-5, rtol=2e-5)


def test_visibility_selection_preserves_order_and_nonempty():
    mass = torch.tensor([0.1, 2., 0.5, 4.])
    assert prune.select_rows(mass, 2).tolist() == [1, 3]
    assert prune.select_rows(mass, 100).tolist() == [3]
    with pytest.raises(ValueError, match="empty or non-finite"):
        prune.select_rows(torch.empty(0), 2)


def test_reconstruction_metric_responds_to_removed_primitive(proto, monkeypatch):
    monkeypatch.setenv("LPP_RASTER", "torch")
    small = replace(proto, canvas_size=24)
    prims = random_prims(1, 4)
    prims.xy /= 256 / 24
    prims.sigma /= 256 / 24
    target = render(prims, proto=small)
    assert prune.reconstruction_psnr(prims, target, small) == pytest.approx(100.)
    assert prune.reconstruction_psnr(prims[:2], target, small) < 99


def test_pruning_config_refuses_changed_threshold(tmp_path):
    path = tmp_path / "config.json"
    prune.write_config(path, {"min_visible_px": 2})
    with pytest.raises(ValueError, match="output config differs"):
        prune.write_config(path, {"min_visible_px": 3})
    assert json.loads(path.read_text()) == {"min_visible_px": 2}
