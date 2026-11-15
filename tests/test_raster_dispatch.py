"""Backend selection must work on CPU and on DCU builds without Triton."""
from types import SimpleNamespace

import pytest
import torch

from lpp.render import dispatch, native_raster


@pytest.mark.parametrize("gpu,dtype,triton,native,expected", [
    (False, torch.float32, True, True, "torch"),
    (True, torch.float64, True, True, "torch"),
    (True, torch.float32, True, True, "triton"),
    (True, torch.float32, False, True, "native"),
    (True, torch.float32, False, False, "torch"),
])
def test_automatic_selection(monkeypatch, gpu, dtype, triton, native, expected):
    monkeypatch.delenv("LPP_RASTER", raising=False)
    monkeypatch.setattr(dispatch, "triton_available", lambda: triton)
    monkeypatch.setattr(native_raster, "available", lambda: native)
    monkeypatch.setattr(dispatch, "render_group_torch", lambda *a: "torch")
    monkeypatch.setattr(native_raster, "render_group_native", lambda *a: "native")
    # Avoid importing Triton on the DCU just to test the dispatch decision.
    import sys
    monkeypatch.setitem(sys.modules, "lpp.render.triton_raster",
                        SimpleNamespace(render_group_triton=lambda *a: "triton"))
    prims = SimpleNamespace(xy=SimpleNamespace(is_cuda=gpu, dtype=dtype))
    assert dispatch.render_group(prims, 32) == expected


def test_explicit_backend_overrides_environment(monkeypatch):
    monkeypatch.setenv("LPP_RASTER", "triton")
    monkeypatch.setattr(dispatch, "render_group_torch", lambda *a: "torch")
    assert dispatch.render_group(None, 32, backend="torch") == "torch"


def test_explicit_native_failure_is_visible(monkeypatch):
    monkeypatch.setenv("LPP_RASTER", "native")

    def fail(*args):
        raise RuntimeError("native build failed")

    monkeypatch.setattr(native_raster, "render_group_native", fail)
    with pytest.raises(RuntimeError, match="native build failed"):
        dispatch.render_group(None, 32)


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown rasterizer backend"):
        dispatch.render_group(None, 32, backend="typo")
