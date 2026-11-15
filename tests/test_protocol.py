import pytest
import torch

from lpp.protocol import UniformAttr


def test_budget_fits_context(proto):
    assert proto.worst_case_positions() <= proto.max_positions


def test_events_cover_every_layer(proto):
    ids = {proto.layer_event(k) for k in range(proto.num_layers)}
    ids |= {proto.redraw_event(k) for k in range(proto.num_layers)}
    ids |= {proto.refine_event(k) for k in range(proto.num_layers)}
    assert len(ids) == 3 * proto.num_layers
    assert proto.events[-1] == "EOS"
    assert proto.event_id("PRIMITIVE") == 0


def test_vocab_sizes(proto):
    assert proto.version == 2
    assert proto.attr_vocab_sizes == {"x": 512, "y": 512, "sigma_x": 128, "sigma_y": 128, "theta": 128,
                                      "beta": 8, "color": 4096, "alpha": 16}


def test_v1_still_loads():
    from lpp.protocol import REPO_ROOT, load_protocol
    v1 = load_protocol(REPO_ROOT / "protocols" / "v1.json")
    assert v1.version == 1 and v1.attr_names == ("x", "y", "shape", "color", "alpha")
    assert v1.beta_levels == (1.0, 2.0, 4.0, 8.0)


def test_periodic_and_log_attrs(proto):
    import math
    th = proto.attr("theta")
    assert th.quantize(torch.tensor([math.pi - 1e-4, -1e-4, 0.0])).tolist() == [0, 0, 0]
    sx = proto.attr("sigma_x")
    assert sx.dequantize(torch.tensor(0)).item() == pytest.approx(0.5)
    assert sx.dequantize(torch.tensor(127)).item() == pytest.approx(128.0, rel=1e-5)


def test_uniform_roundtrip_and_ends(proto):
    for name in ("x", "y", "alpha", "sigma_x", "theta", "beta"):
        a = proto.attr(name)
        idx = torch.arange(a.bins)
        assert torch.equal(a.quantize(a.dequantize(idx)), idx)
        if isinstance(a, UniformAttr):
            assert abs(a.dequantize(torch.tensor(0)).item() - a.min) < 1e-6
            assert abs(a.dequantize(torch.tensor(a.bins - 1)).item() - a.max) < 1e-5
        # out-of-range values never produce invalid tokens
        assert all(0 <= t < a.bins for t in a.quantize(torch.tensor([-1e9, 1e-9, 1e9])).tolist())


def test_quantization_error_bounded(proto):
    a = proto.attr("x")
    v = torch.rand(10000) * (a.max - a.min) + a.min
    assert (a.dequantize(a.quantize(v)) - v).abs().max() <= a.step / 2 + 1e-5
