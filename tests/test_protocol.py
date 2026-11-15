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
    assert proto.attr_vocab_sizes == {"x": 256, "y": 256, "shape": 4096, "color": 4096, "alpha": 16}


def test_uniform_roundtrip_and_ends(proto):
    for name in ("x", "y", "alpha"):
        a = proto.attr(name)
        assert isinstance(a, UniformAttr)
        idx = torch.arange(a.bins)
        assert torch.equal(a.quantize(a.dequantize(idx)), idx)
        assert abs(a.dequantize(torch.tensor(0)).item() - a.min) < 1e-6
        assert abs(a.dequantize(torch.tensor(a.bins - 1)).item() - a.max) < 1e-5
        # out-of-range values clamp instead of producing invalid tokens
        assert a.quantize(torch.tensor([-1e9, 1e9])).tolist() == [0, a.bins - 1]


def test_quantization_error_bounded(proto):
    a = proto.attr("x")
    v = torch.rand(10000) * (a.max - a.min) + a.min
    assert (a.dequantize(a.quantize(v)) - v).abs().max() <= a.step / 2 + 1e-5
