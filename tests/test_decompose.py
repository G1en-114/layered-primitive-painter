import pytest
import torch

from lpp.codec import Codec
from lpp.decompose import DecomposeConfig, LayeredDecomposer
from lpp.decompose.fill import morton_code, push_pull_fill
from lpp.render.reference import render_layers, stack_layers
from lpp.synthetic import synthetic_scene

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_push_pull_keeps_valid_and_fills_rest():
    img = torch.rand(2, 3, 37, 37)
    valid = (torch.rand(2, 1, 37, 37) > 0.7).float()
    out = push_pull_fill(img, valid)
    assert torch.equal(out[valid.expand_as(img) > 0], img[valid.expand_as(img) > 0])
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_morton_order():
    xs = torch.tensor([0, 1, 0, 1, 2])
    ys = torch.tensor([0, 0, 1, 1, 0])
    assert morton_code(xs, ys).tolist() == [0, 1, 2, 3, 4]


@pytest.fixture(scope="module")
def result(proto):
    try:
        codec = Codec.load(device=DEV)
    except FileNotFoundError:
        pytest.skip("run tools/build_prior_codebooks.py first")
    size = 64
    img, label, _, _ = synthetic_scene(2, size, seed=1, device=DEV)
    cfg = DecomposeConfig(size=size, round_sigmas=(48.0, 16.0, 6.0), round_fracs=(0.2, 0.4, 0.4),
                          steps_per_round=40, budget_scale=0.8)
    out = LayeredDecomposer(codec, cfg)(img, label)
    return img, label, codec, out


def test_output_is_well_formed(proto, result):
    _, _, _, out = result
    for d in out:
        n = d.layer.numel()
        assert n > 0 and all(t.numel() == n for t in d.tokens.values())
        for name, t in d.tokens.items():
            assert 0 <= t.min() and t.max() < proto.attr_vocab_sizes[name]
        assert (d.layer.diff() >= 0).all(), "layers must be back to front"
        for k in d.layer.unique():
            assert (d.round[d.layer == k].diff() >= 0).all(), "rounds must be coarse to fine"
            assert (d.layer == k).sum() <= proto.layers[k].max_primitives


def test_tokens_reproduce_reported_quality(result):
    """Re-rendering the emitted tokens gives the PSNR the decomposer reported: the
    training target is exactly what was fitted (quantization-aware fitting)."""
    img, _, codec, out = result
    for b, d in enumerate(out):
        prims = codec.decode({n: t.unsqueeze(0).to(DEV) for n, t in d.tokens.items()})
        layers = render_layers(prims, d.layer.unsqueeze(0).to(DEV), img.shape[-1])
        mse = ((stack_layers(layers) - img[b : b + 1]) ** 2).mean()
        assert abs(-10 * torch.log10(mse).item() - d.psnr) < 0.05


def test_fit_beats_flat_color_and_respects_layers(result):
    img, _, _, out = result
    for b, d in enumerate(out):
        flat = -10 * torch.log10(((img[b] - img[b].mean((1, 2), keepdim=True)) ** 2).mean()).item()
        assert d.psnr > flat + 6, (d.psnr, flat)
        assert max(d.layer_leak[1:]) < 0.15, d.layer_leak
