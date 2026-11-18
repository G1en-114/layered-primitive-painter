import torch

from lpp.render.blend import multiply, screen


def layer(color, coverage):
    """A flat layer: colour [3], coverage [H, W] -> premultiplied (P, T)."""
    A = coverage[None, None]
    return torch.tensor(color).view(1, 3, 1, 1) * A, 1 - A


def test_multiply_darkens_and_tints_inside_the_layer_only():
    cov = torch.zeros(4, 4)
    cov[:, :2] = 1.0
    P, T = layer([0.8, 0.6, 0.4], cov)
    a = torch.ones(1, 1, 4, 4) * 0.5
    out = multiply(P, a, torch.tensor([[[0.5, 0.5, 1.0]]]))         # half-strength bluish shadow
    assert torch.allclose(out[0, :, 0, 0], torch.tensor([0.8 * 0.75, 0.6 * 0.75, 0.4]))
    assert (out[0, :, :, 2:] == 0).all()                            # clipped: the empty half stays empty
    assert (out <= P + 1e-7).all()                                  # never lightens


def test_screen_lightens_towards_the_stroke_colour_and_respects_coverage():
    cov = torch.full((2, 2), 0.5)                                   # half-covered layer
    P, T = layer([0.2, 0.2, 0.2], cov)
    out = screen(P, T, torch.ones(1, 1, 2, 2), torch.tensor([[[1.0, 1.0, 1.0]]]))
    assert torch.allclose(out, 1 - T)                               # full white highlight: C' = 1, P' = A
    assert (out >= P).all()


def test_multiply_strokes_commute():
    P, T = layer([0.9, 0.7, 0.5], torch.ones(3, 3))
    a = torch.rand(1, 2, 3, 3)
    c = torch.rand(1, 2, 3)
    assert torch.allclose(multiply(P, a, c), multiply(P, a.flip(1), c.flip(1)), atol=1e-6)
