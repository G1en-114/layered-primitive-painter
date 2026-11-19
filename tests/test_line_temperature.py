"""SampleConfig.line_temperature: per-row temperature in _sample (LINE rows vs Gaussian rows)."""
import torch

from lpp.sample import SampleConfig, _sample


def test_row_temperature_matches_scalar_and_greedy():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 50, generator=g)
    u = torch.rand(6, generator=g)
    cfg = SampleConfig(temperature=1.0, top_p=0.9)
    plain = _sample(logits, cfg, u, "x")
    same = _sample(logits, cfg, u, "x", torch.ones(6))                 # all rows at the base temperature
    assert torch.equal(plain, same)
    cold = _sample(logits, cfg, u, "x", torch.full((6,), 1e-6))         # near zero: the top token
    assert torch.equal(cold, logits.argmax(-1))
    mixed = _sample(logits, cfg, u, "x", torch.tensor([1e-6, 1, 1e-6, 1, 1e-6, 1]))
    assert torch.equal(mixed[::2], logits.argmax(-1)[::2]) and torch.equal(mixed[1::2], plain[1::2])
