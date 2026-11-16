"""Image conditioning: prefix plumbing in collate, model and sampler."""
import numpy as np
import torch

from lpp.codec import Codec
from lpp.data.sequences import collate
from lpp.model import ModelConfig, Painter
from lpp.sample import SampleConfig, generate
from lpp.sequence import build
from test_sequence import fake_decomposition

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _model(codec, image: bool):
    torch.manual_seed(0)
    cfg = ModelConfig.tiny()
    cfg.image, cfg.image_dim = image, 16
    return Painter(cfg, codec.proto, codec).to(DEV)


def test_collate_marks_rows_without_image_invalid():
    seq = build(*fake_decomposition(1))
    img = np.ones((5, 16), np.float16)
    b = collate([(seq, None, img), (seq, None, None)])
    assert b["image"].shape == (2, 5, 16) and b["image_valid"].tolist() == [[True] * 5, [False] * 5]
    assert "image" not in collate([(seq, None), (seq, None)])


def test_image_prefix_changes_predictions_and_trains():
    codec = Codec.load(device=DEV)
    m = _model(codec, image=True)
    seq = build(*fake_decomposition(2))
    imgs = [np.random.default_rng(i).normal(size=(5, 16)).astype(np.float16) for i in range(2)]
    b = {k: v.to(DEV) for k, v in collate([(seq, None, imgs[0]), (seq, None, imgs[1])]).items()}
    lp = m.record_logprobs(b)
    assert not torch.allclose(lp[0], lp[1])                # same records, different images
    loss = m(b)["loss"]
    loss.backward()
    assert m.image_proj.weight.grad.abs().sum() > 0
    # no image rows: identical to omitting the prefix
    b2 = {k: v.to(DEV) for k, v in collate([(seq, None, None), (seq, None, None)]).items()}
    assert torch.allclose(m.record_logprobs(b2)[0], m.record_logprobs(b2)[1])


def test_generate_with_image_prefix_runs_and_guards():
    codec = Codec.load(device=DEV)
    m = _model(codec, image=True).eval()
    img = torch.randn(2, 5, 16, device=DEV)
    seqs = generate(m, batch=2, cfg=SampleConfig(max_records=30), image=img,
                    image_valid=torch.ones(2, 5, dtype=torch.bool, device=DEV),
                    generator=torch.Generator(device=DEV).manual_seed(0), codec=codec)
    assert len(seqs) == 2
    plain = _model(codec, image=False).eval()
    try:
        generate(plain, batch=2, cfg=SampleConfig(max_records=30), image=img,
                 image_valid=torch.ones(2, 5, dtype=torch.bool, device=DEV), codec=codec)
    except ValueError:
        pass
    else:
        raise AssertionError("image tokens accepted by a model without image conditioning")


def _target_model(codec):
    torch.manual_seed(0)
    cfg = ModelConfig.tiny()
    cfg.target, cfg.target_size, cfg.target_gate_init = True, 32, 0.0   # exact no-op at init for this test
    cfg.canvas_every = 1                       # tiny has 2 layers: cross-attention at every layer
    m = Painter(cfg, codec.proto, codec).to(DEV)
    assert len(m.target_cross) == cfg.n_layers
    return m


def test_collate_targets():
    seq = build(*fake_decomposition(1))
    t = np.full((32, 32, 3), 255, np.uint8)
    b = collate([(seq, None, None, t), (seq, None, None, None)])
    assert b["target"].shape == (2, 3, 32, 32) and b["target_valid"].tolist() == [True, False]
    assert float(b["target"][0].max()) == 1.0 and float(b["target"][1].abs().sum()) == 0.0


def test_target_cross_starts_as_noop_then_conditions_and_trains():
    codec = Codec.load(device=DEV)
    m = _target_model(codec)
    seq = build(*fake_decomposition(2))
    rng = np.random.default_rng(0)
    tg = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
    b = {k: v.to(DEV) for k, v in collate([(seq, None, None, tg[0]), (seq, None, None, tg[1])]).items()}
    plain = {k: v for k, v in b.items() if not k.startswith("target")}
    with torch.no_grad():
        assert torch.allclose(m.record_logprobs(b), m.record_logprobs(plain), atol=1e-6)   # gates at 0
        for c in m.target_cross.values():
            c.gate.fill_(0.5)
        lp = m.record_logprobs(b)
    assert (lp[0] - lp[1]).abs().max() > 1e-4                       # same records, different targets
    m(b)["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.target_enc.parameters())


def test_generate_with_target_runs():
    codec = Codec.load(device=DEV)
    m = _target_model(codec).eval()
    seqs = generate(m, batch=2, cfg=SampleConfig(max_records=20), target=torch.rand(2, 3, 32, 32, device=DEV),
                    generator=torch.Generator(device=DEV).manual_seed(0), codec=codec)
    assert len(seqs) == 2
