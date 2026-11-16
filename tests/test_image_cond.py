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
