"""Long descriptions must reach the encoder before being pooled to the painter prefix."""
from types import SimpleNamespace

import pytest
import torch

from lpp.text import TextEncoder


@pytest.fixture
def encoder_factory(monkeypatch):
    import transformers

    class Inputs(dict):
        def to(self, device):
            return Inputs({k: v.to(device) for k, v in self.items()})

    class Tokenizer:
        def __call__(self, captions, *, padding, truncation, return_tensors):
            assert not truncation
            lengths = [int(c) for c in captions]
            ids = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
            valid = torch.zeros_like(ids)
            for i, n in enumerate(lengths):
                ids[i, :n] = torch.arange(1, n + 1)
                valid[i, :n] = 1
            return Inputs(input_ids=ids, attention_mask=valid)

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=512)
            self.seen_length = None

        def forward(self, input_ids, attention_mask):
            self.seen_length = input_ids.shape[1]
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, 4))

    monkeypatch.setattr(transformers.BertTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(transformers.ChineseCLIPModel, "from_pretrained",
                        lambda *a, **kw: SimpleNamespace(text_model=Tower()))
    return lambda **kwargs: TextEncoder("unused", **kwargs)


def test_long_caption_tail_survives_pooling(encoder_factory):
    encoder = encoder_factory(max_positions=32)
    features, valid = encoder(["143", None, "4"])
    assert encoder.model.seen_length == 143
    assert features.shape == (3, 32, 4)
    assert valid.sum(1).tolist() == [32, 0, 4]
    assert features[0, -1, 0] > 140
    assert torch.equal(features[2, :4, 0], torch.arange(1, 5).float())
    assert torch.count_nonzero(features[1]) == 0


def test_over_capacity_is_explicit(encoder_factory):
    encoder = encoder_factory()
    with pytest.raises(ValueError, match="caption needs 513 tokens"):
        encoder(["513"])
    assert encoder.model.seen_length is None


def test_explicit_limit_is_not_silently_truncated(encoder_factory):
    with pytest.raises(ValueError, match="text encoder limit is 96"):
        encoder_factory(max_tokens=96)(["143"])


def test_empty_conditions_do_not_run_encoder(encoder_factory):
    encoder = encoder_factory()
    features, valid = encoder([None, ""])
    assert features.shape == (2, 0, 4) and valid.shape == (2, 0)
    assert encoder.model.seen_length is None
