

def test_cached_text_encoder_matches_encoder_under_bf16():
    """Cached features equal the encoder's after the bf16 cast text_proj applies under autocast."""
    import torch
    from lpp.text import CachedTextEncoder

    class Fake(torch.nn.Module):            # deterministic stand-in with the TextEncoder interface
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 8)
            self.model, self.dim = self.lin, 8

        def forward(self, caps):
            B, L = len(caps), max((len(c) for c in caps if c), default=0)
            feats, valid = torch.zeros(B, L, 8), torch.zeros(B, L, dtype=torch.bool)
            for i, c in enumerate(caps):
                if c:
                    x = torch.tensor([[float(ord(ch) % 97)] for ch in c])
                    feats[i, :len(c)], valid[i, :len(c)] = self.lin(x).detach(), True
            return feats[:, :L], valid[:, :L]

    enc = Fake()
    cached = CachedTextEncoder(enc, max_entries=3)
    batches = [["猫", None, "金发女性", "猫"], ["光头男性", "猫", None], [None, None], ["a", "b", "c", "d", "猫"]]
    for caps in batches:
        f1, v1 = enc(caps)
        f2, v2 = cached(caps)
        assert torch.equal(v1, v2)
        assert torch.equal(f1.to(torch.bfloat16), f2.to(torch.bfloat16))
    assert cached.hits >= 2 and len(cached.store) <= 3
