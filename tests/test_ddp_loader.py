"""The DDP data path: every rank reads a disjoint slice of each epoch, together covering it."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import torch  # noqa: E402

import train  # noqa: E402


class Ids(torch.utils.data.Dataset):
    def __len__(self):
        return 40

    def __getitem__(self, i):
        return i


def epoch(rank, world, batch=4, seed=7):
    args = SimpleNamespace(batch=batch, workers=0, seed=seed)
    original = train.collate
    train.collate = lambda xs: list(xs)            # the real collate expects sequences
    try:
        loader = train.loader(Ids(), args, rank=rank, world=world)
        out = []
        for _ in range(40 // world // batch):
            out += next(loader)
        return out
    finally:
        train.collate = original


def test_ranks_disjoint_and_cover_epoch():
    a, b = epoch(0, 2), epoch(1, 2)
    assert len(a) == len(b) == 20
    assert not set(a) & set(b)
    assert set(a) | set(b) == set(range(40))


def test_single_process_unchanged():
    assert sorted(epoch(0, 1)) == list(range(40))
