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


def test_checkpoint_rotation_survives_a_truncated_last(tmp_path):
    import io
    torch.save({"step": 1}, tmp_path / "x")       # warm import paths
    log = io.StringIO()
    assert train.save_checkpoint(tmp_path, {"step": 500, "w": torch.zeros(1000)}, log)
    assert train.save_checkpoint(tmp_path, {"step": 1000, "w": torch.zeros(1000)}, log)
    assert (tmp_path / "prev.pt").exists()
    (tmp_path / "last.pt").write_bytes(b"")          # a save interrupted by a full disk
    orig = torch.load
    train.torch.load = lambda f, **kw: orig(f, map_location="cpu", weights_only=False)
    try:
        assert train.load_checkpoint(tmp_path)["step"] == 500
    finally:
        train.torch.load = orig
