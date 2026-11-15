import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import test_decomposition_plan as plans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
spec = importlib.util.spec_from_file_location("ready_training", Path(__file__).resolve().parents[1] /
                                            "tools/start_ready_training.py")
ready = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ready)


@pytest.fixture
def launch_plan(tmp_path):
    source, job = plans.finished_plan.__wrapped__(tmp_path)
    data = Path(source["data"])
    index = data / "index.json"
    index.write_text(json.dumps({"shards": [{"shard": 0, "ids": [10, 11]}]}))
    source.update(part="0/1", jobs=[job])
    decomposition_plan = tmp_path / "decomposition.json"
    decomposition_plan.write_text(json.dumps(source))
    raw = Path(job["out"])
    clean = tmp_path / "clean"
    clean.mkdir()
    cfg = json.loads((raw / "config.json").read_text())
    cfg.update(pruning_version=2, psnr_metric="post_prune_rgb_mse",
               target_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
    (clean / "config.json").write_text(json.dumps(cfg))
    with np.load(raw / "decomp_000.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update(source_sha256=np.array(ready.digest(raw / "decomp_000.npz")),
                  source_psnr=arrays["psnr"].copy(), unpruned_psnr=arrays["psnr"] + 0.03)
    np.savez(clean / "decomp_000.npz", **arrays)
    plan = dict(decomposition_plan=str(decomposition_plan), clean_dir=str(clean))
    contents = {
        "captions": '\n'.join(json.dumps({"id": i, "short": "portrait", "long": "a portrait"}) for i in [10, 11]),
        "attributes": json.dumps({"values": {"10": [], "11": []}}),
        "split_manifest": '{"id":10,"split":"train"}\n{"id":11,"split":"val"}\n',
    }
    for key, text in contents.items():
        path = tmp_path / (key + ".jsonl")
        path.write_text(text)
        plan[key] = str(path)
    return plan


def test_partial_trajectory_cannot_start_training(launch_plan):
    (Path(launch_plan["clean_dir"]) / "decomp_000.npz").unlink()
    assert ready.check_ready(launch_plan) is None


def test_complete_cleaned_trajectory_has_expected_split(launch_plan):
    names, stats = ready.check_ready(launch_plan)
    assert names == ["decomp_000.npz"]
    assert (stats["images"], stats["train_images"], stats["val_images"]) == (2, 1, 1)
    assert stats["max_psnr_loss_db"] == pytest.approx(0.03)


def test_source_change_blocks_launch(launch_plan):
    source = json.loads(Path(launch_plan["decomposition_plan"]).read_text())
    path = Path(source["jobs"][0]["out"]) / "decomp_000.npz"
    with path.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="raw source hash mismatch"):
        ready.check_ready(launch_plan)


def test_large_pruning_loss_blocks_launch(launch_plan):
    path = Path(launch_plan["clean_dir"]) / "decomp_000.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["psnr"] -= 0.2
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="reconstruction loss limit"):
        ready.check_ready(launch_plan)


def test_duplicate_caption_ids_block_launch(launch_plan):
    path = Path(launch_plan["captions"])
    path.write_text(path.read_text() + '\n' + path.read_text().splitlines()[0])
    with pytest.raises(ValueError, match="caption IDs"):
        ready.check_ready(launch_plan)
