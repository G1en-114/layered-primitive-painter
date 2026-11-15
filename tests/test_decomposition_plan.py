import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("decomposition_plan",
                                            Path(__file__).resolve().parents[1] / "tools/run_decomposition_plan.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.fixture
def finished_plan(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    (data / "index.json").write_text(json.dumps({"shards": [{"shard": 0, "ids": [10, 11]},
                                                           {"shard": 1, "ids": [12]}]}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"version": 2, "attributes": [{"name": "x"}, {"name": "y"}]}))
    plan = {"data": str(data), "protocol": str(protocol), "part": "0/2", "steps": 80}
    job = {"out": str(out), "seed": 1}
    (out / "config.json").write_text(json.dumps({"data": str(data), "steps": 80, "seed": 1,
                                                 "flip": False, "protocol_version": 2,
                                                 "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}))
    np.savez(out / "decomp_000.npz", ids=[10, 11], offsets=[0, 1, 3], tokens=np.zeros((3, 2), np.int16),
             layer=[0, 1, 1], round=[0, 1, 1], psnr=[30., 31.])
    return plan, job


def test_completed_partition_is_validated(finished_plan):
    plan, job = finished_plan
    assert queue.validate_output(plan, job) == 2


def test_wrong_seed_is_rejected(finished_plan):
    plan, job = finished_plan
    with pytest.raises(ValueError, match="output config"):
        queue.validate_output(plan, dict(job, seed=2))


def test_missing_shard_is_not_complete(finished_plan):
    plan, job = finished_plan
    with pytest.raises(FileNotFoundError):
        queue.validate_output(dict(plan, part="1/2"), job)


def test_corrupted_offsets_are_rejected(finished_plan):
    plan, job = finished_plan
    np.savez(Path(job["out"]) / "decomp_000.npz", ids=[10, 11], offsets=[0, 2, 7],
             tokens=np.zeros((3, 2)), layer=[0, 1, 1], round=[0, 1, 1], psnr=[30., 31.])
    with pytest.raises(ValueError, match="primitive offsets"):
        queue.validate_output(plan, job)
