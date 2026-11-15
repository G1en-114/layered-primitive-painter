import importlib.util
import json
import sys
from pathlib import Path

import pytest

import test_decomposition_plan as plans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
spec = importlib.util.spec_from_file_location("sync_plan", Path(__file__).resolve().parents[1] /
                                            "tools/sync_decomposition_plan.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


@pytest.fixture
def finished_plan(tmp_path):
    return plans.finished_plan.__wrapped__(tmp_path)


def test_remote_paths_normalized_without_changing_protocol(finished_plan):
    plan, job = finished_plan
    job["name"] = "test"
    config = json.loads((Path(job["out"]) / "config.json").read_text())
    remote = dict(config, protocol="/remote/protocol.json", data="/remote/images")
    normalized = sync.canonical_config(plan, job, remote)
    assert normalized["data"] == plan["data"]
    assert normalized["protocol"] == plan["protocol"]
    for key, value in {"seed": 99, "flip": True, "protocol_sha256": "bad"}.items():
        with pytest.raises(ValueError, match="remote trajectory"):
            sync.canonical_config(plan, job, dict(remote, **{key: value}))


def test_conflicting_config_is_never_overwritten(tmp_path):
    path = tmp_path / "new" / "config.json"
    sync.ensure_config(path, {"seed": 2})
    sync.ensure_config(path, {"seed": 2})
    with pytest.raises(ValueError, match="configs differ"):
        sync.ensure_config(path, {"seed": 3})
    assert json.loads(path.read_text()) == {"seed": 2}
