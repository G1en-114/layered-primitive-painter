"""Run a finite, resumable decomposition plan, joining existing jobs before advancing.

The JSON plan names the data, protocol, machine's shard partition, and each trajectory.
No training starts automatically. Existing producers are joined by PID plus output path;
their output is validated before another trajectory starts. A flock prevents duplicate
controllers for the same plan. Restart with a new plan after an interrupted external PID.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def expected_shards(plan):
    part, parts = map(int, plan["part"].split("/"))
    if not 0 <= part < parts:
        raise ValueError("invalid shard partition")
    return [s for s in json.loads((Path(plan["data"]) / "index.json").read_text())["shards"]
            if s["shard"] % parts == part]


def validate_output(plan, job):
    out = Path(job["out"])
    protocol = Path(job.get("protocol", plan["protocol"]))
    raw = json.loads(protocol.read_text())
    config = json.loads((out / "config.json").read_text())
    expected = {"steps": plan["steps"], "seed": job["seed"], "flip": job.get("flip", False),
                "protocol_version": raw["version"],
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{out}: output config does not match the planned trajectory")
    if Path(config["data"]).resolve() != Path(plan["data"]).resolve():
        raise ValueError(f"{out}: output belongs to different image data")
    count = 0
    for shard in expected_shards(plan):
        path = out / f"decomp_{shard['shard']:03d}.npz"
        with np.load(path) as z:
            ids, offsets, tokens, layers, rounds, psnr = (z[k] for k in
                                                          ("ids", "offsets", "tokens", "layer", "round", "psnr"))
            if ids.tolist() != shard["ids"] or len(offsets) != len(ids) + 1:
                raise ValueError(f"{path}: image IDs / offsets do not match the source")
            if offsets[0] != 0 or offsets[-1] != len(tokens) or (np.diff(offsets) <= 0).any():
                raise ValueError(f"{path}: invalid primitive offsets")
            if tokens.ndim != 2 or tokens.shape[1] != len(raw["attributes"]):
                raise ValueError(f"{path}: wrong attribute width")
            if len(layers) != len(tokens) or len(rounds) != len(tokens) or len(psnr) != len(ids):
                raise ValueError(f"{path}: inconsistent array lengths")
            if not np.isfinite(psnr).all():
                raise ValueError(f"{path}: non-finite reconstruction metric")
            count += len(ids)
    return count


def producer_alive(pid, output):
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    if not args:
        return False
    # Match an actual argv item, not a shell containing similar text.
    return b"--out" in args and os.fsencode(output) in args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.poll <= 60:
        parser.error("poll must be 1..60 seconds")
    plan = json.loads(args.plan.read_text())
    lock = args.plan.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.plan.with_suffix(".state.json")

    def status(job, phase, **details):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "job": job["name"],
                  "output": job["out"], "phase": phase, **details}
        tmp = state_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, state_path)
        print(json.dumps(record), flush=True)

    env = dict(os.environ, **plan.get("environment", {}))
    for job in plan["jobs"]:
        try:
            if job.get("wait_pid"):
                status(job, "waiting_existing_producer", pid=job["wait_pid"])
                while producer_alive(job["wait_pid"], job["out"]):
                    time.sleep(args.poll)
            shards = expected_shards(plan)
            all_present = all((Path(job["out"]) / f"decomp_{s['shard']:03d}.npz").exists() for s in shards)
            if not all_present:
                if job.get("join_only"):
                    raise RuntimeError("existing producer ended before completing all required shards")
                command = [sys.executable, "-u", str(Path(__file__).with_name("decompose_dataset.py")),
                           "--data", plan["data"], "--out", job["out"], "--steps", str(plan["steps"]),
                           "--batch", str(plan["batch"]), "--seed", str(job["seed"]), "--part", plan["part"],
                           "--protocol", job.get("protocol", plan["protocol"])]
                if job.get("flip"):
                    command.append("--flip")
                with Path(job["log"]).open("a") as log:
                    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT)
                    status(job, "running", pid=child.pid, log=job["log"])
                    code = child.wait()
                if code:
                    raise RuntimeError(f"decomposer exited with status {code}")
            count = validate_output(plan, job)
            status(job, "complete_validated", images=count, shards=len(shards))
        except Exception as exc:
            status(job, "failed", error=str(exc))
            raise
    print("FINITE_PLAN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
