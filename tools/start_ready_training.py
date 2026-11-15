"""Wait for one complete cleaned trajectory, verify it, publish and start a finite run.

The launch plan contains explicit local/remote paths and a training argv list. No
training starts on a partial trajectory or after a failed data integrity check.
One lock per local plan and a remote run lock prevent duplicate launches.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np

from run_decomposition_plan import expected_shards, validate_output


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_ready(plan):
    source = json.loads(Path(plan["decomposition_plan"]).read_text())
    source["part"] = "0/1"
    job = source["jobs"][plan.get("trajectory", 0)]
    clean = Path(plan["clean_dir"])
    shards = expected_shards(source)
    names = [f"decomp_{s['shard']:03d}.npz" for s in shards]
    if not all((clean / name).exists() for name in names):
        return None
    images = validate_output(source, dict(job, out=str(clean)))
    cfg = json.loads((clean / "config.json").read_text())
    if cfg.get("pruning_version") != 2 or cfg.get("psnr_metric") != "post_prune_rgb_mse":
        raise ValueError("training requires remeasured post-prune metrics")
    if cfg.get("target_index_sha256") != digest(Path(source["data"]) / "index.json"):
        raise ValueError("cleaned targets use a different source index")
    total_before = total_after = 0
    losses = []
    for name in names:
        raw = Path(job["out"]) / name
        with np.load(clean / name) as z, np.load(raw) as original:
            if str(z["source_sha256"]) != digest(raw):
                raise ValueError(f"{name}: raw source hash mismatch")
            if not np.array_equal(z["source_psnr"], original["psnr"]):
                raise ValueError(f"{name}: original metrics were changed")
            loss = z["unpruned_psnr"] - z["psnr"]
            if not np.isfinite(loss).all() or (loss > plan.get("max_psnr_loss_db", 0.1) + 1e-5).any():
                raise ValueError(f"{name}: pruning exceeds the reconstruction loss limit")
            total_before += len(original["tokens"])
            total_after += len(z["tokens"])
            losses.extend(loss.tolist())
    ids = {i for s in shards for i in s["ids"]}
    captions = [json.loads(line) for line in Path(plan["captions"]).read_text().splitlines() if line.strip()]
    caption_ids = [r["id"] for r in captions]
    if len(caption_ids) != len(set(caption_ids)) or set(caption_ids) != ids:
        raise ValueError("caption IDs must cover the trajectory exactly, without duplicates")
    if not all(r.get("short") and r.get("long") for r in captions):
        raise ValueError("empty training caption")
    attrs = json.loads(Path(plan["attributes"]).read_text())
    if not ids <= {int(i) for i in attrs["values"]}:
        raise ValueError("attributes do not cover all images")
    records = [json.loads(line) for line in Path(plan["split_manifest"]).read_text().splitlines() if line.strip()]
    splits = {r["id"]: r["split"] for r in records}
    if len(splits) != len(records) or set(splits) != ids or set(splits.values()) != {"train", "val"}:
        raise ValueError("split manifest must contain every image once and have train/val partitions")
    stats = {"images": images, "shards": len(names), "primitives_before": total_before,
             "primitives_after": total_after, "mean_psnr_loss_db": float(np.mean(losses)),
             "max_psnr_loss_db": float(np.max(losses)),
             "train_images": sum(v == "train" for v in splits.values()),
             "val_images": sum(v == "val" for v in splits.values())}
    return names, stats


REMOTE_LAUNCH = r'''
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin)
for name,expected in p['sha256'].items():
 path=Path(name)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
  raise ValueError('remote SHA256 mismatch: '+name)
run=Path(p['run_dir']);run.mkdir(parents=True,exist_ok=True)
lock=(run/'.launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record=run/'launch.json'
if record.exists():
 old=json.loads(record.read_text())
 if old['argv']!=p['argv']:raise ValueError('run directory belongs to different arguments')
 proc=Path('/proc/%s/cmdline'%old['pid'])
 args=proc.read_bytes().split(b'\0') if proc.exists() else []
 if os.fsencode(p['run_dir']) in args and b'tools/train.py' in args:
  print(json.dumps(dict(old,already_running=True)));sys.exit(0)
 raise RuntimeError('previous training process ended; inspect its log before resuming')
if (run/'last.pt').exists() or (run/'log.jsonl').exists():
 raise ValueError('refusing to overwrite an existing training run')
with (run/'train.stdout.log').open('a') as log:
 child=subprocess.Popen([sys.executable,'-u',*p['argv']],cwd=p['code_dir'],
  env=dict(os.environ,**p['environment']),stdin=subprocess.DEVNULL,stdout=log,
  stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=child.pid,argv=p['argv'],started_unix=time.time(),stats=p['stats'])
tmp=record.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,record)
print(json.dumps(result))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--wait-hours", type=float, default=3)
    args = ap.parse_args()
    if not 1 <= args.poll <= 60 or args.wait_hours <= 0:
        ap.error("poll must be 1..60 seconds and wait-hours positive")
    lock = args.plan.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    state_file = args.plan.with_suffix(".state.json")

    def state(phase, **details):
        record = dict(phase=phase, time_unix=time.time(), **details)
        tmp = state_file.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, state_file)
        print(json.dumps(record), flush=True)

    try:
        state("waiting_clean_data", clean_dir=plan["clean_dir"])
        deadline = time.monotonic() + args.wait_hours * 3600
        while (ready := check_ready(plan)) is None:
            if time.monotonic() > deadline:
                raise TimeoutError("cleaned trajectory did not complete within wait limit")
            time.sleep(args.poll)
        names, stats = ready
        state("publishing", stats=stats)
        ssh = ["ssh", "-F", "/dev/null", "-S", plan["socket"], "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=12", "-o", "LogLevel=ERROR"]
        host = plan.get("host", "x")
        subprocess.run([*ssh, host, "mkdir -p " + shlex.quote(plan["remote_clean_dir"])],
                       check=True, timeout=30)
        remote_hashes = {}
        transfers = [(str(Path(plan["clean_dir"]) / name), str(Path(plan["remote_clean_dir"]) / name))
                     for name in ["config.json", *names]]
        transfers += [(plan[k], plan["remote_" + k]) for k in ("captions", "attributes", "split_manifest")]
        for local, remote in transfers:
            subprocess.run(["rsync", "-rt", "--checksum", "--timeout=60", "--protect-args",
                            "--partial-dir=.rsync-partial", "-e", shlex.join(ssh), local, f"{host}:{remote}"],
                           check=True, timeout=300)
            remote_hashes[remote] = digest(local)
        for name in plan["verify_code"]:
            remote_hashes[str(Path(plan["code_dir"]) / name)] = digest(name)
        request = {k: plan[k] for k in ("run_dir", "code_dir", "argv", "environment")}
        request.update(sha256=remote_hashes, stats=stats)
        state("starting_training", stats=stats)
        result = subprocess.run([*ssh, host, "python -c " + shlex.quote(REMOTE_LAUNCH)],
                                input=json.dumps(request), text=True, capture_output=True,
                                check=True, timeout=90)
        launched = json.loads(result.stdout)
        state("training_launched", remote=launched, run_dir=plan["run_dir"])
    except Exception as exc:
        state("failed", error=str(exc), stderr=getattr(exc, "stderr", None))
        raise


if __name__ == "__main__":
    main()
