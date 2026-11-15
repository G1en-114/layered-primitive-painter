"""Collect the remote half of a finite trajectory plan, validate, then optionally prune.

Uses rsync partial directories and independently checks SHA256. Remote config paths are
normalized to the local plan without changing seeds, flips or protocol contents. No model
training is started. --watch stops after the finite set of trajectories is complete.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from run_decomposition_plan import expected_shards, validate_output

REMOTE_INVENTORY = r'''
import hashlib,json,sys
from pathlib import Path
request=json.load(sys.stdin)
plan=json.loads(Path(request['plan']).read_text())
part,parts=map(int,plan['part'].split('/'))
index=json.loads((Path(plan['data'])/'index.json').read_text())
allowed={f"decomp_{s['shard']:03d}.npz" for s in index['shards'] if s['shard']%parts==part}
jobs=[]
for job in plan['jobs']:
 root=Path(job['out']);config=root/'config.json';files=[]
 if config.exists():
  for name in sorted(allowed):
   path=root/name
   if not path.exists():continue
   stat=path.stat();signature=[stat.st_size,stat.st_mtime_ns]
   previous=request.get('cache',{}).get(str(path),{})
   if previous.get('signature')==signature:digest=previous['sha256']
   else:
    with path.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
   files.append({'name':name,'path':str(path),'signature':signature,'sha256':digest})
 jobs.append({'seed':job['seed'],'flip':job.get('flip',False),'out':job['out'],
              'config':json.loads(config.read_text()) if config.exists() else None,'files':files})
print(json.dumps({'part':plan['part'],'jobs':jobs}))
'''


def canonical_config(plan, job, remote_config):
    protocol = Path(job.get("protocol", plan["protocol"]))
    expected = {"seed": job["seed"], "flip": job.get("flip", False), "steps": plan["steps"],
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "protocol_version": json.loads(protocol.read_text())["version"]}
    if any(remote_config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"remote trajectory does not match {job['name']}")
    return dict(remote_config, data=plan["data"], protocol=str(protocol))


def ensure_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".incoming.json")
        temporary.write_text(json.dumps(config, indent=1))
        try:
            os.link(temporary, path)  # Create without overwriting a concurrent producer's config.
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    if json.loads(path.read_text()) != config:
        raise ValueError(f"{path}: local and remote trajectory configs differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--remote-plan", required=True)
    parser.add_argument("--socket", default="/run/user/1000/scnet.sock")
    parser.add_argument("--host", default="x")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll", type=float, default=60)
    parser.add_argument("--clean-root", type=Path, default=None,
                        help="after ALL raw trajectories validate, prune here with actual RGB targets")
    args = parser.parse_args()
    if not 1 <= args.poll <= 60:
        parser.error("poll must be 1..60 seconds")
    lock = args.plan.with_suffix(".sync.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    full_plan = dict(plan, part="0/1")
    status_path = args.plan.with_suffix(".sync.json")
    state = json.loads(status_path.read_text()) if status_path.exists() else {"remote_cache": {}, "files": {}}
    ssh = ["ssh", "-F", "/dev/null", "-S", args.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
           "-o", "LogLevel=ERROR"]
    expected = expected_shards(full_plan)

    def save():
        tmp = status_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(tmp, status_path)

    failures = 0
    while True:
        try:
            inventory = subprocess.run([*ssh, args.host, "python -c " + shlex.quote(REMOTE_INVENTORY)],
                                       input=json.dumps({"plan": args.remote_plan, "cache": state["remote_cache"]}),
                                       text=True, capture_output=True, check=True, timeout=120)
            remote = json.loads(inventory.stdout)
            remote_jobs = {(j["seed"], j["flip"]): j for j in remote["jobs"]}
            summaries = []
            for job in plan["jobs"]:
                other = remote_jobs[(job["seed"], job.get("flip", False))]
                out = Path(job["out"])
                if other["config"]:
                    ensure_config(out / "config.json", canonical_config(plan, job, other["config"]))
                missing = []
                for info in other["files"]:
                    path = out / info["name"]
                    state["remote_cache"][info["path"]] = {"signature": info["signature"], "sha256": info["sha256"]}
                    if not path.exists():
                        missing.append(info["name"])
                if missing:
                    with tempfile.NamedTemporaryFile("w", dir=out, prefix=".sync-files-", delete=False) as listing:
                        listing.write("\n".join(missing) + "\n")
                    try:
                        subprocess.run(["rsync", "-rt", "--checksum", "--protect-args", "--timeout=60",
                                        "--partial-dir=.rsync-partial", "--files-from=" + listing.name,
                                        "-e", shlex.join(ssh), f"{args.host}:{other['out']}/", str(out) + "/"],
                                       check=True, timeout=600)
                    finally:
                        Path(listing.name).unlink(missing_ok=True)
                for info in other["files"]:
                    path = out / info["name"]
                    stat = path.stat()
                    signature = [stat.st_size, stat.st_mtime_ns]
                    prior = state["files"].get(str(path), {})
                    if prior.get("signature") != signature or prior.get("sha256") != info["sha256"]:
                        with path.open("rb") as source:
                            digest = hashlib.file_digest(source, "sha256").hexdigest()
                        if digest != info["sha256"]:
                            raise ValueError(f"SHA256 mismatch: {path}; preserve it for investigation")
                        state["files"][str(path)] = {"signature": signature, "sha256": digest}
                count = sum((out / f"decomp_{s['shard']:03d}.npz").exists() for s in expected)
                summary = {"job": job["name"], "present": count, "expected": len(expected),
                           "remote_verified": len(other["files"]), "copied_this_cycle": len(missing)}
                if count == len(expected):
                    summary["validated_images"] = validate_output(full_plan, job)
                summaries.append(summary)
            failures = 0
            changed = state.get("jobs") != summaries
            state.update(jobs=summaries, raw_complete=all("validated_images" in s for s in summaries),
                         phase="collecting", last_check_unix=time.time())
            save()
            if changed:
                print(json.dumps({"phase": state["phase"], "jobs": summaries}), flush=True)
            if state["raw_complete"] or not args.watch:
                break
        except (subprocess.SubprocessError, OSError) as exc:
            failures += 1
            state.update(phase="transfer_retry", error=str(exc), consecutive_failures=failures)
            save()
            print(f"transfer retry {failures}: {exc}", flush=True)
            if failures >= 3 or not args.watch:
                raise
        time.sleep(args.poll)
    if state.get("raw_complete") and args.clean_root:
        env = dict(os.environ, **plan.get("environment", {}))
        for i, job in enumerate(plan["jobs"]):
            destination = args.clean_root / f"t{i}"
            state.update(phase="pruning", cleaning_source=job["out"], cleaning_output=str(destination))
            save()
            with args.clean_root.with_suffix(".log").open("a") as log:
                subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("prune_sequences.py")),
                                "--src", job["out"], "--out", str(destination), "--data", plan["data"],
                                "--protocol", job.get("protocol", plan["protocol"]), "--min-visible", "2.0"],
                               env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_output(full_plan, dict(job, out=str(destination)))
        state.update(phase="clean_complete", clean_root=str(args.clean_root))
    else:
        state["phase"] = "raw_complete" if state.get("raw_complete") else "snapshot_complete"
    save()
    print(json.dumps({"phase": state["phase"], "raw_complete": state.get("raw_complete")}), flush=True)


if __name__ == "__main__":
    main()
