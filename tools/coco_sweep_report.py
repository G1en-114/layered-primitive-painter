"""Summarise the COCO budget sweep (docs/protocol_v3_design.md section 3): PSNR and primitives per group.

    python tools/coco_sweep_report.py --data data/coco256_val_v1 --runs data/seq/coco_pilot_C*
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    groups = json.loads((args.data / "pilot_groups.json").read_text())
    group_of = {i: g for g, ids in groups.items() for i in ids}
    report = {}
    for run in sorted(args.runs, key=lambda p: int(str(p).rsplit("C", 1)[-1])):
        per = {g: {"psnr": [], "prims": []} for g in groups}
        for f in sorted(run.glob("decomp_*.npz")):
            with np.load(f) as z:
                n = np.diff(z["offsets"])
                for image_id, p, k in zip(z["ids"].tolist(), z["psnr"].tolist(), n.tolist()):
                    per[group_of[image_id]]["psnr"].append(p)
                    per[group_of[image_id]]["prims"].append(k)
        report[run.name] = {g: {"images": len(v["psnr"]), "psnr": round(float(np.mean(v["psnr"])), 3),
                                "psnr_p10": round(float(np.percentile(v["psnr"], 10)), 3),
                                "primitives": round(float(np.mean(v["prims"])), 1),
                                "primitives_max": int(np.max(v["prims"]))}
                            for g, v in per.items() if v["psnr"]}
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
