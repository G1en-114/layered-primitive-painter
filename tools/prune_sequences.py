"""Remove (nearly) invisible primitives from decomposition shards.

    python tools/prune_sequences.py --src data/seq/v2 --out data/seq/v2p --min-visible 2.0

A primitive's visible mass = sum over pixels of its opacity times the transmittance of
everything painted after it (render.reference.visible_mass). On v2 face decompositions,
primitives under 2 px are ~4% of all and removing them costs ~0.03 dB: they are noise the
model would otherwise have to learn. Order, layers and rounds of the kept primitives are
unchanged. Shards already present in --out are skipped; output config.json records the source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from lpp.codec import Codec  # noqa: E402
from lpp.protocol import load_protocol  # noqa: E402
from lpp.render.reference import visible_mass  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-visible", type=float, default=2.0, help="pixels at the 256 px canvas")
    ap.add_argument("--protocol", type=Path, default=None)
    args = ap.parse_args()
    proto = load_protocol(args.protocol) if args.protocol else load_protocol()
    src_cfg = json.loads((args.src / "config.json").read_text())
    if src_cfg.get("protocol_version", 1) != proto.version:
        raise SystemExit(f"{args.src} is protocol v{src_cfg.get('protocol_version', 1)}, not v{proto.version}")
    codec = Codec.load(proto=proto, device="cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = dict(src_cfg, pruned_from=str(args.src), min_visible_px=args.min_visible)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=1))

    for f in sorted(args.src.glob("decomp_*.npz")):
        dst = args.out / f.name
        if dst.exists():
            continue
        t0 = time.perf_counter()
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        off = data["offsets"]
        keep_rows = []
        for i in range(len(off) - 1):
            tok = torch.from_numpy(data["tokens"][off[i]:off[i + 1]].astype(np.int64)).cuda()
            prims = codec.decode({n: tok[:, k].unsqueeze(0) for k, n in enumerate(proto.attr_names)})
            with torch.no_grad():
                vis = visible_mass(prims, proto.canvas_size, proto)[0]
            keep_rows.append(off[i] + torch.nonzero(vis >= args.min_visible).flatten().cpu().numpy())
        rows = np.concatenate(keep_rows)
        new_off = np.concatenate([[0], np.cumsum([len(r) for r in keep_rows])]).astype(np.int64)
        out = dict(data, tokens=data["tokens"][rows], layer=data["layer"][rows], round=data["round"][rows],
                   offsets=new_off)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        os.replace(tmp, dst)
        print(f"{f.name}: kept {len(rows)}/{off[-1]} primitives ({1 - len(rows) / off[-1]:.1%} removed) "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
