"""CelebA attribute classifier: a CLIP-independent check of whether generated faces show the
attributes of the image their prompt describes.

    python tools/attr_classifier.py --data data/celebamask256 --attributes data/celebamask256/attributes.json \
        --split-manifest data/celebamask256_split_v1.jsonl --out out/attr_classifier

ImageNet ResNet-18, fine-tuned on the training split (all 40 attributes, flip augmentation).
Reports balanced accuracy per attribute on the validation split; eval_text.py uses the checkpoint
(--attr-classifier) to compare generated images with the attributes of their prompt's source image.
EVAL_ATTRS are clearly visible in a 256px portrait and usually named in captions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EVAL_ATTRS = ["Male", "Smiling", "Eyeglasses", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Bald",
              "Wearing_Hat", "Wearing_Lipstick", "Wearing_Earrings", "Bangs", "No_Beard"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _Block(torch.nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        nn = torch.nn
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                           if stride != 1 or cin != cout else None)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet18(torch.nn.Module):
    """torchvision's resnet18 layout with the same parameter names, in plain PyTorch:
    the DCU's DTK PyTorch ships without torchvision, inference must not need it."""

    def __init__(self, n: int):
        super().__init__()
        nn = torch.nn
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        chans, cin = (64, 128, 256, 512), 64
        for i, c in enumerate(chans):
            setattr(self, f"layer{i + 1}", nn.Sequential(_Block(cin, c, 1 if i == 0 else 2), _Block(c, c, 1)))
            cin = c
        self.fc = nn.Linear(512, n)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(x.mean((2, 3)))


def build_model(n: int, pretrained: bool = True) -> torch.nn.Module:
    m = ResNet18(n)
    if pretrained:                              # training only: ImageNet weights via torchvision
        import torchvision
        tv = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        state = {k: v for k, v in tv.state_dict().items() if not k.startswith("fc.")}
        missing, unexpected = m.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) == {"fc.weight", "fc.bias"}, (missing, unexpected)
    return m


def prep(imgs_uint8: torch.Tensor) -> torch.Tensor:
    """[B, H, W, 3] uint8 -> normalized [B, 3, 224, 224]."""
    x = imgs_uint8.permute(0, 3, 1, 2).float().div(255)
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class AttrClassifier:
    def __init__(self, ckpt: Path, device: str):
        s = torch.load(ckpt, map_location=device, weights_only=False)
        self.names, self.thresholds = s["names"], torch.tensor(s["thresholds"], device=device)
        self.model = build_model(len(self.names), pretrained=False).to(device).eval()
        self.model.load_state_dict(s["model"])
        self.device = device

    @torch.no_grad()
    def predict(self, imgs: np.ndarray, batch: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), batch):
            x = prep(torch.from_numpy(np.ascontiguousarray(imgs[i:i + batch])).to(self.device))
            out.append((torch.sigmoid(self.model(x)) > self.thresholds).cpu())
        return torch.cat(out).numpy()


def balanced_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    pos, neg = true, ~true
    tpr = (pred & pos).sum() / max(pos.sum(), 1)
    tnr = (~pred & neg).sum() / max(neg.sum(), 1)
    return float((tpr + tnr) / 2) if pos.any() and neg.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--attributes", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    a = json.loads(args.attributes.read_text())
    names, values = a["names"], {int(k): v for k, v in a["values"].items()}
    split = {int(r["id"]): r["split"] for r in map(json.loads, args.split_manifest.read_text().splitlines())}
    index = json.loads((args.data / "index.json").read_text())
    shards, ids, where = [], [], []
    for s in index["shards"]:
        shards.append(np.load(args.data / f"images_{s['shard']:03d}.npy", mmap_mode="r"))
        ids += s["ids"]
        where += [(len(shards) - 1, r) for r in range(len(s["ids"]))]

    def gather(rows: np.ndarray) -> np.ndarray:          # memory-mapped: 30 shards would need ~6 GB RAM
        return np.stack([shards[where[r][0]][where[r][1]] for r in rows])
    y = np.array([values[i] for i in ids], dtype=bool)
    tr = np.array([split[i] == "train" for i in ids])
    va = ~tr
    model = build_model(len(names)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_frac = torch.tensor(y[tr].mean(0), dtype=torch.float32, device=dev).clamp(0.01, 0.99)
    pos_weight = (1 - pos_frac) / pos_frac            # balances rare attributes
    tr_idx = np.flatnonzero(tr)
    steps = args.epochs * (len(tr_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    rng = np.random.default_rng(0)
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        for b0 in range(0, len(perm) - args.batch + 1, args.batch):
            sel = np.sort(perm[b0:b0 + args.batch])
            x = torch.from_numpy(gather(sel)).to(dev)
            if rng.random() < 0.5:
                x = x.flip(2)                           # horizontal flip ([B, H, W, 3])
            t = torch.from_numpy(y[sel]).float().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(prep(x))
            loss = F.binary_cross_entropy_with_logits(logits.float(), t, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"epoch {ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        probs = torch.cat([torch.sigmoid(model(prep(torch.from_numpy(gather(np.flatnonzero(va)[i:i + 256])).to(dev))))
                           for i in range(0, int(va.sum()), 256)]).cpu().numpy()
    yv = y[va]
    # per-attribute threshold maximising balanced accuracy on validation (reported as such)
    thresholds, report = [], {}
    for k, n in enumerate(names):
        cands = np.linspace(0.05, 0.95, 19)
        accs = [balanced_accuracy(probs[:, k] > c, yv[:, k]) for c in cands]
        best = int(np.nanargmax(accs)) if not np.all(np.isnan(accs)) else 9
        thresholds.append(float(cands[best]))
        report[n] = {"balanced_acc": round(accs[best], 4), "threshold": thresholds[-1], "val_pos": int(yv[:, k].sum())}
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "names": names, "thresholds": thresholds, "eval_attrs": EVAL_ATTRS},
               args.out / "attr_resnet18.pt")
    summary = {"eval_attrs_mean_balanced_acc": round(float(np.mean([report[n]["balanced_acc"] for n in EVAL_ATTRS])), 4),
               "per_attribute": report,
               "note": "thresholds tuned on the validation split; real-image accuracy is an upper reference"}
    (args.out / "report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({n: report[n] for n in EVAL_ATTRS}, indent=1))
    print("mean balanced acc (eval attrs):", summary["eval_attrs_mean_balanced_acc"])


if __name__ == "__main__":
    main()
