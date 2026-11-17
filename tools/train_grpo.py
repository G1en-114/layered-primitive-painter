"""GRPO post-training with rendered rewards (docs: python/lpp/rl.py).

    python tools/train_grpo.py --init runs/s2/last.pt --out runs/grpo_s2 \
        --text-encoder models/chinese-clip-vit-base-patch16 --captions captions_clean_v1.jsonl \
        --split-manifest celebamask256_split_v1.jsonl --steps 300

Each step: P training-split prompts x G samples (temperature 1, no truncation, no guidance),
rendered and scored by the frozen Chinese-CLIP (image-text similarity) minus an orphan
penalty and an early-stop penalty; advantages are normalised within each prompt's group;
the policy takes the clipped surrogate step with a k3 KL penalty to the frozen initial model.
Samples with guidance for the fixed prompts of train.py are rendered every --sample-every.

v2 options (realism, 2026-11-17): --source-emb adds a reward for looking like the prompt's own
source image (CLIP image-image cosine; an anime prompt is pulled towards its anime source, a photo
prompt towards its photo, so "realistic" never means "photographic"); --anime-frac draws that share
of prompts from anime captions when the pool holds both domains; the attribute reward only exists
for CelebA images, so other groups get no attribute advantage.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from lpp.clip_score import ClipScorer  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.data.splits import load_split_manifest  # noqa: E402
from lpp.model import ModelConfig, Painter  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.rl import RewardConfig, group_advantages, grpo_loss, render, rewards, rollout_batch  # noqa: E402
from lpp.sample import SampleConfig, generate  # noqa: E402
from lpp.text import TextEncoder, with_domain  # noqa: E402
from train import render_samples, to_gpu  # noqa: E402

REQUIRE = (0, 2)       # background and skin before EOS, in rollouts and in the replayed grammar


def prompt_pool(captions: Path, split_manifest: Path) -> list[tuple[str, int]]:
    """(caption, source image id) for every long/short caption of a training-split image.

    Captions with a "domain" field (v3: 真人照片 / 动漫) get the same prefix as in supervised training."""
    splits = load_split_manifest(split_manifest)
    pool = []
    for line in captions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if splits.get(int(r["id"])) == "train":
                pool += [(with_domain(c, r.get("domain")), int(r["id"])) for c in (r.get("long"), r.get("short")) if c]
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True, help="supervised checkpoint to start from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-encoder", type=Path, required=True)
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--prompts", type=int, default=4, help="prompts per step")
    ap.add_argument("--group", type=int, default=8, help="samples per prompt")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL weight to the initial model")
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=1, help="updates per rollout batch (1 = on-policy)")
    ap.add_argument("--micro-batch", type=int, default=16, help="sequences per forward/backward chunk")
    ap.add_argument("--orphan-weight", type=float, default=0.2)
    ap.add_argument("--early-penalty", type=float, default=5.0)
    ap.add_argument("--attr-classifier", type=Path, default=None,
                    help="tools/attr_classifier.py checkpoint: reward agreement with the prompt image's attributes")
    ap.add_argument("--attributes", type=Path, default=None, help="CelebA attributes.json (source labels)")
    ap.add_argument("--attr-weight", type=float, default=5.0, help="points for full agreement on REWARD_ATTRS")
    ap.add_argument("--adv-mode", choices=["sum", "per-component"], default="sum",
                    help="per-component: normalise CLIP, attribute and orphan terms within each group separately and "
                         "add them with --w-* weights, so no term drowns the others by raw scale")
    ap.add_argument("--w-clip", type=float, default=1.0)
    ap.add_argument("--w-attr", type=float, default=1.0)
    ap.add_argument("--w-orphan", type=float, default=0.5)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=2.0, help="guidance for the periodic sample grid only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--source-emb", type=Path, default=None,
                    help="npz (ids, emb) from tools/source_embeddings.py: reward likeness to the prompt's source image")
    ap.add_argument("--w-src", type=float, default=1.0, help="per-component weight of the source-likeness advantage")
    ap.add_argument("--anime-frac", type=float, default=None,
                    help="share of prompts drawn from anime captions (default: as they come in the pool)")
    ap.add_argument("--keep-snapshots", action="store_true",
                    help="also write weights-only model_stepNNN.pt at every save (for evaluation off the run)")
    ap.add_argument("--sdpa-backend", choices=["auto", "math"], default="auto")
    args = ap.parse_args()
    if args.group < 2:
        ap.error("--group needs at least 2 samples to form an advantage")
    if args.sdpa_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    src = torch.load(args.init, map_location=dev, weights_only=False)
    proto = _parse(src["protocol"])
    codec = Codec.load(proto=proto, device=dev)
    cfg = ModelConfig(**src["config"])
    policy = Painter(cfg, proto, codec).to(dev)
    policy.load_state_dict(src["model"])
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step = 0
    ckpt = args.out / "last.pt"
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        policy.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        rng.setstate(state["rng"])
        print(f"resumed at step {step}")
    encoder = TextEncoder(args.text_encoder, max_positions=proto.max_text_positions).to(dev)
    clip = ClipScorer(args.text_encoder, dev)
    pool = prompt_pool(args.captions, args.split_manifest)
    rcfg = RewardConfig(orphan_weight=args.orphan_weight, early_penalty=args.early_penalty,
                        attr_weight=args.attr_weight if args.attr_classifier else 0.0)
    attr_fn = None
    if args.attr_classifier is not None:
        from attr_classifier import REWARD_ATTRS, AttrClassifier
        if args.attributes is None:
            ap.error("--attr-classifier needs --attributes")
        a = json.loads(args.attributes.read_text())
        cols = [a["names"].index(n) for n in REWARD_ATTRS]
        labels = {int(k): torch.tensor([v[c] for c in cols], dtype=torch.float32) for k, v in a["values"].items()}
        clf = AttrClassifier(args.attr_classifier, dev)

        def attr_fn(images, image_ids):
            """mean probability the classifier gives the source image's true value, over REWARD_ATTRS;
            NaN for images without CelebA labels (anime)"""
            out = torch.full((len(image_ids),), float("nan"))
            have = [k for k, i in enumerate(image_ids) if i in labels]
            if have:
                p = clf.probs(np.stack([images[k] for k in have]))[:, cols].cpu()
                t = torch.stack([labels[image_ids[k]] for k in have])
                out[have] = (t * p + (1 - t) * (1 - p)).mean(1)
            return out
    src_emb = None
    if args.source_emb is not None:
        z = np.load(args.source_emb)
        src_row = {int(i): r for r, i in enumerate(z["ids"])}
        src_emb = torch.from_numpy(z["emb"].astype(np.float32))
        missing = sum(i not in src_row for _, i in pool)
        if missing:
            raise SystemExit(f"--source-emb lacks {missing} of the {len(pool)} prompt images")
    anime_pool = [p for p in pool if p[0].startswith("动漫")]
    face_pool = [p for p in pool if not p[0].startswith("动漫")]
    if args.anime_frac is not None and not (anime_pool and face_pool):
        ap.error("--anime-frac needs both anime and face prompts in the pool")
    sc = SampleConfig(temperature=1.0, top_p=1.0, cfg_scale=1.0, require_before_eos=REQUIRE,
                      max_records=args.max_records)
    (args.out / "config.json").write_text(json.dumps(vars(args) | {"init_step": src.get("step"),
                                                                   "prompt_pool": len(pool)}, indent=1, default=str))
    print(f"{policy.num_params() / 1e6:.1f}M policy from {args.init} (step {src.get('step')}); "
          f"{len(pool)} training prompts", flush=True)
    log = (args.out / "log.jsonl").open("a")
    gen = torch.Generator(device=dev).manual_seed(args.seed + 7919 * step)

    while step < args.steps:
        t0 = time.perf_counter()
        if args.anime_frac is None:
            picks = [rng.choice(pool) for _ in range(args.prompts)]
        else:
            picks = [rng.choice(anime_pool if rng.random() < args.anime_frac else face_pool) for _ in range(args.prompts)]
        rep = [c for c, _ in picks for _ in range(args.group)]
        rep_ids = [i for _, i in picks for _ in range(args.group)]
        with torch.no_grad():
            text, text_valid = encoder(rep)
            seqs = generate(policy, batch=len(rep), cfg=sc, text=text, text_valid=text_valid,
                            generator=gen, codec=codec)
        t_gen = time.perf_counter()
        images, orphans, faces = render(seqs, codec)
        r = rewards(images, orphans, faces, rep, clip, rcfg,
                    attr_agreement=attr_fn(images, rep_ids) if attr_fn is not None else None,
                    source=src_emb[[src_row[i] for i in rep_ids]] if src_emb is not None else None)
        if args.adv_mode == "sum":
            adv = group_advantages(r["reward"], args.group)
        else:
            clip_term = r["clip"] - args.early_penalty * (1 - r["face"])
            adv = args.w_clip * group_advantages(clip_term, args.group)
            adv = adv + args.w_orphan * group_advantages(-r["orphan"], args.group)
            if "attr" in r:
                has = ~torch.isnan(r["attr"])        # groups are one prompt, so one domain each
                a_adv = group_advantages(torch.nan_to_num(r["attr"]), args.group)
                adv = adv + args.w_attr * torch.where(has, a_adv, torch.zeros_like(a_adv))
            if "src" in r:
                adv = adv + args.w_src * group_advantages(r["src"], args.group)
        adv = adv.to(dev)
        # micro-batches with gradient accumulation: the loss is a mean over sequences, so
        # chunk losses weighted by chunk/total sum to the full-batch loss (same gradient)
        chunks = []
        for c0 in range(0, len(seqs), args.micro_batch):
            sl = slice(c0, c0 + args.micro_batch)
            b = to_gpu(rollout_batch(seqs[sl], rep[sl], proto, REQUIRE, draft=cfg.order == "draft"), encoder, codec, cfg)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                old = policy.record_logprobs(b, b["event_allowed"])
                refl = ref.record_logprobs(b, b["event_allowed"])
            chunks.append((b, old, refl, adv[sl]))
        policy.train()
        for _ in range(args.ppo_epochs):
            opt.zero_grad(set_to_none=True)
            stats = {}
            for b, old, refl, a in chunks:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logp = policy.record_logprobs(b, b["event_allowed"])
                loss, st = grpo_loss(logp, old, refl, a, b["tgt_event"] >= 0, args.clip_eps, args.beta)
                w = a.shape[0] / len(seqs)
                (loss * w).backward()
                for k, v in st.items():
                    stats[k] = stats.get(k, 0.0) + v * w
                del logp, loss
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
            opt.step()
        del chunks
        step += 1
        rec = {"step": step, "reward": round(float(r["reward"].mean()), 4),
               "reward_std_in_group": round(float(r["reward"].view(-1, args.group).std(1).mean()), 4),
               "clip": round(float(r["clip"].mean()), 4), "orphan": round(float(r["orphan"].mean()), 4),
               **({"attr": round(float(r["attr"].nanmean()), 4)} if "attr" in r else {}),
               **({"src": round(float(r["src"].mean()), 4)} if "src" in r else {}),
               "anime_share": round(sum(c.startswith("动漫") for c in rep) / len(rep), 3),
               "face": round(float(r["face"].mean()), 4),
               "length": round(sum(len(s) for s in seqs) / len(seqs), 1), "gnorm": round(gnorm, 4),
               **{k: round(v, 5) for k, v in stats.items()},
               "gen_s": round(t_gen - t0, 2), "update_s": round(time.perf_counter() - t_gen, 2)}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % args.sample_every == 0 or step == args.steps:
            render_samples(policy, codec, 6, args.out / f"samples_{step:05d}.png", seed=step, encoder=encoder,
                           cfg_scale=args.cfg_scale)
        if step % args.save_every == 0 or step == args.steps:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": policy.state_dict(), "opt": opt.state_dict(), "step": step,
                        "config": cfg.to_dict(), "protocol": proto.raw, "rng": rng.getstate(),
                        "args": vars(args) | {"out": str(args.out), "init": str(args.init)}}, tmp)
            tmp.replace(ckpt)
            if args.keep_snapshots:
                snap = args.out / f"model_step{step:03d}.pt"
                torch.save({"model": policy.state_dict(), "step": step, "config": cfg.to_dict(),
                            "protocol": proto.raw}, snap.with_suffix(".tmp"))
                snap.with_suffix(".tmp").replace(snap)


if __name__ == "__main__":
    main()
