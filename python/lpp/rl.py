"""GRPO post-training pieces: rollouts -> batches, rendered rewards, group advantages, loss.

Rollouts are sampled at temperature 1 with no truncation and no guidance, so the model's
record_logprobs() (with the grammar mask replayed here) are the exact sampling
log-probabilities and the policy gradient is unbiased. Classifier-free guidance is still
used at inference; RL improves the conditional distribution it is built from.

Reward per sample = 100 * cos(CLIP image, CLIP prompt)
                    - orphan_weight * 100 * orphan rate
                    - early_penalty if the skin layer (2) was never painted
                    + attr_weight * attribute agreement (optional, see rewards()).
Advantages are normalised inside each prompt's group (GRPO), so the reward scale only
matters relative to the KL weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .clip_score import ClipScorer
from .codec import Codec
from .data.sequences import collate
from .metrics import orphan_stats
from .protocol import Protocol
from .render.dispatch import render_group
from .render.reference import flatten_on_base
from .sequence import DrawSeq, Grammar, paint_order


@dataclass
class RewardConfig:
    orphan_weight: float = 0.2      # 100 * orphan rate * this: 5% orphans -> -1 point
    early_penalty: float = 5.0      # no skin layer: not a face
    attr_weight: float = 0.0        # points for full agreement with the prompt image's attributes
    image_size: int = 256


def grammar_masks(seqs: list[DrawSeq], proto: Protocol, require_before_eos=(), draft: bool = False) -> list[torch.Tensor]:
    """Per sequence, the allowed-event mask before each target record: [len - 1, n_events].

    Replays the sampler's grammar (no REDRAW/REFINE from BOS, same EOS requirements)."""
    out = []
    for s in seqs:
        g = Grammar(proto, require_before_eos=frozenset(require_before_eos), allow_special=False, draft=draft)
        g.bos()
        masks = []
        for ev in s.event[1:].tolist():
            m = g.allowed_mask()
            if not m[ev]:            # the sampler forces EOS when it runs out of room
                m = torch.zeros_like(m)
                m[ev] = True
            masks.append(m)
            if ev in g.allowed_events():
                g.step(ev)
        out.append(torch.stack(masks))
    return out


def rollout_batch(seqs: list[DrawSeq], captions: list[str], proto: Protocol, require_before_eos=(),
                  draft: bool = False) -> dict:
    """Training batch (data.collate layout) for generated sequences, plus event_allowed [B, T, E]."""
    batch = collate([(s, c) for s, c in zip(seqs, captions)])
    B, T = batch["kind"].shape
    allowed = torch.ones(B, T, len(proto.events), dtype=torch.bool)
    for b, m in enumerate(grammar_masks(seqs, proto, require_before_eos, draft)):
        allowed[b, : m.shape[0]] = m
    batch["event_allowed"] = allowed
    return batch


@torch.no_grad()
def render(seqs: list[DrawSeq], codec: Codec, size: int = 256) -> tuple[list[np.ndarray], list[float], list[bool]]:
    """RGB uint8 images, orphan rates and 'has a face' flags of generated sequences."""
    proto, dev = codec.proto, codec.color_codes.device
    images, orphans, faces = [], [], []
    for s in seqs:
        attrs, lay = paint_order(s)
        faces.append(2 in lay.tolist())
        if attrs.shape[0] == 0:
            images.append(np.full((size, size, 3), 255, np.uint8))
            orphans.append(1.0)
            continue
        prims = codec.decode({k: attrs[:, i].unsqueeze(0).to(dev) for i, k in enumerate(proto.attr_names)})
        orphans.append(float(orphan_stats(prims, lay.to(dev), 128, proto=proto)["orphan"]))
        img = flatten_on_base(render_group(prims, size, proto), proto)[0]
        images.append((img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return images, orphans, faces


@torch.no_grad()
def rewards(images, orphans, faces, prompts: list[str], clip: ClipScorer, cfg: RewardConfig,
            attr_agreement: torch.Tensor | None = None) -> dict:
    """attr_agreement [B] in [0, 1]: optional, e.g. mean classifier probability of the prompt
    image's true attribute values (computed by the caller, which knows the source labels)."""
    img, txt = clip.images(images), clip.texts(prompts)
    clip_score = 100 * (img * txt).sum(1).cpu()
    orphan = torch.tensor(orphans)
    face = torch.tensor(faces, dtype=torch.float32)
    total = clip_score - cfg.orphan_weight * 100 * orphan - cfg.early_penalty * (1 - face)
    out = {"clip": clip_score, "orphan": orphan, "face": face}
    if attr_agreement is not None:
        agree = attr_agreement.float().cpu()
        total = total + cfg.attr_weight * agree
        out["attr"] = agree
    out["reward"] = total
    return out


def group_advantages(reward: torch.Tensor, group: int, eps: float = 1e-4) -> torch.Tensor:
    """(r - group mean) / (group std + eps); rows are consecutive groups of `group` samples."""
    r = reward.view(-1, group)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1)


def grpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor, adv: torch.Tensor,
              mask: torch.Tensor, clip_eps: float = 0.2, beta: float = 0.04) -> tuple[torch.Tensor, dict]:
    """Record-level clipped surrogate + k3 KL to the reference, averaged per sequence then over the batch.

    logp/logp_old/logp_ref [B, T] (record log-probs), adv [B], mask [B, T] records with a target.
    """
    m = mask.float()
    n = m.sum(1).clamp(min=1)
    ratio = torch.exp(logp - logp_old)
    a = adv[:, None]
    surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a)
    d = logp_ref - logp
    kl = torch.exp(d) - d - 1                                # k3 estimator, >= 0
    policy = -((surrogate * m).sum(1) / n).mean()
    kl_term = ((kl * m).sum(1) / n).mean()
    loss = policy + beta * kl_term
    stats = {"policy": float(policy.detach()), "kl": float(kl_term.detach()),
             "clip_frac": float((((ratio - 1).abs() > clip_eps).float() * m).sum() / m.sum().clamp(min=1))}
    return loss, stats


__all__ = ["RewardConfig", "grammar_masks", "rollout_batch", "render", "rewards", "group_advantages",
           "grpo_loss"]
