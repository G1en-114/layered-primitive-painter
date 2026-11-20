"""Reference decode loop on the exported ONNX graphs (the C++ runtime mirrors this file).

    python tools/ort_generate.py --onnx out/onnx/small_t0 --ckpt out/small_v2_t0_text_canvas/model_step8000.pt \
        --text-encoder data/models/chinese-clip-vit-base-patch16 --n 6 --compare

Same algorithm as lpp.sample.generate: grammar-masked event, then the attributes one by one,
classifier-free guidance by running conditional and unconditional rows together, canvas
checkpoints re-encoded through canvas_enc. The network runs only through onnxruntime; sampling,
the grammar and canvas rendering reuse the library (they become C++ / CUDA in the runtime).
With --compare, both samplers get the same uniforms and the token sequences are compared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from lpp.canvas import add_primitives, compose, empty_state, is_checkpoint  # noqa: E402
from lpp.codec import Codec  # noqa: E402
from lpp.protocol import _parse  # noqa: E402
from lpp.sample import SampleConfig, _sample  # noqa: E402
from lpp.sequence import KIND_BOS, KIND_CONTROL, KIND_PRIMITIVE, NO_ATTR, DrawSeq, Grammar  # noqa: E402


class OrtPainter:
    def __init__(self, onnx_dir: Path, providers: list[str]):
        import onnxruntime as ort
        self.manifest = json.loads((onnx_dir / "manifest.json").read_text())
        self.cfg = self.manifest["config"]
        self.proto = _parse(self.manifest["protocol"])
        self.attr_names = self.manifest["attr_names"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # the KV cache grows by one position per call, so every call has new shapes; the CPU arena keeps a block for
        # each and grew past 5 GB for two base-size variants (plan v4 deployment run). Plain malloc/free instead.
        so.enable_cpu_mem_arena = False
        self.s = {name: ort.InferenceSession(str(onnx_dir / g["file"]), so, providers=providers)
                  for name, g in self.manifest["graphs"].items()}

    def run(self, name: str, **feeds):
        return self.s[name].run(None, feeds)


def ort_generate(m: OrtPainter, codec: Codec, batch: int, cfg: SampleConfig, text=None, text_valid=None,
                 uniforms=None) -> list[DrawSeq]:
    proto, c = m.proto, m.cfg
    B, A = batch, len(m.attr_names)
    L, kv, hd = c["n_layers"], c["n_kv_heads"], c["d_model"] // c["n_heads"]
    Lt = 0 if text is None else text.shape[1]
    guided = text is not None and cfg.cfg_scale != 1.0
    if guided:
        text = np.concatenate([text, np.zeros_like(text)])
        text_valid = np.concatenate([text_valid, np.zeros_like(text_valid)])
    R = 2 * B if guided else B
    max_records = cfg.max_records or c["max_positions"] - Lt
    Tmax = Lt + max_records
    K = np.zeros((L, R, kv, Tmax, hd), np.float32)
    V = np.zeros_like(K)
    kmask = np.zeros((R, Tmax), bool)
    length, next_pos = 0, np.zeros(R, np.int64)

    def rows(a):
        return np.concatenate([a, a]) if guided else a

    def guide(logits):
        t = torch.from_numpy(logits).float()
        if not guided:
            return t
        cnd, unc = t.chunk(2)
        return unc + cfg.cfg_scale * (cnd - unc)

    def chunk(x, valid, canvas_tokens, draw):
        nonlocal length, next_pos
        T = x.shape[1]
        P = max(length, 1)                              # a masked dummy slot when the cache is empty
        h, kn, vn = m.run("chunk", x=x, valid=valid, past_k=K[:, :, :, :P], past_v=V[:, :, :, :P],
                          past_mask=kmask[:, :P] if length else np.zeros((R, 1), bool), start_pos=next_pos,
                          canvas_tokens=canvas_tokens, draw=draw)
        K[:, :, :, length:length + T], V[:, :, :, length:length + T] = kn, vn
        kmask[:, length:length + T] = valid
        length += T
        next_pos = next_pos + valid.sum(1)
        return h

    rand = uniforms or (lambda step, field: torch.rand(B))
    prim_ev, eos = proto.event_id("PRIMITIVE"), proto.event_id("EOS")
    strokes_ev = set(proto.stroke_events)                                     # PRIMITIVE (+ LINE in v3)
    line_ev = proto.event_id("LINE") if proto.has_lines else None
    fixed = {e: proto.fixed_tokens(e) for e in proto.stroke_events}
    grammars = [Grammar(proto, require_before_eos=frozenset(cfg.require_before_eos), allow_special=False,
                        draft=m.cfg.get("order", "layer") == "draft")
                for _ in range(B)]
    for g in grammars:
        g.bos()

    # canvas state (per-layer premultiplied), as lpp.canvas.CanvasTracker
    size, k_ck = c["canvas_size"], c["canvas_k"]
    dev = codec.color_codes.device
    state = empty_state(B, proto.num_layers, size, dev)
    base = torch.tensor(proto.base_color, device=dev)
    tokens = m.run("canvas_enc", canvas=compose(state, base).cpu().numpy())[0]
    pending, since = [[] for _ in range(B)], [0] * B

    rec = (np.full((B, 1), KIND_BOS), np.full((B, 1), -1), np.full((B, 1), -1), np.full((B, 1, A), NO_ATTR))
    x = m.run("embed_record", kind=rows(rec[0]), event=rows(rec[1]), layer=rows(rec[2]), attrs=rows(rec[3]))[0]
    valid = np.ones((R, 1), bool)
    if Lt:
        x = np.concatenate([m.run("embed_text", text=text)[0], x], 1)
        valid = np.concatenate([text_valid, valid], 1)
    draw = np.zeros(valid.shape, bool)
    draw[:, -1] = True
    h = chunk(x, valid, rows(tokens), draw)[:, -1]

    out = {"kind": [rec[0][:, 0]], "event": [rec[1][:, 0]], "layer": [rec[2][:, 0]], "attrs": [rec[3][:, 0]]}
    done = [False] * B
    zeros_e, zeros_a = np.zeros(R, np.int64), np.zeros((R, A), np.int64)
    for step in range(1, max_records):
        logits = guide(m.run("decode_heads", h=h, event=zeros_e, attrs=zeros_a)[0])
        allowed = torch.zeros(B, len(proto.events), dtype=torch.bool)
        if step == max_records - 1:
            allowed[:, eos] = True
        else:
            for b, g in enumerate(grammars):
                allowed[b, [eos] if g.done else g.allowed_events()] = True
        event = _sample(logits.masked_fill(~allowed, float("-inf")), cfg, rand(step, "event"), "event")
        ev = event.tolist()
        attrs = np.full((B, A), NO_ATTR, np.int64)
        if any(e in strokes_ev for e in ev):
            for i, name in enumerate(m.attr_names):
                outs = m.run("decode_heads", h=h, event=rows(event.numpy()), attrs=rows(np.maximum(attrs, 0)))
                attrs[:, i] = _sample(guide(outs[1 + i]), cfg, rand(step, name), name).numpy()
            for e, fx in fixed.items():                                       # as lpp.sample.generate
                for nm, tk in fx.items():
                    attrs[np.array(ev) == e, list(m.attr_names).index(nm)] = tk
            attrs[~np.isin(np.array(ev), list(strokes_ev))] = NO_ATTR
        layer = []
        for b, g in enumerate(grammars):
            if g.done:
                layer.append(-1)
                continue
            if step == max_records - 1 and ev[b] == eos:   # forced end, as lpp.sample.generate
                g.done = True
            else:
                g.step(ev[b])
            layer.append(g.current if not g.done else -1)
        kind = [KIND_PRIMITIVE if e in strokes_ev else KIND_CONTROL for e in ev]
        live = [not d for d in done]
        for key, v in (("kind", kind), ("event", ev), ("layer", layer), ("attrs", attrs)):
            out[key].append(np.asarray(v))
        done = [d or e == eos for d, e in zip(done, ev)]
        if all(done):
            break
        # canvas checkpoints (same rule and rendering as CanvasTracker.update)
        hit = []
        for b in range(B):
            if not live[b]:
                continue
            if kind[b] == KIND_PRIMITIVE:
                pending[b].append((attrs[b].copy(), layer[b], ev[b] == line_ev))
                since[b] += 1
            if is_checkpoint(kind[b], since[b], k_ck):
                hit.append(b)
                since[b] = 0
        if hit:
            n = max(len(pending[b]) for b in hit)
            if n:
                tok = torch.zeros(len(hit), n, A, dtype=torch.long)
                lay = torch.full((len(hit), n), -1, dtype=torch.long)
                isl = torch.zeros((len(hit), n), dtype=torch.bool)
                for j, b in enumerate(hit):
                    if pending[b]:
                        tok[j, :len(pending[b])] = torch.from_numpy(np.stack([a for a, _, _ in pending[b]])).clamp(min=0)
                        lay[j, :len(pending[b])] = torch.tensor([la for _, la, _ in pending[b]])
                        isl[j, :len(pending[b])] = torch.tensor([ln for _, _, ln in pending[b]])
                tok, lay = tok.to(dev), lay.to(dev)
                prims = codec.decode_canvas({nm: tok[..., i] for i, nm in enumerate(proto.attr_names)}, isl.to(dev))
                prims = prims.with_alpha(prims.alpha * (lay >= 0))
                idx = torch.tensor(hit, device=dev)
                P_, T_ = add_primitives((state[0][idx], state[1][idx]), prims, lay, size, proto)
                state[0][idx], state[1][idx] = P_, T_
            idx = torch.tensor(hit, device=dev)
            canvas = compose((state[0][idx], state[1][idx]), base)
            tokens[hit] = m.run("canvas_enc", canvas=canvas.cpu().numpy())[0]
            for b in hit:
                pending[b] = []
        x = m.run("embed_record", kind=rows(np.array(kind)[:, None]), event=rows(np.array(ev)[:, None]),
                  layer=rows(np.array(layer)[:, None]), attrs=rows(attrs[:, None]))[0]
        h = chunk(x, rows(np.array(live)[:, None]), rows(tokens), np.ones((R, 1), bool))[:, -1]

    ev_all = np.stack(out["event"], 1)
    seqs = []
    for b in range(B):
        hits = np.flatnonzero(ev_all[b] == eos)
        end = int(hits[0]) + 1 if len(hits) else ev_all.shape[1]
        seqs.append(DrawSeq(torch.from_numpy(np.stack(out["kind"], 1)[b, :end]).long(),
                            torch.from_numpy(ev_all[b, :end]).long(),
                            torch.from_numpy(np.stack(out["attrs"], 1)[b, :end]).long(),
                            torch.from_numpy(np.stack(out["layer"], 1)[b, :end]).long()))
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="the exported checkpoint (for --compare)")
    ap.add_argument("--text-encoder", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-records", type=int, default=300)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write text.npy, text_valid.npy, uniforms.npy (full table) and the ORT records for the C++ runtime")
    args = ap.parse_args()
    from lpp.text import TextEncoder
    m = OrtPainter(args.onnx, [args.provider])
    codec = Codec.load(proto=m.proto, device="cpu")
    prompts = ["金发碧眼的女性，微笑，浅色背景", "黑色短发的男性，戴眼镜，深色背景", "棕色长卷发的女性，涂红色口红，戴耳环",
               "灰白头发的男性，留着胡子，表情严肃", "黑色长直发的女性，有刘海，白色上衣，绿色背景", "光头男性，微笑，穿深色西装"]
    # cycled to --n: [: n] alone capped every batch at 6, so "B = 8 / 32" benchmark inputs silently had 6 rows
    prompts = [prompts[i % len(prompts)] for i in range(args.n)]
    enc = TextEncoder(args.text_encoder, max_positions=m.proto.max_text_positions)
    text, tv = enc(prompts)
    sc = SampleConfig(temperature=1.0, top_p=0.9, cfg_scale=args.cfg_scale, max_records=args.max_records,
                      require_before_eos=(0, 2))
    # one full table [max_records, 1 + A, B] (field 0 = event, 1 + i = attribute i), shared with C++
    g = torch.Generator().manual_seed(0)
    fields = ["event"] + list(m.attr_names)
    table = torch.rand(args.max_records, len(fields), len(prompts), generator=g)

    def uniforms(step, field):
        return table[step, fields.index(field)]

    t0 = time.perf_counter()
    seqs = ort_generate(m, codec, len(prompts), sc, text.numpy(), tv.numpy(), uniforms)
    dt = time.perf_counter() - t0
    n_rec = sum(len(s) for s in seqs)
    print(f"ORT ({args.provider}): {len(seqs)} sequences, {n_rec} records in {dt:.1f}s ({n_rec / dt:.0f} records/s)")
    if args.dump is not None:
        args.dump.mkdir(parents=True, exist_ok=True)
        np.save(args.dump / "text.npy", text.numpy().astype(np.float32))
        np.save(args.dump / "text_valid.npy", tv.numpy().astype(bool))
        np.save(args.dump / "uniforms.npy", table.numpy().astype(np.float32))
        A = len(m.attr_names)
        rec = np.full((len(seqs), max(len(s) for s in seqs), 3 + A), -2, np.int64)
        for b, s in enumerate(seqs):
            rec[b, :len(s), 0], rec[b, :len(s), 1], rec[b, :len(s), 2] = s.kind, s.event, s.layer
            rec[b, :len(s), 3:] = s.attrs
        np.save(args.dump / "ort_python_records.npy", rec)
        print(f"wrote {args.dump}")
    if args.compare:
        from lpp.model import ModelConfig, Painter
        from lpp.sample import generate
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = Painter(ModelConfig(**state["config"]), m.proto, codec).eval()
        model.load_state_dict(state["model"])
        ref = generate(model, batch=len(prompts), cfg=sc, text=text, text_valid=tv, uniforms=uniforms, codec=codec)
        agree, total, first_diff = 0, 0, []
        for a, b in zip(seqs, ref):
            n = min(len(a), len(b))
            same = (a.event[:n] == b.event[:n]) & (a.attrs[:n] == b.attrs[:n]).all(1)
            agree += int(same.sum())
            total += max(len(a), len(b))
            bad = torch.nonzero(~same)
            first_diff.append(int(bad[0]) if len(bad) else (n if len(a) != len(b) else None))
        print(f"token agreement with PyTorch: {agree}/{total} records; first divergence per sample: {first_diff}")


if __name__ == "__main__":
    main()
