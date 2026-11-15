"""Write Chinese captions for face images with a local Qwen2.5-VL checkpoint (offline).

    python tools/caption_vlm.py --model /root/private_data/models/Qwen2.5-VL-7B-Instruct \
        --images /root/private_data/caption_images --out /root/private_data/captions.jsonl

Two captions per image as two plain lines (no JSON: fewer tokens to generate): a short
one (<= 15 characters) and a detailed one (<= 50). The
prompt asks for what the layers can actually express: hair, face and skin, features
and expression, makeup, accessories, clothing, background color and framing.
Resumable: ids already in the output file are skipped. Output lines:
    {"id": 123, "short": "...", "long": "...", "raw": "..."}

Throughput note (Hygon DCU, DTK 26.04): generation is overhead-bound, not bandwidth-bound,
so fewer output tokens and bigger batches are what help.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROMPT = (
    "用中文描述这张人像照片，只输出两行，不要编号、引号或其他内容：\n"
    "第一行：15字以内的简短描述。\n"
    "第二行：50字以内的详细描述，依次写发型发色、肤色、五官与表情、妆容、"
    "饰品（眼镜、帽子、耳环、项链，没有就不写）、服装、背景颜色。"
    "不要猜测身份、年龄数字或姓名。"
)


def load(model_dir: str, dtype=torch.bfloat16):
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen2VLImageProcessor  # slow processor: needs no torchvision
    if torch.version.hip:
        # DTK builds of PyTorch dispatch SDPA to a FlashAttention library they do not ship
        # ("No matching libraries found for flash_attn_2_cuda*.so") and are compiled without
        # memory-efficient attention: allow only the plain math kernel.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    imgp = Qwen2VLImageProcessor.from_pretrained(model_dir)
    return model, tok, imgp


def build_inputs(tok, imgp, images: list[Image.Image]):
    """Chat-format prompt with the image placeholder expanded to the right token count."""
    vis = imgp(images=images, return_tensors="pt")
    merge = imgp.merge_size ** 2
    texts = []
    for thw in vis["image_grid_thw"]:
        n_tok = int(thw.prod()) // merge
        texts.append(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            "<|vision_start|>" + "<|image_pad|>" * n_tok + "<|vision_end|>" + PROMPT +
            "<|im_end|>\n<|im_start|>assistant\n")
    enc = tok(texts, return_tensors="pt", padding=True)
    return {**enc, "pixel_values": vis["pixel_values"], "image_grid_thw": vis["image_grid_thw"]}


def parse(text: str) -> dict:
    """Two plain lines: short, then long. Tolerates stray labels like '第一行：'."""
    lines = [re.sub(r"^\s*(第[一二]行|简短描述|详细描述)?\s*[:：]?\s*", "", line).strip()
             for line in text.strip().splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0] and lines[1]:
        return {"short": lines[0], "long": lines[1]}
    if len(lines) == 1 and len(lines[0]) >= 8:
        # ~20% of outputs skip the short line and give only the detailed one:
        # keep it, and use its first two clauses as the short caption
        clauses = [c for c in re.split(r"[，,。；;]", lines[0]) if c]
        return {"short": "，".join(clauses[:2]), "long": lines[0]}
    return {"short": "", "long": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from lpp.runtime_env import configure_threads
    print(f"torch threads: {configure_threads()}", flush=True)

    ids = sorted(int(p.stem) for p in args.images.glob("*.jpg"))[: args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line.strip()}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} images, {len(done)} done, {len(todo)} to caption", flush=True)

    model, tok, imgp = load(args.model)
    t0, n, bad = time.perf_counter(), 0, 0
    with open(args.out, "a", encoding="utf-8") as f:
        for b0 in range(0, len(todo), args.batch):
            chunk = todo[b0 : b0 + args.batch]
            images = [Image.open(args.images / f"{i}.jpg").convert("RGB") for i in chunk]
            inputs = {k: v.to("cuda") for k, v in build_inputs(tok, imgp, images).items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     repetition_penalty=1.05)
            texts = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(chunk, texts):
                rec = {"id": i, **parse(t), "raw": t}
                bad += rec["short"] == ""
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += len(chunk)
            rate = n / (time.perf_counter() - t0)
            print(f"{len(done) + n}/{len(ids)}  {rate:.1f} img/s  unparsed {bad}  eta "
                  f"{(len(todo) - n) / rate / 60:.0f} min | {texts[0][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
