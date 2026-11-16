"""Captions for the mixed faces + anime base model (plan v3): one JSONL with a `domain` per image.

    python tools/build_captions_v3.py --faces data/captions/captions_clean_v1.jsonl \
        --anime-manifest data/anime_portraits256_v2/manifest.jsonl --out data/captions/captions_v3.jsonl

Faces keep their VLM captions (domain 真人照片). Anime captions are rebuilt from the upstream Danbooru
tags: the first version used hair and eye colour only. About 100 frequent tags are mapped to Chinese
phrases by category (subject, hair, eyes, expression, accessories, clothing, framing, background);
body-focused and sexual tags are deliberately not mapped. lpp.text.CaptionStore prepends the domain
word to every caption variant, so prompts can ask for either style.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SUBJECT = {"1girl": "女孩", "1boy": "男孩", "2girls": "两个女孩", "multiple_girls": "几个女孩"}
HAIR_LEN = {"very_long_hair": "很长的", "long_hair": "长", "medium_hair": "中长", "short_hair": "短"}
HAIR_COL = {"black_hair": "黑色", "brown_hair": "棕色", "blonde_hair": "金色", "blue_hair": "蓝色",
            "purple_hair": "紫色", "silver_hair": "银色", "white_hair": "白色", "pink_hair": "粉色",
            "red_hair": "红色", "green_hair": "绿色", "grey_hair": "灰色", "orange_hair": "橙色",
            "aqua_hair": "青色", "multicolored_hair": "多色"}
HAIR_STYLE = {"twintails": "双马尾", "ponytail": "马尾", "braid": "编发", "ahoge": "呆毛", "blunt_bangs": "齐刘海",
              "bangs": "刘海", "sidelocks": "鬓发", "hair_bun": "丸子头", "side_ponytail": "侧马尾", "two_side_up": "两侧扎发"}
EYES = {"blue_eyes": "蓝色", "red_eyes": "红色", "brown_eyes": "棕色", "green_eyes": "绿色", "purple_eyes": "紫色",
        "yellow_eyes": "黄色", "pink_eyes": "粉色", "grey_eyes": "灰色", "aqua_eyes": "青色", "orange_eyes": "橙色",
        "black_eyes": "黑色", "heterochromia": "异色"}
EXPRESSION = {"smile": "微笑", ":d": "开心地笑", "open_mouth": "张着嘴", "closed_mouth": "闭着嘴", "blush": "脸红",
              "one_eye_closed": "眨一只眼", "fang": "露出小虎牙", "parted_lips": "嘴唇微张", "tears": "流泪",
              "closed_eyes": "闭着眼睛", "expressionless": "面无表情", "angry": "生气", "surprised": "惊讶"}
ACCESSORY = {"hair_ornament": "发饰", "hair_ribbon": "发带", "hair_bow": "发蝴蝶结", "hairband": "发箍",
             "hairclip": "发夹", "hair_flower": "头戴花朵", "hat": "帽子", "ribbon": "丝带", "bow": "蝴蝶结",
             "earrings": "耳环", "glasses": "眼镜", "choker": "颈环", "necklace": "项链", "animal_ears": "兽耳",
             "cat_ears": "猫耳", "tail": "尾巴", "horns": "角", "halo": "光环", "headphones": "耳机", "wings": "翅膀",
             "beret": "贝雷帽", "witch_hat": "女巫帽", "maid_headdress": "女仆头饰", "hood": "兜帽", "mask": "面具"}
CLOTHING = {"school_uniform": "校服", "serafuku": "水手服", "shirt": "衬衫", "white_shirt": "白衬衫", "jacket": "夹克",
            "dress": "连衣裙", "kimono": "和服", "japanese_clothes": "日式服装", "sailor_collar": "水手领",
            "necktie": "领带", "bowtie": "领结", "gloves": "手套", "detached_sleeves": "分离式袖子", "hoodie": "连帽衫",
            "sweater": "毛衣", "coat": "外套", "armor": "盔甲", "uniform": "制服", "maid": "女仆装",
            "swimsuit": "泳装", "skirt": "短裙", "scarf": "围巾", "vest": "马甲", "cape": "披风"}
FRAMING = {"upper_body": "上半身", "cowboy_shot": "七分身", "portrait": "头像特写", "close-up": "特写",
           "looking_at_viewer": "看向画面", "head_tilt": "歪着头", "looking_back": "回头看", "profile": "侧脸"}
BACKGROUND = {"white_background": "白色背景", "grey_background": "灰色背景", "simple_background": "简单背景",
              "outdoors": "户外", "sky": "天空背景", "night": "夜晚", "day": "白天", "blurry": "背景虚化",
              "indoors": "室内", "flower": "花朵", "cherry_blossoms": "樱花", "water": "水边", "snow": "雪景",
              "gradient_background": "渐变背景", "blue_background": "蓝色背景", "pink_background": "粉色背景"}


# a more specific tag makes the generic one redundant ("齐刘海、刘海", "校服、水手服", "微笑、开心地笑")
IMPLIES = {"blunt_bangs": "bangs", "serafuku": "school_uniform", ":d": "smile", "white_shirt": "shirt",
           "kimono": "japanese_clothes", "cat_ears": "animal_ears", "hair_ribbon": "ribbon", "hair_bow": "bow",
           "very_long_hair": "long_hair", "witch_hat": "hat", "beret": "hat"}


def pick(tags: set[str], table: dict[str, str]) -> list[str]:
    tags = tags - {g for s, g in IMPLIES.items() if s in tags}
    return [v for k, v in table.items() if k in tags]


def anime_caption(tags: set[str], rng: random.Random) -> dict[str, str]:
    subj = (pick(tags, SUBJECT) or ["角色"])[0]
    length = (pick(tags, HAIR_LEN) or [""])[0]
    colour = (pick(tags, HAIR_COL) or [""])[0]
    hair = f"{length}{colour}头发" if (length or colour) else ""
    hair = hair.replace("长头发", "长发").replace("短头发", "短发")
    styles = pick(tags, HAIR_STYLE)
    eyes = pick(tags, EYES)
    eye = f"{eyes[0]}眼睛" if eyes else ""
    expr = pick(tags, EXPRESSION)[:2]
    acc = pick(tags, ACCESSORY)[:3]
    cloth = pick(tags, CLOTHING)[:2]
    frame = pick(tags, FRAMING)[:2]
    bg = pick(tags, BACKGROUND)[:2]
    head = "，".join(x for x in [subj, hair, "、".join(styles[:2]), eye] if x)
    parts = [head]
    if expr:
        parts.append("、".join(expr))
    if acc:
        parts.append("戴着" + "、".join(acc))
    if cloth:
        parts.append("穿着" + "、".join(cloth))
    if frame:
        parts.append("、".join(frame))
    if bg:
        parts.append("、".join(bg))
    long = "，".join(parts) + "。"
    short = "，".join(x for x in [subj, hair, eye] if x) + "。"
    return {"short": short, "long": long}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", type=Path, required=True)
    ap.add_argument("--anime-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--face-split", type=Path, default=None, help="face split manifest (id/split JSONL) to merge")
    ap.add_argument("--split-out", type=Path, default=None,
                    help="write a combined faces + anime split manifest (anime split from its manifest)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n_face = n_anime = 0
    with args.out.open("w", encoding="utf-8") as f:
        for line in args.faces.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                f.write(json.dumps({"id": r["id"], "domain": "真人照片", "short": r["short"], "long": r["long"]},
                                   ensure_ascii=False) + "\n")
                n_face += 1
        for line in args.anime_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tags = {t.strip() for t in r.get("original_prompt", "").split(",") if t.strip()}
            f.write(json.dumps({"id": r["id"], "domain": "动漫", **anime_caption(tags, rng)}, ensure_ascii=False) + "\n")
            n_anime += 1
    print(f"{n_face} face + {n_anime} anime captions -> {args.out}")
    if args.split_out:
        rows = [line for line in args.face_split.read_text().splitlines() if line.strip()] if args.face_split else []
        for line in args.anime_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(json.dumps({"id": r["id"], "split": r["split"]}))
        args.split_out.write_text("\n".join(rows) + "\n")
        print(f"{len(rows)} split rows -> {args.split_out}")


if __name__ == "__main__":
    main()
