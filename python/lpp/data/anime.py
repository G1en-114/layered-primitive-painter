"""Portrait screening, crop geometry and duplicate grouping for anime images."""
from __future__ import annotations

import hashlib
import re

import numpy as np
from PIL import Image, ImageOps

SOURCE = "huanngzh/anime_face_control_60k"
ID_OFFSET = 200_000_000
# Conservative screening for the initial portrait subset; not a content classifier.
EXCLUDED_TAGS = frozenset({"nude", "naked", "nipples", "pussy", "penis", "sex", "sexual_intercourse",
                          "panties", "underwear", "lingerie", "bra", "spread_legs", "multiple_girls",
                          "multiple_boys", "comic", "multiple_views"})
COLORS = {"black": "黑色", "white": "白色", "silver": "银色", "grey": "灰色", "gray": "灰色",
          "brown": "棕色", "blonde": "金色", "red": "红色", "pink": "粉色", "blue": "蓝色",
          "green": "绿色", "purple": "紫色", "orange": "橙色", "yellow": "黄色", "aqua": "青色"}


def tags_from_prompt(prompt: str) -> set[str]:
    return {tag.strip().lower().replace(" ", "_") for tag in prompt.split(",") if tag.strip()}


def screen_tags(tags: set[str]) -> str | None:
    if tags & EXCLUDED_TAGS:
        return "excluded_tags"
    counts = [int(m.group(1)) for tag in tags if (m := re.fullmatch(r"(\d+)(?:girl|boy)s?", tag))]
    if not counts or sum(counts) != 1:
        return "not_single_person"
    return None


def face_crop_box(landmarks, size: tuple[int, int], min_fraction=0.18, expansion=2.15):
    """Square head crop from all 28 supplied face points; no guessed anatomical indices."""
    points = np.asarray(landmarks, dtype=np.float64)
    w, h = size
    if points.shape != (28, 2) or not np.isfinite(points).all():
        raise ValueError("invalid_landmarks")
    if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
        raise ValueError("landmarks_outside_image")
    low, high = points.min(0), points.max(0)
    extent = high - low
    if extent.min() < 8 or extent.max() < min(w, h) * min_fraction:
        raise ValueError("face_too_small")
    if not 0.45 <= extent[0] / extent[1] <= 2.0:
        raise ValueError("extreme_landmark_geometry")
    side = min(min(w, h), max(1, round(float(extent.max()) * expansion)))
    center = (low + high) / 2
    # Shift upwards to retain hair rather than extending far into the torso.
    left = min(max(0, round(float(center[0]) - side / 2)), w - side)
    top = min(max(0, round(float(center[1]) - side * 0.63)), h - side)
    box = (left, top, left + side, top + side)
    if (points[:, 0] < left).any() or (points[:, 0] >= left + side).any() or \
            (points[:, 1] < top).any() or (points[:, 1] >= top + side).any():
        raise ValueError("landmarks_outside_crop")
    return box


def template_captions(tags: set[str]) -> dict[str, str]:
    """Use face-related tags; original full-body captions are retained only as metadata."""
    details = []
    for suffix, noun in (("hair", "头发"), ("eyes", "眼睛")):
        values = list(dict.fromkeys(zh for en, zh in COLORS.items() if f"{en}_{suffix}" in tags))
        if values:
            details.append("与".join(values) + noun)
    for tag, zh in (("smile", "微笑"), ("open_mouth", "张嘴"), ("closed_eyes", "闭眼"),
                    ("blush", "脸颊泛红"), ("glasses", "戴眼镜")):
        if tag in tags:
            details.append(zh)
    return {"short": "动漫人物头像", "long": "，".join(["动漫人物头像", *details]) + "。"}


_DCT = np.cos(np.pi / 32 * (np.arange(32)[None, :] + 0.5) * np.arange(8)[:, None])


def phash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coefficients = (_DCT @ pixels @ _DCT.T).ravel()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return sum(int(value) << i for i, value in enumerate(bits))


class DuplicateIndex:
    """Exact pixel hashes and pHash BK-tree with RGB verification, including mirrors.

    Conservative near-duplicate detection, not an identity/person matching model.
    """

    def __init__(self, distance=4, max_rgb_error=0.035):
        self.distance, self.max_rgb_error = distance, max_rgb_error
        self.root = None
        self.exact = {}
        self.images = {}

    def _neighbors(self, key):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (key ^ node[0]).bit_count()
            if distance <= self.distance:
                yield from node[1]
            pending.extend(child for edge, child in node[2].items()
                           if distance - self.distance <= edge <= distance + self.distance)

    def _insert(self, key, image_id):
        if self.root is None:
            self.root = [key, [image_id], {}]
            return
        node = self.root
        while True:
            distance = (key ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(image_id)
                return
            if distance not in node[2]:
                node[2][distance] = [key, [image_id], {}]
                return
            node = node[2][distance]

    def add(self, image: Image.Image, image_id: int) -> int | None:
        image = image.convert("RGB")
        mirrored = ImageOps.mirror(image)
        exact = min(hashlib.sha256(image.tobytes()).hexdigest(),
                    hashlib.sha256(mirrored.tobytes()).hexdigest())
        if exact in self.exact:
            return self.exact[exact]
        small = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS)).astype(np.float32) / 255
        key, mirror_key = phash(image), phash(mirrored)
        for candidate in set(self._neighbors(key)) | set(self._neighbors(mirror_key)):
            other = self.images[candidate]
            error = min(np.abs(small - other).mean(), np.abs(small[:, ::-1] - other).mean())
            if error <= self.max_rgb_error:
                return candidate
        self.exact[exact] = image_id
        self.images[image_id] = small
        self._insert(key, image_id)
        return None


def split_for_id(image_id: int, val_fraction=0.05) -> str:
    value = int.from_bytes(hashlib.sha256(f"anime-portrait-v1:{image_id}".encode()).digest()[:8], "big")
    return "val" if value / 2**64 < val_fraction else "train"
