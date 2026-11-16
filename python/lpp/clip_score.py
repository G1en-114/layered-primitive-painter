"""Frozen Chinese-CLIP image/text features for evaluation metrics and RL rewards."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLIP_CONTEXT = 52   # Chinese-CLIP was trained with 52 text tokens; score with the same context


class ClipScorer:
    def __init__(self, model_dir: Path | str, device: str):
        from transformers import BertTokenizer, ChineseCLIPImageProcessor, ChineseCLIPModel
        self.model = ChineseCLIPModel.from_pretrained(str(model_dir)).to(device).eval()
        self.tok = BertTokenizer.from_pretrained(str(model_dir))
        self.proc = ChineseCLIPImageProcessor.from_pretrained(str(model_dir))
        self.device = device

    @torch.no_grad()
    def images(self, arrays: list[np.ndarray], batch: int = 64) -> torch.Tensor:
        out = []
        for i in range(0, len(arrays), batch):
            px = self.proc(images=[Image.fromarray(a) for a in arrays[i:i + batch]], return_tensors="pt")
            vis = self.model.vision_model(pixel_values=px["pixel_values"].to(self.device))
            out.append(self.model.visual_projection(vis.pooler_output).float())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)

    @torch.no_grad()
    def texts(self, captions: list[str], batch: int = 64) -> torch.Tensor:
        # Explicit [CLS] pooling + projection, as in Chinese-CLIP: some transformers releases
        # build the text tower without a pooler, and get_text_features then fails on None
        out = []
        for i in range(0, len(captions), batch):
            enc = self.tok(captions[i:i + batch], padding=True, truncation=True, max_length=CLIP_CONTEXT,
                           return_tensors="pt").to(self.device)
            hidden = self.model.text_model(**enc).last_hidden_state
            out.append(self.model.text_projection(hidden[:, 0]).float())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)
