from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image

from embeddings.base import BaseEmbedder, EmbeddingOutput


class ClipEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, batch_size: int = 64, device: Optional[str] = None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> EmbeddingOutput:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            out.append(self.model.get_text_features(**inputs).cpu().numpy())
        values = np.concatenate(out, axis=0) if out else np.empty((0, 0), dtype=np.float32)
        return EmbeddingOutput(values=values, valid_mask=np.ones(len(texts), dtype=bool))

    @torch.no_grad()
    def embed_images(self, image_urls: list[Optional[str]]) -> EmbeddingOutput:
        vectors = []
        masks = []
        # Process valid images in each batch. Invalid images receive a zero vector
        # after the embedding dimension becomes known; they are NOT represented by
        # the same gray dummy image (which would create artificial similarity).
        dim = int(self.model.config.projection_dim)
        for i in range(0, len(image_urls), self.batch_size):
            batch_urls = image_urls[i:i + self.batch_size]
            loaded = [self._safe_load(url) for url in batch_urls]
            valid_positions = [j for j, image in enumerate(loaded) if image is not None]
            batch_vecs = np.zeros((len(batch_urls), dim), dtype=np.float32)
            if valid_positions:
                images = [loaded[j] for j in valid_positions]
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs).cpu().numpy()
                for row, pos in enumerate(valid_positions):
                    batch_vecs[pos] = feats[row]
            vectors.append(batch_vecs)
            masks.extend([x is not None for x in loaded])
        values = np.concatenate(vectors, axis=0) if vectors else np.empty((0, dim), dtype=np.float32)
        return EmbeddingOutput(values=values, valid_mask=np.asarray(masks, dtype=bool))

    @staticmethod
    def _safe_load(url: Optional[str]) -> Optional[Image.Image]:
        if not url:
            return None
        try:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None
