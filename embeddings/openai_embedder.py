from __future__ import annotations

import numpy as np
from typing import Optional

from embeddings.base import BaseEmbedder, EmbeddingOutput


class OpenAIEmbedder(BaseEmbedder):
    """Optional experimental backend.

    Text uses the embeddings API. Images are captioned first and then embedded,
    so this is NOT equivalent to a native image embedding model; keep that caveat
    when comparing against CLIP.
    """
    def __init__(self, text_model: str, caption_model: str, batch_size: int = 64):
        from openai import OpenAI

        self.client = OpenAI()
        self.text_model = text_model
        self.caption_model = caption_model
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> EmbeddingOutput:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=self.text_model, input=batch)
            out.extend(d.embedding for d in resp.data)
        values = np.asarray(out, dtype=np.float32)
        return EmbeddingOutput(values, np.ones(len(texts), dtype=bool))

    def embed_images(self, image_urls: list[Optional[str]]) -> EmbeddingOutput:
        captions: list[str] = []
        mask = []
        for url in image_urls:
            caption = self._caption(url)
            captions.append(caption)
            mask.append(bool(caption))
        result = self.embed_texts(captions)
        result.valid_mask = np.asarray(mask, dtype=bool)
        result.values[~result.valid_mask] = 0.0
        return result

    def _caption(self, url: Optional[str]) -> str:
        if not url:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.caption_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this product image in one concise sentence."},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""
