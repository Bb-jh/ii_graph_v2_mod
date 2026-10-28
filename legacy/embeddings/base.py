from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EmbeddingOutput:
    values: np.ndarray
    valid_mask: np.ndarray


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> EmbeddingOutput:
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, image_urls: list[Optional[str]]) -> EmbeddingOutput:
        raise NotImplementedError
