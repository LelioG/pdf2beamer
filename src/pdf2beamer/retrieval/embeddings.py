"""Embedding interfaces for local retrieval.

Qwen3-Embedding is implemented in qwen_embedding.py. This module keeps the
base interface and deterministic FakeEmbedder used by tests and local
development without model downloads.
"""

from abc import ABC, abstractmethod
import hashlib
import math
import re


class BaseEmbedder(ABC):
    """Abstract interface for local text embedding backends."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Implementations may override this with a more efficient batched path.
        """

        return [self.embed_text(text) for text in texts]


class FakeEmbedder(BaseEmbedder):
    """Deterministic hash-bucket embedder for tests.

    It uses stable BLAKE2b hashes instead of Python's randomized ``hash()`` and
    L2-normalizes the output so cosine similarity behaves predictably.
    """

    def __init__(self, dimension: int = 16) -> None:
        if dimension <= 0:
            raise ValueError("FakeEmbedder dimension must be positive.")
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokens(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dimension
            sign = 1.0 if ((value >> 8) & 1) == 0 else -1.0
            vector[bucket] += sign

        return _l2_normalize(vector)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
