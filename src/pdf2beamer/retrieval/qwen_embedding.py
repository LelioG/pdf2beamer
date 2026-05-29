"""Local Qwen embedding adapter backed by sentence-transformers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pdf2beamer.errors import (
    LocalModelInferenceError,
    LocalModelLoadError,
    OptionalDependencyNotInstalledError,
)
from pdf2beamer.retrieval.embeddings import BaseEmbedder


class LocalQwenEmbedder(BaseEmbedder):
    """Embed text with a local Qwen3-Embedding model folder."""

    def __init__(
        self,
        model_path: str | Path,
        instruction: str | None = None,
        batch_size: int = 8,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.instruction = instruction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self._model = self._load_model()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return plain Python float lists."""

        prepared = [self._prepare_text(text) for text in texts]
        try:
            embeddings = self._model.encode(
                prepared,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
            )
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            raise LocalModelInferenceError(f"Local Qwen embedding failed: {exc}") from exc
        return [_to_float_list(row) for row in _as_rows(embeddings)]

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise LocalModelLoadError(
                f"Local Qwen embedding model directory does not exist: {self.model_path}"
            )
        if not self.model_path.is_dir():
            raise LocalModelLoadError(
                f"Local Qwen embedding model path is not a directory: {self.model_path}"
            )
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise OptionalDependencyNotInstalledError(
                "sentence-transformers is not installed. "
                "Install pdf2beamer with the [models] extra."
            ) from exc
        try:
            return sentence_transformers.SentenceTransformer(
                str(self.model_path),
                local_files_only=True,
                device=self.device,
            )
        except TypeError:
            try:
                if self.device is None:
                    return sentence_transformers.SentenceTransformer(str(self.model_path))
                return sentence_transformers.SentenceTransformer(
                    str(self.model_path),
                    device=self.device,
                )
            except Exception as exc:  # pragma: no cover - backend-specific failure path
                raise LocalModelLoadError(
                    f"Failed to load local Qwen embedding model: {exc}"
                ) from exc
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            raise LocalModelLoadError(f"Failed to load local Qwen embedding model: {exc}") from exc

    def _prepare_text(self, text: str) -> str:
        if not self.instruction:
            return text
        return f"Instruct: {self.instruction}\nQuery: {text}"


def _as_rows(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, (int, float)) for item in value):
            return [value]
        return value
    return list(value)


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]
