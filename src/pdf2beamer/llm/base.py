"""Dependency-injected interfaces for local model components."""

from abc import ABC, abstractmethod
from typing import Any

JsonDict = dict[str, Any]


class BaseGenerator(ABC):
    """Interface for structured local generation."""

    @abstractmethod
    def generate_json(self, prompt: str, schema_name: str | None = None) -> JsonDict | str:
        """Generate a JSON object or JSON string for a prompt."""

    def generate_text(self, prompt: str) -> str:
        """Optional text-generation path for implementations that expose it."""

        result = self.generate_json(prompt)
        return str(result)


class FakeGenerator(BaseGenerator):
    """Deterministic test generator for structured JSON workflows."""

    def __init__(self, response: JsonDict | None = None) -> None:
        self.response = response

    def generate_json(self, prompt: str, schema_name: str | None = None) -> JsonDict:
        del schema_name
        if self.response is not None:
            return self.response
        return _default_argument_graph_response(prompt)


class BaseEmbedder(ABC):
    """Legacy interface for local text embeddings.

    Retrieval modules define the current embedding interface. This class remains
    for compatibility with early adapter placeholders.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts locally."""


class BaseReranker(ABC):
    """Legacy interface for local query-document reranking."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Return document indices and scores sorted by relevance."""


class FakeEmbedder(BaseEmbedder):
    """Deterministic legacy test embedder that does not load weights."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(sum(map(ord, text)) % 997)] for text in texts]


class FakeReranker(BaseReranker):
    """Deterministic legacy test reranker preserving input order."""

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        del query
        return [(index, 1.0 / (index + 1)) for index, _ in enumerate(documents)]


def _default_argument_graph_response(prompt: str) -> JsonDict:
    chunk_ids = _extract_prompt_chunk_ids(prompt)
    pages = _extract_prompt_pages(prompt)
    evidence = chunk_ids[:1]
    source_pages = pages[:1]
    return {
        "nodes": [
            {
                "id": "problem_1",
                "type": "problem",
                "text": (
                    "The paper addresses a scientific problem described in the "
                    "provided chunks."
                ),
                "evidence_chunk_ids": evidence,
                "source_pages": source_pages,
                "confidence": 0.6,
            },
            {
                "id": "contribution_1",
                "type": "contribution",
                "text": "The paper contributes a method grounded in the provided chunks.",
                "evidence_chunk_ids": evidence,
                "source_pages": source_pages,
                "confidence": 0.6,
            },
            {
                "id": "method_1",
                "type": "method",
                "text": "The method is summarized from the provided chunks.",
                "evidence_chunk_ids": evidence,
                "source_pages": source_pages,
                "confidence": 0.6,
            },
            {
                "id": "result_1",
                "type": "result",
                "text": "The results are summarized only from the provided chunks.",
                "evidence_chunk_ids": evidence,
                "source_pages": source_pages,
                "confidence": 0.6,
            },
            {
                "id": "takeaway_1",
                "type": "takeaway",
                "text": "The takeaway is based on the provided evidence.",
                "evidence_chunk_ids": evidence,
                "source_pages": source_pages,
                "confidence": 0.6,
            },
        ],
        "edges": [
            {
                "source": "problem_1",
                "target": "contribution_1",
                "relation": "addressed_by",
                "confidence": 0.6,
            },
            {
                "source": "contribution_1",
                "target": "method_1",
                "relation": "implemented_by",
                "confidence": 0.6,
            },
            {
                "source": "method_1",
                "target": "result_1",
                "relation": "evaluated_by",
                "confidence": 0.6,
            },
            {
                "source": "result_1",
                "target": "takeaway_1",
                "relation": "summarizes",
                "confidence": 0.6,
            },
        ],
    }


def _extract_prompt_chunk_ids(prompt: str) -> list[str]:
    ids: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("[chunk_id:") and stripped.endswith("]"):
            ids.append(stripped.removeprefix("[chunk_id:").removesuffix("]").strip())
    return ids


def _extract_prompt_pages(prompt: str) -> list[int]:
    pages: list[int] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped.startswith("[pages:") or not stripped.endswith("]"):
            continue
        raw_pages = stripped.removeprefix("[pages:").removesuffix("]")
        for part in raw_pages.split(","):
            try:
                pages.append(int(part.strip()))
            except ValueError:
                continue
    return pages
