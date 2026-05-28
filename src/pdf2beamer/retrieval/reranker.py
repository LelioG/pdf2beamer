"""Local reranking abstraction for retrieved PaperChunk candidates."""

import re
from abc import ABC, abstractmethod
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.retrieval.embedding_index import EmbeddingIndex, SearchResult, search_chunks
from pdf2beamer.retrieval.embeddings import BaseEmbedder


class RerankCandidate(BaseModel):
    """Candidate chunk passed from embedding retrieval into a reranker."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    retrieval_score: float
    metadata: dict[str, str] = Field(default_factory=dict)
    source_pages: list[int] = Field(default_factory=list)
    section_id: str | None = None
    section_title: str | None = None
    chunk_type: str


class RerankResult(BaseModel):
    """Reranked chunk result with original and reranker scores."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    retrieval_score: float
    rerank_score: float
    combined_score: float
    metadata: dict[str, str] = Field(default_factory=dict)
    source_pages: list[int] = Field(default_factory=list)
    section_id: str | None = None
    section_title: str | None = None
    chunk_type: str
    rank: int | None = Field(default=None, ge=1)


class BaseReranker(ABC):
    """Abstract interface for local reranking backends."""

    @abstractmethod
    def score(self, query: str, text: str) -> float:
        """Score one query-text pair."""

    def score_batch(self, query: str, texts: list[str]) -> list[float]:
        """Score a batch of query-text pairs.

        Real local rerankers can override this with a batched inference path.
        """

        return [self.score(query, text) for text in texts]


class FakeReranker(BaseReranker):
    """Deterministic lexical-overlap reranker for tests."""

    def score(self, query: str, text: str) -> float:
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return 0.0
        text_tokens = set(_tokens(text))
        if not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)


def rerank_candidates(
    query: str,
    candidates: list[RerankCandidate],
    reranker: BaseReranker,
    top_k: int = 5,
    retrieval_weight: float = 0.35,
    rerank_weight: float = 0.65,
) -> list[RerankResult]:
    """Rerank candidates and return top-k results by combined score."""

    if top_k <= 0 or not candidates:
        return []

    scores = reranker.score_batch(query, [candidate.text for candidate in candidates])
    if len(scores) != len(candidates):
        raise ValueError(
            "Reranker returned an unexpected number of scores: "
            f"got {len(scores)}, expected {len(candidates)}.",
        )

    results = [
        RerankResult(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            retrieval_score=candidate.retrieval_score,
            rerank_score=_clamp_01(score),
            combined_score=combine_scores(
                retrieval_score=candidate.retrieval_score,
                rerank_score=score,
                retrieval_weight=retrieval_weight,
                rerank_weight=rerank_weight,
            ),
            metadata=candidate.metadata,
            source_pages=candidate.source_pages,
            section_id=candidate.section_id,
            section_title=candidate.section_title,
            chunk_type=candidate.chunk_type,
        )
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    results.sort(key=lambda result: result.combined_score, reverse=True)
    top_results = results[:top_k]
    return [
        result.model_copy(update={"rank": index + 1}) for index, result in enumerate(top_results)
    ]


def search_result_to_rerank_candidate(search_result: SearchResult) -> RerankCandidate:
    """Convert an embedding SearchResult into a reranker candidate."""

    return RerankCandidate(
        chunk_id=search_result.chunk_id,
        text=search_result.text,
        retrieval_score=search_result.score,
        metadata=search_result.metadata,
        source_pages=search_result.source_pages,
        section_id=search_result.section_id,
        section_title=search_result.section_title,
        chunk_type=search_result.chunk_type,
    )


def retrieve_and_rerank(
    query: str,
    index: EmbeddingIndex,
    embedder: BaseEmbedder,
    reranker: BaseReranker,
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
) -> list[RerankResult]:
    """Run embedding retrieval and local reranking as a single operation."""

    if retrieval_top_k <= 0 or rerank_top_k <= 0:
        return []
    search_results = search_chunks(query, index, embedder, top_k=retrieval_top_k)
    candidates = [search_result_to_rerank_candidate(result) for result in search_results]
    return rerank_candidates(query, candidates, reranker, top_k=rerank_top_k)


def normalize_retrieval_score(score: float) -> float:
    """Map cosine-like retrieval scores from [-1, 1] into [0, 1]."""

    if not isfinite(score):
        return 0.0
    return _clamp_01((score + 1.0) / 2.0)


def combine_scores(
    retrieval_score: float,
    rerank_score: float,
    retrieval_weight: float,
    rerank_weight: float,
) -> float:
    """Combine normalized retrieval score and clamped rerank score."""

    rerank_score = _clamp_01(rerank_score)
    total_weight = retrieval_weight + rerank_weight
    if total_weight == 0.0:
        return rerank_score

    normalized_retrieval = normalize_retrieval_score(retrieval_score)
    retrieval_weight = retrieval_weight / total_weight
    rerank_weight = rerank_weight / total_weight
    return retrieval_weight * normalized_retrieval + rerank_weight * rerank_score


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _clamp_01(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
