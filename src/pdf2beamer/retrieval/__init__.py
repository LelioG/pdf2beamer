"""Chunking, embedding, indexing, and reranking modules."""

from pdf2beamer.retrieval.chunker import PaperChunk, chunk_paper_ir, is_informative_chunk_text
from pdf2beamer.retrieval.embedding_index import (
    EmbeddingIndex,
    EmbeddingRecord,
    SearchResult,
    build_embedding_index,
    cosine_similarity,
    search_chunks,
)
from pdf2beamer.retrieval.embeddings import BaseEmbedder, FakeEmbedder
from pdf2beamer.retrieval.qwen_embedding import LocalQwenEmbedder
from pdf2beamer.retrieval.qwen_reranker import LocalQwenReranker
from pdf2beamer.retrieval.reranker import (
    BaseReranker,
    FakeReranker,
    RerankCandidate,
    RerankResult,
    combine_scores,
    normalize_retrieval_score,
    rerank_candidates,
    retrieve_and_rerank,
    search_result_to_rerank_candidate,
)

__all__ = [
    "BaseEmbedder",
    "EmbeddingIndex",
    "EmbeddingRecord",
    "FakeEmbedder",
    "LocalQwenEmbedder",
    "PaperChunk",
    "is_informative_chunk_text",
    "search_result_to_rerank_candidate",
    "retrieve_and_rerank",
    "rerank_candidates",
    "normalize_retrieval_score",
    "combine_scores",
    "RerankResult",
    "RerankCandidate",
    "FakeReranker",
    "LocalQwenReranker",
    "BaseReranker",
    "SearchResult",
    "build_embedding_index",
    "chunk_paper_ir",
    "cosine_similarity",
    "search_chunks",
]
