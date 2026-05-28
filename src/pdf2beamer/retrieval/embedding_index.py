"""Local embedding index for PaperChunk retrieval."""

from math import sqrt
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pdf2beamer.retrieval.chunker import PaperChunk
from pdf2beamer.retrieval.embeddings import BaseEmbedder


class EmbeddingRecord(BaseModel):
    """One embedded PaperChunk stored in an index."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, str] = Field(default_factory=dict)
    source_pages: list[int] = Field(default_factory=list)
    section_id: str | None = None
    section_title: str | None = None
    chunk_type: str


class SearchResult(BaseModel):
    """Search result returned by cosine top-k retrieval."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)
    source_pages: list[int] = Field(default_factory=list)
    section_id: str | None = None
    section_title: str | None = None
    chunk_type: str


class EmbeddingIndex(BaseModel):
    """Serializable local vector index backed by a list of records."""

    model_config = ConfigDict(extra="forbid")

    records: list[EmbeddingRecord] = Field(default_factory=list)
    embedding_dim: int | None = Field(default=None, ge=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "EmbeddingIndex":
        """Ensure records agree with the declared embedding dimension."""

        if not self.records:
            return self

        first_dim = len(self.records[0].embedding)
        if first_dim == 0:
            raise ValueError("Embedding records must not contain empty embeddings.")
        expected_dim = self.embedding_dim or first_dim
        for record in self.records:
            if len(record.embedding) != expected_dim:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"record {record.chunk_id} has {len(record.embedding)} values, "
                    f"expected {expected_dim}.",
                )
        self.embedding_dim = expected_dim
        return self

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return records sorted by descending cosine similarity."""

        if top_k <= 0 or not self.records:
            return []
        if self.embedding_dim is not None and len(query_embedding) != self.embedding_dim:
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"got {len(query_embedding)}, expected {self.embedding_dim}.",
            )

        results = [
            SearchResult(
                chunk_id=record.chunk_id,
                text=record.text,
                score=cosine_similarity(query_embedding, record.embedding),
                metadata=record.metadata,
                source_pages=record.source_pages,
                section_id=record.section_id,
                section_title=record.section_title,
                chunk_type=record.chunk_type,
            )
            for record in self.records
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def save_json(self, path: str | Path) -> None:
        """Save the index to JSON."""

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "EmbeddingIndex":
        """Load an index from JSON."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity with explicit dimension and zero-vector handling."""

    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}.")

    norm_a = sqrt(sum(value * value for value in a))
    norm_b = sqrt(sum(value * value for value in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    score = sum(left * right for left, right in zip(a, b, strict=True)) / (norm_a * norm_b)
    return max(-1.0, min(1.0, score))


def build_embedding_index(chunks: list[PaperChunk], embedder: BaseEmbedder) -> EmbeddingIndex:
    """Embed PaperChunk objects and build a local index."""

    warnings: list[str] = []
    if not chunks:
        return EmbeddingIndex(records=[], embedding_dim=None, warnings=["No chunks provided."])

    valid_chunks: list[tuple[PaperChunk, str]] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            warnings.append(f"Skipped empty chunk: {chunk.id}")
            continue
        valid_chunks.append((chunk, text))

    if not valid_chunks:
        return EmbeddingIndex(records=[], embedding_dim=None, warnings=warnings)

    try:
        embeddings = embedder.embed_texts([text for _chunk, text in valid_chunks])
    except Exception as exc:
        raise ValueError(f"Failed to embed {len(valid_chunks)} chunks: {exc}") from exc

    if len(embeddings) != len(valid_chunks):
        raise ValueError(
            "Embedder returned an unexpected number of vectors: "
            f"got {len(embeddings)}, expected {len(valid_chunks)}."
        )

    records: list[EmbeddingRecord] = []
    embedding_dim: int | None = None
    for (chunk, text), embedding in zip(valid_chunks, embeddings, strict=True):
        if not embedding:
            raise ValueError(f"Embedding for chunk {chunk.id} is empty.")
        if embedding_dim is None:
            embedding_dim = len(embedding)
        elif len(embedding) != embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"chunk {chunk.id} has {len(embedding)} values, expected {embedding_dim}.",
            )

        records.append(
            EmbeddingRecord(
                chunk_id=chunk.id,
                text=text,
                embedding=embedding,
                metadata=chunk.metadata,
                source_pages=chunk.source_pages,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                chunk_type=chunk.chunk_type,
            ),
        )

    return EmbeddingIndex(records=records, embedding_dim=embedding_dim, warnings=warnings)


def search_chunks(
    query: str,
    index: EmbeddingIndex,
    embedder: BaseEmbedder,
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed a query and search the embedding index."""

    if top_k <= 0:
        return []
    query_embedding = embedder.embed_text(query)
    return index.search(query_embedding, top_k=top_k)
