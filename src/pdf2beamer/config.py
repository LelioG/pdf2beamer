"""Configuration models for the pdf2beamer pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GeneratorBackend = Literal["fake", "llama_cpp"]
EmbedderBackend = Literal["fake", "sentence_transformers_qwen"]
RerankerBackend = Literal["fake", "transformers_qwen"]


class PipelineConfig(BaseModel):
    """Runtime configuration for local PDF-to-Beamer generation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model_path: Path | None = None
    embedding_model_path: Path | None = None
    reranker_model_path: Path | None = None
    duration_minutes: int = Field(default=10, ge=1, le=180)
    audience: str = Field(default="technical", min_length=1)
    theme: str = Field(default="clean", min_length=1)
    max_slides: int | None = Field(default=None, ge=1, le=200)
    compile_pdf: bool = True
    latex_engine: str = "latexmk"
    save_intermediate: bool = True
    debug: bool = False
    extract_images: bool = True
    use_fake_models: bool = True
    fail_on_error: bool = True

    generator_backend: GeneratorBackend = "fake"
    n_ctx: int = Field(default=8192, ge=512)
    n_gpu_layers: int = -1
    llama_main_gpu: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=2048, ge=1)
    llama_verbose: bool = False
    llama_use_instructor: bool = True
    instructor_max_retries: int = Field(default=2, ge=0, le=10)

    embedder_backend: EmbedderBackend = "fake"
    embedding_instruction: str | None = None
    embedding_batch_size: int = Field(default=8, ge=1)
    embedding_device: str | None = None

    reranker_backend: RerankerBackend = "fake"
    reranker_instruction: str | None = None
    reranker_batch_size: int = Field(default=8, ge=1)
    reranker_device: str | None = None
    reranker_max_length: int = Field(default=2048, ge=256, le=8192)

    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    rerank_top_k: int = Field(default=3, ge=1, le=20)
    max_context_chars: int = Field(default=1200, ge=200, le=8000)

    @field_validator(
        "audience",
        "theme",
        "latex_engine",
        "generator_backend",
        "embedder_backend",
        "reranker_backend",
    )
    @classmethod
    def normalize_non_empty_text(cls, value: str) -> str:
        """Normalize compact user-facing configuration strings."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator(
        "embedding_instruction",
        "embedding_device",
        "reranker_instruction",
        "reranker_device",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Normalize optional configuration strings."""

        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
