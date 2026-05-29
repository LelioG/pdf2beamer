"""Factories for selecting fake or local model backends from PipelineConfig."""

from pathlib import Path

from pdf2beamer.config import PipelineConfig
from pdf2beamer.errors import InvalidModelConfigurationError
from pdf2beamer.llm import BaseGenerator, FakeGenerator, LocalNemotronGenerator
from pdf2beamer.retrieval import BaseEmbedder, BaseReranker, FakeEmbedder, FakeReranker
from pdf2beamer.retrieval.qwen_embedding import LocalQwenEmbedder
from pdf2beamer.retrieval.qwen_reranker import LocalQwenReranker

_NEMOTRON_GGUF_DIR = Path("models/nemotron-3-nano-4b-gguf")
_DEFAULT_EMBEDDING_DIR = Path("models/Qwen3-Embedding-0.6B")
_DEFAULT_RERANKER_DIR = Path("models/Qwen3-Reranker-0.6B")


def create_generator(config: PipelineConfig) -> BaseGenerator:
    """Create the configured structured generator backend."""

    if config.use_fake_models or config.generator_backend == "fake":
        return FakeGenerator()
    if config.generator_backend == "llama_cpp":
        model_path = _resolve_generator_path(config)
        return LocalNemotronGenerator(
            model_path=model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            verbose=config.llama_verbose,
            main_gpu=config.llama_main_gpu,
            use_instructor=config.llama_use_instructor,
            instructor_max_retries=config.instructor_max_retries,
        )
    raise InvalidModelConfigurationError(
        f"Unsupported generator backend: {config.generator_backend}"
    )


def create_embedder(config: PipelineConfig) -> BaseEmbedder:
    """Create the configured embedding backend."""

    if config.use_fake_models or config.embedder_backend == "fake":
        return FakeEmbedder()
    if config.embedder_backend == "sentence_transformers_qwen":
        model_path = config.embedding_model_path or _DEFAULT_EMBEDDING_DIR
        if not model_path.exists():
            raise InvalidModelConfigurationError(
                "embedding_model_path is required for sentence_transformers_qwen embeddings "
                "when models/Qwen3-Embedding-0.6B is not present."
            )
        return LocalQwenEmbedder(
            model_path=model_path,
            instruction=config.embedding_instruction,
            batch_size=config.embedding_batch_size,
            device=config.embedding_device,
        )
    raise InvalidModelConfigurationError(f"Unsupported embedder backend: {config.embedder_backend}")


def create_reranker(config: PipelineConfig) -> BaseReranker:
    """Create the configured reranking backend."""

    if config.use_fake_models or config.reranker_backend == "fake":
        return FakeReranker()
    if config.reranker_backend == "transformers_qwen":
        model_path = config.reranker_model_path or _DEFAULT_RERANKER_DIR
        if not model_path.exists():
            raise InvalidModelConfigurationError(
                "reranker_model_path is required for transformers_qwen reranking "
                "when models/Qwen3-Reranker-0.6B is not present."
            )
        return LocalQwenReranker(
            model_path=model_path,
            instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            device=config.reranker_device,
            max_length=config.reranker_max_length,
        )
    raise InvalidModelConfigurationError(f"Unsupported reranker backend: {config.reranker_backend}")


def _resolve_generator_path(config: PipelineConfig) -> Path:
    if config.model_path is not None:
        return config.model_path

    candidates = sorted(_NEMOTRON_GGUF_DIR.glob("*.gguf")) if _NEMOTRON_GGUF_DIR.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise InvalidModelConfigurationError(
            f"Multiple Nemotron GGUF files found under {_NEMOTRON_GGUF_DIR}; pass --model."
        )
    raise InvalidModelConfigurationError(
        "model_path is required for Nemotron generation when no .gguf file is present "
        f"under: {_NEMOTRON_GGUF_DIR}."
    )
