"""Download default local model assets from Hugging Face."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pdf2beamer.model_factory import (
    _DEFAULT_EMBEDDING_DIR,
    _DEFAULT_RERANKER_DIR,
    _NEMOTRON_GGUF_DIR,
)

_NEMOTRON_FILENAME = "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"


class DownloadFn(Protocol):
    def __call__(self, *, repo_id: str, local_dir: str, **kwargs: object) -> str: ...


@dataclass(frozen=True)
class ModelDownloadSpec:
    name: str
    repo_id: str
    local_dir: Path
    filename: str | None = None


def default_model_specs(base_dir: Path = Path(".")) -> list[ModelDownloadSpec]:
    """Return the default local model layout expected by real-model backends."""

    return [
        ModelDownloadSpec(
            name="Nemotron generator",
            repo_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            filename=_NEMOTRON_FILENAME,
            local_dir=base_dir / _NEMOTRON_GGUF_DIR,
        ),
        ModelDownloadSpec(
            name="Qwen embedding",
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            local_dir=base_dir / _DEFAULT_EMBEDDING_DIR,
        ),
        ModelDownloadSpec(
            name="Qwen reranker",
            repo_id="Qwen/Qwen3-Reranker-0.6B",
            local_dir=base_dir / _DEFAULT_RERANKER_DIR,
        ),
    ]


def download_default_models(
    base_dir: Path = Path("."),
    *,
    force: bool = False,
    token: str | None = None,
    download_fn: DownloadFn | None = None,
) -> list[Path]:
    """Download default models and return the local model directories."""

    if download_fn is None:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required for model downloads. "
                "Install with: pip install 'pdf2beamer[models]'"
            ) from exc

        def download_fn(*, repo_id: str, local_dir: str, **kwargs: object) -> str:
            filename = kwargs.pop("filename", None)
            if filename:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=str(filename),
                    local_dir=local_dir,
                    token=token,
                    force_download=force,
                )
            return snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=token,
                force_download=force,
            )

    downloaded: list[Path] = []
    for spec in default_model_specs(base_dir):
        spec.local_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, object] = {}
        if spec.filename:
            kwargs["filename"] = spec.filename
        download_fn(repo_id=spec.repo_id, local_dir=str(spec.local_dir), **kwargs)
        downloaded.append(spec.local_dir)
    return downloaded
