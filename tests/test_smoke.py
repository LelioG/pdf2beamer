from pathlib import Path

import pytest
from typer.testing import CliRunner

from pdf2beamer.cli import app

import pdf2beamer
from pdf2beamer import PipelineConfig, create_embedder, create_generator, create_reranker
from pdf2beamer.beamer import BeamerRenderer
from pdf2beamer.errors import InvalidModelConfigurationError
from pdf2beamer.ir import Slide, SlideBullet, SlideIR
from pdf2beamer.llm import FakeGenerator
from pdf2beamer.model_factory import _NEMOTRON_GGUF_DIR
from pdf2beamer.pipeline import GenerationResult
from pdf2beamer.retrieval import FakeEmbedder, FakeReranker


def test_public_api_imports_without_heavy_backends() -> None:
    assert pdf2beamer.PipelineConfig is PipelineConfig
    assert pdf2beamer.PdfToBeamerPipeline is not None
    assert pdf2beamer.BeamerRenderer is BeamerRenderer


def test_default_config_uses_fake_backends() -> None:
    config = PipelineConfig(model_path="models/nemotron-3-nano-4b-gguf/model.gguf")

    assert isinstance(create_generator(config), FakeGenerator)
    assert isinstance(create_embedder(config), FakeEmbedder)
    assert isinstance(create_reranker(config), FakeReranker)
    assert config.model_path == Path("models/nemotron-3-nano-4b-gguf/model.gguf")


def test_real_generator_requires_nemotron_model(monkeypatch) -> None:
    monkeypatch.setattr("pdf2beamer.model_factory._NEMOTRON_GGUF_DIR", Path("missing-nemotron"))
    config = PipelineConfig(use_fake_models=False, generator_backend="llama_cpp")

    with pytest.raises(InvalidModelConfigurationError, match="Nemotron"):
        create_generator(config)

    assert _NEMOTRON_GGUF_DIR.name == "nemotron-3-nano-4b-gguf"


def test_beamer_renderer_writes_main_tex(tmp_path: Path) -> None:
    slide_ir = SlideIR(
        paper_title="Demo Paper",
        audience="technical",
        duration_minutes=5,
        slides=[
            Slide(
                id="slide_01",
                role="title",
                title="Demo Paper",
                main_message="Demo Paper",
                layout="title",
            ),
            Slide(
                id="slide_02",
                role="method",
                title="Method",
                main_message="The method is grounded.",
                layout="bullets",
                bullets=[SlideBullet(text="A compact claim with evidence.", source_ids=["chunk_1"])],
                source_ids=["chunk_1"],
            ),
        ],
    )

    result = BeamerRenderer().render(slide_ir, tmp_path)
    tex = result.tex_path.read_text(encoding="utf-8")

    assert result.tex_path.exists()
    assert "\\documentclass" in tex
    assert "Demo Paper" in tex
    assert "A compact claim with evidence" in tex


def test_cli_generate_passes_minimal_options(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakePipeline:
        def __init__(self, config: PipelineConfig) -> None:
            captured["config"] = config

        def generate(self, pdf_path: Path, output_dir: Path) -> GenerationResult:
            captured["pdf_path"] = pdf_path
            captured["output_dir"] = output_dir
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            tex = output_dir / "main.tex"
            tex.write_text("tex", encoding="utf-8")
            quality = output_dir / "quality_report.json"
            quality.write_text("{}", encoding="utf-8")
            return GenerationResult(
                output_dir=output_dir,
                beamer_tex=tex,
                quality_report_path=quality,
                validation_passed=True,
                compile_success=None,
            )

    monkeypatch.setattr("pdf2beamer.cli.PdfToBeamerPipeline", FakePipeline)
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "relayformer.pdf",
            "--output",
            str(tmp_path / "out"),
            "--duration",
            "7",
            "--max-slides",
            "4",
            "--no-compile",
        ],
    )

    assert result.exit_code == 0
    assert captured["pdf_path"] == Path("relayformer.pdf")
    assert captured["config"].duration_minutes == 7
    assert captured["config"].max_slides == 4
    assert captured["config"].compile_pdf is False
