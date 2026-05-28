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
            "paper.pdf",
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
    assert captured["pdf_path"] == Path("paper.pdf")
    assert captured["config"].duration_minutes == 7
    assert captured["config"].max_slides == 4
    assert captured["config"].compile_pdf is False
    assert captured["config"].fail_on_error is False


def test_source_has_no_paper_specific_scoring_terms() -> None:
    forbidden = {
        "relayformer",
        "glra",
        "global-local",
        "input unification",
        "vml",
    }
    source_text = "\n".join(path.read_text(encoding="utf-8").lower() for path in Path("src").rglob("*.py"))

    assert not forbidden & set(term for term in forbidden if term in source_text)


def test_pipeline_raises_unexpected_errors_by_default(tmp_path: Path, monkeypatch) -> None:
    from pdf2beamer.pipeline import PdfToBeamerPipeline

    def boom(path):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr("pdf2beamer.pipeline.diagnose_pdf", boom)

    with pytest.raises(RuntimeError, match="unexpected bug"):
        PdfToBeamerPipeline(PipelineConfig()).generate("paper.pdf", tmp_path / "out")


def test_visual_selector_uses_relevant_non_duplicate_figures(tmp_path: Path) -> None:
    from pdf2beamer.fusion.source_map import make_source_ref
    from pdf2beamer.ingest.models import BoundingBox
    from pdf2beamer.ir import FigureIR, PaperIR, PaperMetadata, SectionIR
    from pdf2beamer.planning.visual_selector import attach_relevant_visuals
    from pdf2beamer.retrieval import RerankResult

    figure_a = tmp_path / "method_a.png"
    figure_b = tmp_path / "method_b_duplicate.png"
    figure_c = tmp_path / "results.png"
    figure_a.write_bytes(b"same image bytes")
    figure_b.write_bytes(b"same image bytes")
    figure_c.write_bytes(b"different result image")
    source = make_source_ref(page_index=1, confidence=0.8)
    paper = PaperIR(
        metadata=PaperMetadata(title="Paper", authors=[], page_count=4, pdf_path=Path("paper.pdf")),
        sections=[
            SectionIR(
                id="sec_method",
                title="Method",
                level=1,
                source=source,
                confidence=0.8,
            ),
            SectionIR(
                id="sec_results",
                title="Results",
                level=1,
                source=make_source_ref(page_index=3, confidence=0.8),
                confidence=0.8,
            ),
        ],
        figures=[
            FigureIR(
                id="fig_method_a",
                path=figure_a,
                caption="Method architecture overview and model pipeline",
                page_index=1,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),
                linked_section_id="sec_method",
                source=source,
                confidence=0.9,
            ),
            FigureIR(
                id="fig_method_b",
                path=figure_b,
                caption="Method architecture overview and model pipeline",
                page_index=1,
                linked_section_id="sec_method",
                source=source,
                confidence=0.9,
            ),
            FigureIR(
                id="fig_results",
                path=figure_c,
                caption="Results comparison across benchmark datasets",
                page_index=3,
                linked_section_id="sec_results",
                source=make_source_ref(page_index=3, confidence=0.8),
                confidence=0.9,
            ),
        ],
    )
    slides = SlideIR(
        slides=[
            Slide(
                id="slide_01",
                role="method",
                title="Method Overview",
                main_message="The model pipeline is structured.",
                layout="bullets",
                bullets=[SlideBullet(text="The architecture uses a clear method pipeline.")],
            ),
            Slide(
                id="slide_02",
                role="results",
                title="Results",
                main_message="The benchmark comparison improves performance.",
                layout="bullets",
                bullets=[SlideBullet(text="Results compare performance across benchmarks.")],
            ),
        ],
    )
    contexts = [
        RerankResult(
            chunk_id="c_method",
            text="method architecture model pipeline",
            retrieval_score=1.0,
            rerank_score=1.0,
            combined_score=1.0,
            source_pages=[1],
            section_id="sec_method",
            chunk_type="paragraph",
        ),
        RerankResult(
            chunk_id="c_results",
            text="results benchmark comparison performance",
            retrieval_score=1.0,
            rerank_score=1.0,
            combined_score=1.0,
            source_pages=[3],
            section_id="sec_results",
            chunk_type="paragraph",
        ),
    ]

    selected = attach_relevant_visuals(slides, paper, contexts)

    assert selected.slides[0].visuals[0].id == "fig_method_a"
    assert selected.slides[1].visuals[0].id == "fig_results"
    assert selected.slides[0].visuals[0].id != selected.slides[1].visuals[0].id

def test_fallback_evidence_points_are_not_cut_mid_clause() -> None:
    from pdf2beamer.planning.slide_generator import _evidence_points

    points = _evidence_points(
        "This design allows the model to adapt seamlessly to arbitrary input resolutionsand "
        "process visual evidence efficiently across benchmarks."
    )

    assert points
    assert all(len(point) <= 90 for point in points)
    assert not any(point.endswith(("more", "only", "not", "across", "and")) for point in points)
    assert not any("resolutionsand" in point for point in points)

def test_polish_slide_removes_orphan_table_and_figure_references() -> None:
    from pdf2beamer.ir import ArgumentGraph
    from pdf2beamer.planning.slide_generator import _polish_slide

    slide = Slide(
        id="slide_01",
        role="results",
        title="Results",
        main_message="Results",
        layout="bullets",
        bullets=[
            SlideBullet(text="Effect of Input Resolution As shown in Table 2"),
            SlideBullet(text="The corresponding results are presented in Table 1"),
            SlideBullet(text="Figure 3 shows qualitative examples from the evaluation"),
            SlideBullet(text="Operating at raw resolution yields much better results"),
        ],
    )

    polished = _polish_slide(slide, ArgumentGraph(paper_title="Paper"))
    texts = [bullet.text for bullet in polished.bullets]

    assert "Effect of Input Resolution" in texts
    assert "Operating at raw resolution yields much better results" in texts
    assert not any("Table 1" in text or "Figure 3" in text for text in texts)

