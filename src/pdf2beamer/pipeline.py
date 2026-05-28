"""Top-level orchestration API for pdf2beamer."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.beamer import BeamerRenderer, LatexCompileResult, compile_latex, get_theme
from pdf2beamer.config import PipelineConfig
from pdf2beamer.diagnostics import PDFDiagnostics, diagnose_pdf
from pdf2beamer.fusion import build_paper_ir
from pdf2beamer.ingest import extract_with_docling, extract_with_pymupdf
from pdf2beamer.ir import PaperIR, SlideIR
from pdf2beamer.model_factory import create_embedder, create_generator, create_reranker
from pdf2beamer.planning import build_argument_graph, build_deck_plan, generate_slide_ir
from pdf2beamer.quality import QualityReport, build_quality_report
from pdf2beamer.retrieval import (
    BaseEmbedder,
    BaseReranker,
    RerankResult,
    build_embedding_index,
    chunk_paper_ir,
    is_informative_chunk_text,
    retrieve_and_rerank,
)
from pdf2beamer.validators import ValidationResult, validate_slide_ir

_CONTEXT_QUERIES = [
    "main problem of the paper",
    "main contribution of the paper",
    "proposed method",
    "experimental setup",
    "key results",
    "limitations",
    "conclusion and takeaway",
]


class GenerationResult(BaseModel):
    """Paths and status flags for one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    slides_pdf: Path | None = None
    beamer_tex: Path
    paper_ir_path: Path | None = None
    argument_graph_path: Path | None = None
    deck_plan_path: Path | None = None
    slide_ir_path: Path | None = None
    quality_report_path: Path | None = None
    validation_passed: bool = False
    compile_success: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    assets_dir: Path | None = None
    diagnostics: PDFDiagnostics | None = None


class PdfToBeamerPipeline:
    """Local-first PDF-to-Beamer pipeline facade."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def generate(self, pdf_path: str | Path, output_dir: str | Path) -> GenerationResult:
        """Generate a Beamer project from a supported native PDF."""

        input_path = Path(pdf_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = out_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        paths = _artifact_paths(out_dir)
        warnings: list[str] = []
        errors: list[str] = []
        diagnostics: PDFDiagnostics | None = None
        paper_ir: PaperIR | None = None
        slide_ir: SlideIR | None = None
        validation_result: ValidationResult | None = None
        compile_result: LatexCompileResult | None = None
        tex_path = paths["main_tex"]

        try:
            diagnostics = diagnose_pdf(input_path)
            warnings.extend(diagnostics.warnings)
            if not diagnostics.is_supported_native_pdf:
                reason = diagnostics.rejection_reason or "Unsupported PDF."
                errors.append(reason)
                quality = build_quality_report(None, None, None, None, warnings, errors)
                quality.input_type = "native_pdf" if diagnostics.has_text_layer else "unknown"
                quality.page_count = diagnostics.page_count
                _save_quality(paths["quality_report"], quality)
                return _result(
                    out_dir, tex_path, paths, assets_dir, diagnostics,
                    False, False, warnings, errors,
                )

            pymupdf_extraction = extract_with_pymupdf(
                input_path,
                assets_dir=assets_dir,
                extract_images=self.config.extract_images,
            )
            docling_extraction = extract_with_docling(input_path)
            paper_ir = build_paper_ir(docling_extraction, pymupdf_extraction)
            _save_optional(paths["paper_ir"], paper_ir, self.config.save_intermediate)

            chunks = chunk_paper_ir(paper_ir)
            _save_json_payload(paths["chunks"], [chunk.model_dump() for chunk in chunks])
            embedder = create_embedder(self.config)
            reranker = create_reranker(self.config)
            embedding_index = build_embedding_index(chunks, embedder)
            _save_optional(paths["embedding_index"], embedding_index, self.config.save_intermediate)
            contexts = _retrieve_contexts(
                embedding_index,
                embedder,
                reranker,
                retrieval_top_k=self.config.retrieval_top_k,
                rerank_top_k=self.config.rerank_top_k,
                max_context_chars=self.config.max_context_chars,
            )

            generator = create_generator(self.config)
            argument_graph = build_argument_graph(paper_ir.metadata.title, contexts, generator)
            _save_optional(paths["argument_graph"], argument_graph, self.config.save_intermediate)

            deck_plan = build_deck_plan(
                argument_graph,
                duration_minutes=self.config.duration_minutes,
                audience=self.config.audience,
                max_slides=self.config.max_slides,
                contexts=contexts,
            )
            _save_optional(paths["deck_plan"], deck_plan, self.config.save_intermediate)

            slide_ir = generate_slide_ir(
                deck_plan,
                argument_graph,
                generator,
                contexts=contexts,
                paper_ir=paper_ir,
            )
            _save_optional(paths["slide_ir"], slide_ir, self.config.save_intermediate)

            validation_result = validate_slide_ir(slide_ir, figure_base_dir=out_dir)
            _save_optional(
                paths["validation_report"], validation_result, self.config.save_intermediate
            )
            warnings.extend(
                issue.message for issue in validation_result.issues if issue.severity != "error"
            )
            errors.extend(
                issue.message for issue in validation_result.issues if issue.severity == "error"
            )

            renderer = BeamerRenderer(theme=get_theme(self.config.theme))
            render_result = renderer.render(slide_ir, out_dir)
            tex_path = render_result.tex_path
            warnings.extend(render_result.warnings)

            if self.config.compile_pdf:
                compile_result = compile_latex(tex_path, engine=self.config.latex_engine)
                warnings.extend(compile_result.warnings)
                if not compile_result.success:
                    errors.extend(compile_result.errors)
            else:
                compile_result = None

            quality = build_quality_report(
                paper_ir,
                slide_ir,
                validation_result,
                compile_result,
                warnings=warnings,
                errors=errors,
            )
            _save_quality(paths["quality_report"], quality)
            return _result(
                out_dir,
                tex_path,
                paths,
                assets_dir,
                diagnostics,
                validation_result.passed,
                None if compile_result is None else compile_result.success,
                warnings,
                errors,
                slides_pdf=compile_result.pdf_path if compile_result else None,
            )
        except Exception as exc:
            if self.config.fail_on_error:
                raise
            message = str(exc) if self.config.debug else _short_error(exc)
            errors.append(message)
            quality = build_quality_report(
                paper_ir, slide_ir, validation_result, compile_result, warnings, errors
            )
            _save_quality(paths["quality_report"], quality)
            return _result(
                out_dir, tex_path, paths, assets_dir, diagnostics,
                False, False, warnings, errors,
            )


def _retrieve_contexts(
    index: Any,
    embedder: BaseEmbedder,
    reranker: BaseReranker,
    retrieval_top_k: int = 8,
    rerank_top_k: int = 3,
    max_context_chars: int = 1200,
) -> list[RerankResult]:
    seen: set[str] = set()
    contexts: list[RerankResult] = []
    for query in _CONTEXT_QUERIES:
        results = retrieve_and_rerank(
            query,
            index,
            embedder,
            reranker,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
        )
        for result in results:
            if result.chunk_id in seen or not is_informative_chunk_text(
                result.text,
                section_title=result.section_title,
            ):
                continue
            seen.add(result.chunk_id)
            metadata = dict(result.metadata)
            metadata["retrieval_query"] = query
            contexts.append(
                _trim_context(result.model_copy(update={"metadata": metadata}), max_context_chars)
            )

    contexts.extend(_seed_contexts_from_index(index, seen, max_context_chars))
    return contexts


def _seed_contexts_from_index(
    index: Any,
    seen: set[str],
    max_context_chars: int = 1200,
) -> list[RerankResult]:
    records = list(getattr(index, "records", []) or [])
    seeds: list[RerankResult] = []
    for role, query in (
        ("problem", "main problem of the paper"),
        ("contribution", "main contribution of the paper"),
        ("method", "proposed method"),
        ("result", "key results"),
        ("takeaway", "conclusion and takeaway"),
    ):
        candidates = [
            record
            for record in records
            if record.chunk_id not in seen
            and is_informative_chunk_text(record.text, section_title=record.section_title)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda record: _seed_context_score(record, role), reverse=True)
        best = candidates[0]
        if _seed_context_score(best, role) <= 0.0:
            continue
        seen.add(best.chunk_id)
        metadata = dict(best.metadata)
        metadata["retrieval_query"] = query
        metadata["context_seed"] = role
        seeds.append(
            _trim_context(
                RerankResult(
                    chunk_id=best.chunk_id,
                    text=best.text,
                    retrieval_score=0.0,
                    rerank_score=0.0,
                    combined_score=0.65,
                    metadata=metadata,
                    source_pages=best.source_pages,
                    section_id=best.section_id,
                    section_title=best.section_title,
                    chunk_type=best.chunk_type,
                ),
                max_context_chars,
            ),
        )
    return seeds


def _trim_context(context: RerankResult, max_chars: int) -> RerankResult:
    text = " ".join(context.text.split())
    if len(text) <= max_chars:
        return context.model_copy(update={"text": text})

    window = text[:max_chars].rstrip()
    for separator in (". ", "; ", ", "):
        pos = window.rfind(separator)
        if pos >= max(120, int(max_chars * 0.65)):
            return context.model_copy(update={"text": window[: pos + 1].strip()})
    return context.model_copy(update={"text": window.strip()})


def _seed_context_score(record: Any, role: str) -> float:
    text = str(record.text).lower()
    section = str(record.section_title or "").lower()
    score = 0.0
    if "abstract" in section:
        if role == "contribution":
            score += 3.5
        elif role in {"problem", "takeaway"}:
            score += 1.5
        else:
            score += 0.5
    if "introduction" in section:
        if role == "problem":
            score += 2.8
        elif role in {"contribution", "takeaway"}:
            score += 2.0
        else:
            score += 0.5
    if any(term in section for term in ("method", "approach", "algorithm", "model", "architecture")):
        score += 3.0 if role == "method" else 0.2
    if any(term in section for term in ("experiment", "result", "evaluation", "ablation", "benchmark")):
        score += 3.0 if role == "result" else -0.2
    if "conclusion" in section:
        score += 4.0 if role == "takeaway" else 0.2

    role_terms = {
        "problem": ("challenge", "problem", "aims", "demands", "task", "goal"),
        "contribution": ("propose", "contribution", "framework", "approach"),
        "method": ("method", "architecture", "algorithm", "model", "implementation"),
        "result": ("outperform", "improve", "achieve", "results", "benchmarks"),
        "takeaway": ("overall", "demonstrate", "effective", "scalable", "conclusion"),
    }
    score += sum(0.4 for term in role_terms.get(role, ()) if term in text)
    if section.startswith("a.") and role == "result":
        score -= 1.2
    if any(term in section for term in ("references", "bibliography")):
        score -= 10.0
    return score


def _artifact_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "paper_ir": out_dir / "paper_ir.json",
        "argument_graph": out_dir / "argument_graph.json",
        "deck_plan": out_dir / "deck_plan.json",
        "slide_ir": out_dir / "slide_ir.json",
        "quality_report": out_dir / "quality_report.json",
        "main_tex": out_dir / "main.tex",
        "chunks": out_dir / "chunks.json",
        "embedding_index": out_dir / "embedding_index.json",
        "validation_report": out_dir / "validation_report.json",
    }


def _save_optional(path: Path, model: Any, enabled: bool) -> None:
    if not enabled:
        return
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _save_json_payload(path: Path, payload: Any) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _save_quality(path: Path, quality: QualityReport) -> None:
    quality.save_json(path)


def _result(
    out_dir: Path,
    tex_path: Path,
    paths: dict[str, Path],
    assets_dir: Path,
    diagnostics: PDFDiagnostics | None,
    validation_passed: bool,
    compile_success: bool | None,
    warnings: list[str],
    errors: list[str],
    slides_pdf: Path | None = None,
) -> GenerationResult:
    return GenerationResult(
        output_dir=out_dir,
        slides_pdf=slides_pdf,
        beamer_tex=tex_path,
        paper_ir_path=paths["paper_ir"] if paths["paper_ir"].exists() else None,
        argument_graph_path=paths["argument_graph"] if paths["argument_graph"].exists() else None,
        deck_plan_path=paths["deck_plan"] if paths["deck_plan"].exists() else None,
        slide_ir_path=paths["slide_ir"] if paths["slide_ir"].exists() else None,
        quality_report_path=paths["quality_report"] if paths["quality_report"].exists() else None,
        validation_passed=validation_passed,
        compile_success=compile_success,
        warnings=warnings,
        errors=errors,
        assets_dir=assets_dir,
        diagnostics=diagnostics,
    )


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
