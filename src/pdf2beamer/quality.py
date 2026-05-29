"""Quality report construction for pipeline runs."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.beamer import LatexCompileResult
from pdf2beamer.ir import PaperIR, SlideIR
from pdf2beamer.validators import ValidationResult


class QualityReport(BaseModel):
    """Serializable quality report for one generation run."""

    model_config = ConfigDict(extra="forbid")

    status: str
    input_type: str
    page_count: int
    slide_count: int
    figures_detected: int = 0
    figures_used: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    latex_compilation: str = "not_run"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)

    def save_json(self, path: str | Path) -> None:
        """Save the report to JSON."""

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


def build_quality_report(
    paper_ir: PaperIR | None,
    slide_ir: SlideIR | None,
    validation_result: ValidationResult | None,
    compile_result: LatexCompileResult | None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> QualityReport:
    """Build an approximate V1 quality report."""

    all_warnings = list(warnings or [])
    all_errors = list(errors or [])
    if validation_result is not None:
        all_warnings.extend(
            issue.message for issue in validation_result.issues if issue.severity == "warning"
        )
        all_errors.extend(
            issue.message for issue in validation_result.issues if issue.severity == "error"
        )
    if compile_result is not None:
        all_warnings.extend(compile_result.warnings)
        all_errors.extend(compile_result.errors)
    slide_count = len(slide_ir.slides) if slide_ir is not None else 0
    figures_detected = len(paper_ir.figures) if paper_ir is not None else 0
    figures_used = 0
    grounded = 0
    ungrounded = 0
    if slide_ir is not None:
        for slide in slide_ir.slides:
            for bullet in slide.bullets:
                if bullet.source_ids:
                    grounded += 1
                else:
                    ungrounded += 1
            figures_used += sum(1 for visual in slide.visuals if visual.type in {"figure", "image"})
    latex_status = "not_run"
    if compile_result is not None:
        latex_status = "success" if compile_result.success else "failed"
    status = "success" if not all_errors else "failed"
    return QualityReport(
        status=status,
        input_type="native_pdf" if paper_ir is not None else "unknown",
        page_count=paper_ir.metadata.page_count if paper_ir is not None else 0,
        slide_count=slide_count,
        figures_detected=figures_detected,
        figures_used=figures_used,
        grounded_claims=grounded,
        ungrounded_claims=ungrounded,
        latex_compilation=latex_status,
        warnings=all_warnings,
        errors=all_errors,
        confidence={"pipeline": 0.5 if all_errors else 0.8},
    )
