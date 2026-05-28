"""Visual and figure validation for SlideIR."""

from pathlib import Path

from pdf2beamer.ir import SlideIR
from pdf2beamer.validators.base import BaseValidator, ValidationResult, ValidationSeverity, issue

_SUPPORTED_VISUAL_TYPES = {"figure", "table", "equation", "image", "unknown"}
_VISUAL_LAYOUTS = {"figure_left_bullets_right", "figure_top_bullets_bottom", "table", "equation"}


class FigureValidator(BaseValidator):
    """Validate visual references in SlideIR."""

    def validate(self, slide_ir: SlideIR, **kwargs: object) -> ValidationResult:
        base_dir = kwargs.get("figure_base_dir")
        base_path = Path(base_dir) if base_dir is not None else None
        issues = []
        for slide in slide_ir.slides:
            if slide.layout in _VISUAL_LAYOUTS and not slide.visuals:
                issues.append(issue(
                    severity=ValidationSeverity.WARNING,
                    code="layout_expects_visual",
                    message="Slide layout expects at least one visual.",
                    slide_id=slide.id,
                    field="visuals",
                ))
            for index, visual in enumerate(slide.visuals):
                if visual.type not in _SUPPORTED_VISUAL_TYPES:
                    issues.append(issue(
                        severity=ValidationSeverity.WARNING,
                        code="unsupported_visual_type",
                        message=f"Unsupported visual type: {visual.type}.",
                        slide_id=slide.id,
                        field=f"visuals[{index}].type",
                    ))
                if visual.path is not None:
                    path = Path(visual.path)
                    candidate = _resolve_visual_candidate(path, base_path)
                    if not candidate.exists():
                        issues.append(issue(
                            severity=ValidationSeverity.WARNING,
                            code="visual_path_missing",
                            message=f"Visual path does not exist: {visual.path}.",
                            slide_id=slide.id,
                            field=f"visuals[{index}].path",
                        ))
                if visual.type in {"figure", "image"} and not visual.caption:
                    issues.append(issue(
                        severity=ValidationSeverity.INFO,
                        code="visual_missing_caption",
                        message="Figure or image visual has no caption.",
                        slide_id=slide.id,
                        field=f"visuals[{index}].caption",
                    ))
                if visual.type in {"table", "equation"} and not visual.content:
                    issues.append(issue(
                        severity=ValidationSeverity.WARNING,
                        code="visual_missing_content",
                        message=f"{visual.type.title()} visual has no structured content.",
                        slide_id=slide.id,
                        field=f"visuals[{index}].content",
                    ))
        return ValidationResult(passed=True, issues=issues)


def _resolve_visual_candidate(path: Path, base_path: Path | None) -> Path:
    if path.is_absolute() or base_path is None:
        return path
    candidate = base_path / path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return candidate
