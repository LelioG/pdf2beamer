"""Validation modules for generated intermediate representations."""

from pathlib import Path

from pdf2beamer.ir import SlideIR
from pdf2beamer.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from pdf2beamer.validators.density import DensityValidator
from pdf2beamer.validators.figures import FigureValidator
from pdf2beamer.validators.grounding import GroundingValidator
from pdf2beamer.validators.latex import LatexContentValidator
from pdf2beamer.validators.narrative import NarrativeValidator


def validate_slide_ir(
    slide_ir: SlideIR,
    figure_base_dir: str | Path | None = None,
) -> ValidationResult:
    """Run the standard SlideIR validator suite."""

    result = ValidationResult(passed=True, issues=[])
    validators: list[BaseValidator] = [
        GroundingValidator(),
        DensityValidator(),
        FigureValidator(),
        NarrativeValidator(),
        LatexContentValidator(),
    ]
    for validator in validators:
        result = result.merge(validator.validate(slide_ir, figure_base_dir=figure_base_dir))
    return result


__all__ = [
    "BaseValidator",
    "DensityValidator",
    "FigureValidator",
    "GroundingValidator",
    "LatexContentValidator",
    "NarrativeValidator",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSeverity",
    "validate_slide_ir",
]
