"""Grounding validation for SlideIR."""

import re

from pdf2beamer.ir import SlideIR
from pdf2beamer.validators.base import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    issue,
)

_NUMERIC_OR_CLAIM_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*%?|\b(improves|outperforms|achieves|reduces|increases)\b)",
    re.IGNORECASE,
)
_EXEMPT_ROLES = {"title", "takeaway", "appendix"}


class GroundingValidator(BaseValidator):
    """Check whether scientific slide claims carry source ids."""

    def validate(self, slide_ir: SlideIR, **kwargs: object) -> ValidationResult:
        issues = []
        for slide in slide_ir.slides:
            if slide.role in _EXEMPT_ROLES:
                continue
            if slide.main_message and not slide.source_ids:
                issues.append(
                    issue(
                        severity=ValidationSeverity.WARNING,
                        code="main_message_without_sources",
                        message="Slide main_message has no source ids.",
                        slide_id=slide.id,
                        field="main_message",
                        suggestion="Add source_ids grounded in ArgumentGraph nodes or chunks.",
                    ),
                )
            for index, bullet in enumerate(slide.bullets):
                if not bullet.source_ids:
                    issues.append(
                        issue(
                            severity=ValidationSeverity.WARNING,
                            code="bullet_without_sources",
                            message="Slide bullet has no source ids.",
                            slide_id=slide.id,
                            field=f"bullets[{index}]",
                        ),
                    )
                if _NUMERIC_OR_CLAIM_RE.search(bullet.text) and not bullet.source_ids:
                    issues.append(
                        issue(
                            severity=ValidationSeverity.WARNING,
                            code="numerical_or_comparative_claim_without_sources",
                            message="Numerical or comparative claim has no source ids.",
                            slide_id=slide.id,
                            field=f"bullets[{index}]",
                        ),
                    )
        return ValidationResult(passed=True, issues=issues)
