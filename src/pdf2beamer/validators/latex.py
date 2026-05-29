"""Pre-render LaTeX safety validation for SlideIR text."""

import re

from pdf2beamer.ir import SlideIR
from pdf2beamer.validators.base import BaseValidator, ValidationResult, ValidationSeverity, issue

_RAW_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")


class LatexContentValidator(BaseValidator):
    """Warn about raw LaTeX-like content before deterministic escaping."""

    def validate(self, slide_ir: SlideIR, **kwargs: object) -> ValidationResult:
        issues = []
        for slide in slide_ir.slides:
            fields = [("title", slide.title), ("main_message", slide.main_message)]
            fields.extend(
                (f"bullets[{index}]", bullet.text) for index, bullet in enumerate(slide.bullets)
            )
            for field, text in fields:
                if _RAW_COMMAND_RE.search(text):
                    issues.append(
                        issue(
                            severity=ValidationSeverity.WARNING,
                            code="raw_latex_command",
                            message="Text appears to contain a raw LaTeX command.",
                            slide_id=slide.id,
                            field=field,
                        )
                    )
                if text.count("{") != text.count("}"):
                    issues.append(
                        issue(
                            severity=ValidationSeverity.WARNING,
                            code="unmatched_braces",
                            message="Text contains unmatched braces.",
                            slide_id=slide.id,
                            field=field,
                        )
                    )
                if text.count("$") % 2 == 1:
                    issues.append(
                        issue(
                            severity=ValidationSeverity.WARNING,
                            code="suspicious_dollar_signs",
                            message="Text contains an unmatched dollar sign.",
                            slide_id=slide.id,
                            field=field,
                        )
                    )
        return ValidationResult(passed=True, issues=issues)
