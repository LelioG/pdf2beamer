"""Slide density validation."""

from pdf2beamer.ir import SlideIR
from pdf2beamer.validators.base import BaseValidator, ValidationResult, ValidationSeverity, issue


class DensityValidator(BaseValidator):
    """Warn when slides are too dense for presentation use."""

    def __init__(
        self,
        max_bullets: int = 6,
        max_words_per_bullet: int = 14,
        max_chars_per_bullet: int = 90,
        max_title_chars: int = 80,
        max_main_message_chars: int = 180,
    ) -> None:
        self.max_bullets = max_bullets
        self.max_words_per_bullet = max_words_per_bullet
        self.max_chars_per_bullet = max_chars_per_bullet
        self.max_title_chars = max_title_chars
        self.max_main_message_chars = max_main_message_chars

    def validate(self, slide_ir: SlideIR, **kwargs: object) -> ValidationResult:
        issues = []
        for slide in slide_ir.slides:
            if len(slide.bullets) > self.max_bullets:
                issues.append(issue(
                    severity=ValidationSeverity.WARNING,
                    code="too_many_bullets",
                    message=f"Slide has more than {self.max_bullets} bullets.",
                    slide_id=slide.id,
                    field="bullets",
                ))
            if len(slide.title) > self.max_title_chars:
                issues.append(issue(
                    severity=ValidationSeverity.WARNING,
                    code="title_too_long",
                    message="Slide title is long.",
                    slide_id=slide.id,
                    field="title",
                ))
            if len(slide.main_message) > self.max_main_message_chars:
                issues.append(issue(
                    severity=ValidationSeverity.WARNING,
                    code="main_message_too_long",
                    message="Slide main message is long.",
                    slide_id=slide.id,
                    field="main_message",
                ))
            for index, bullet in enumerate(slide.bullets):
                if len(bullet.text.split()) > self.max_words_per_bullet:
                    issues.append(issue(
                        severity=ValidationSeverity.WARNING,
                        code="bullet_too_wordy",
                        message="Slide bullet has too many words.",
                        slide_id=slide.id,
                        field=f"bullets[{index}]",
                    ))
                if len(bullet.text) > self.max_chars_per_bullet:
                    issues.append(issue(
                        severity=ValidationSeverity.WARNING,
                        code="bullet_too_long",
                        message="Slide bullet is too long.",
                        slide_id=slide.id,
                        field=f"bullets[{index}]",
                    ))
        return ValidationResult(passed=True, issues=issues)
