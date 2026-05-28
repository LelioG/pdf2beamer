"""Narrative order validation for SlideIR."""

from pdf2beamer.ir import SlideIR
from pdf2beamer.validators.base import BaseValidator, ValidationResult, ValidationSeverity, issue


class NarrativeValidator(BaseValidator):
    """Validate coarse scientific presentation narrative flow."""

    def validate(self, slide_ir: SlideIR, **kwargs: object) -> ValidationResult:
        issues = []
        roles = [slide.role for slide in slide_ir.slides]
        first = {role: roles.index(role) for role in set(roles)}
        if "title" in first and first["title"] != 0:
            issues.append(
                issue(
                    severity=ValidationSeverity.WARNING,
                    code="title_not_first",
                    message="Title slide is not first.",
                )
            )
        self._order(roles, "problem", "method", issues)
        self._order(roles, "method", "results", issues)
        self._order(roles, "results", "takeaway", issues)
        for role in ("problem", "method", "results"):
            if role not in roles:
                issues.append(
                    issue(
                        severity=ValidationSeverity.WARNING,
                        code=f"missing_{role}_slide",
                        message=f"Deck has no {role} slide.",
                    )
                )
        if "contribution" not in roles:
            issues.append(
                issue(
                    severity=ValidationSeverity.WARNING,
                    code="missing_contribution_slide",
                    message="Deck has no contribution slide.",
                )
            )
        for index in range(1, len(roles)):
            if roles[index] == roles[index - 1]:
                issues.append(
                    issue(
                        severity=ValidationSeverity.WARNING,
                        code="duplicate_consecutive_roles",
                        message=f"Consecutive slides share role {roles[index]!r}.",
                        slide_id=slide_ir.slides[index].id,
                        field="role",
                    )
                )
        return ValidationResult(passed=True, issues=issues)

    def _order(self, roles: list[str], before: str, after: str, issues: list[object]) -> None:
        if before in roles and after in roles and roles.index(before) > roles.index(after):
            issues.append(
                issue(
                    severity=ValidationSeverity.WARNING,
                    code="narrative_order",
                    message=f"{before} slide appears after {after} slide.",
                    field="role",
                )
            )
