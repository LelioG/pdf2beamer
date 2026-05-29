"""Shared validation result models and runner primitives."""

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.ir import SlideIR


class ValidationSeverity:
    """Validation severity constants."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    """One structured validation issue."""

    model_config = ConfigDict(extra="forbid")

    severity: str
    code: str
    message: str
    slide_id: str | None = None
    field: str | None = None
    suggestion: str | None = None


class ValidationResult(BaseModel):
    """Result of running one or more SlideIR validators."""

    model_config = ConfigDict(extra="forbid")

    passed: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)

    def error_count(self) -> int:
        """Return the number of error-level issues."""

        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR)

    def warning_count(self) -> int:
        """Return the number of warning-level issues."""

        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge two validation results."""

        issues = [*self.issues, *other.issues]
        return ValidationResult(
            passed=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
        )


class BaseValidator(ABC):
    """Abstract base class for SlideIR validators."""

    @abstractmethod
    def validate(self, slide_ir: SlideIR, **kwargs: object) -> ValidationResult:
        """Validate a SlideIR instance."""


def issue(
    *,
    severity: str,
    code: str,
    message: str,
    slide_id: str | None = None,
    field: str | None = None,
    suggestion: str | None = None,
) -> ValidationIssue:
    """Convenience helper for creating validation issues."""

    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        slide_id=slide_id,
        field=field,
        suggestion=suggestion,
    )


def empty_result() -> ValidationResult:
    """Return a passing validation result."""

    return ValidationResult(passed=True, issues=[])


def figure_base_path(value: str | Path | None) -> Path | None:
    """Normalize optional base path values used by validators."""

    return Path(value) if value is not None else None
