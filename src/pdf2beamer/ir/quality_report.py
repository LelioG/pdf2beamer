"""Quality report model for pipeline outputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceScores(BaseModel):
    """Stage-level confidence estimates."""

    model_config = ConfigDict(extra="forbid")

    extraction: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion: float | None = Field(default=None, ge=0.0, le=1.0)
    argument_graph: float | None = Field(default=None, ge=0.0, le=1.0)
    slide_generation: float | None = Field(default=None, ge=0.0, le=1.0)


class QualityReport(BaseModel):
    """Serializable quality report written after each run."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "partial", "failed"]
    input_type: Literal["native_pdf", "image_only", "unknown"]
    page_count: int = Field(ge=0)
    slide_count: int = Field(ge=0)
    figures_detected: int = Field(default=0, ge=0)
    figures_used: int = Field(default=0, ge=0)
    grounded_claims: int = Field(default=0, ge=0)
    ungrounded_claims: int = Field(default=0, ge=0)
    latex_compilation: Literal["success", "failed", "skipped", "not_run"] = "not_run"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence: ConfidenceScores = Field(default_factory=ConfidenceScores)
