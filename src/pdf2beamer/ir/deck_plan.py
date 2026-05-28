"""Pydantic models for planned presentation structure."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pdf2beamer._compat import StrEnum


class SlideRole(StrEnum):
    """Narrative role of a slide."""

    TITLE = "title"
    PROBLEM = "problem"
    GAP = "gap"
    CONTRIBUTION = "contribution"
    INTUITION = "intuition"
    METHOD = "method"
    ARCHITECTURE = "architecture"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    LIMITATIONS = "limitations"
    TAKEAWAY = "takeaway"
    APPENDIX = "appendix"


class SlideLayout(StrEnum):
    """Renderer-supported slide layouts."""

    TITLE = "title"
    BULLETS = "bullets"
    TWO_COLUMNS = "two_columns"
    FIGURE_LEFT_BULLETS_RIGHT = "figure_left_bullets_right"
    FIGURE_TOP_BULLETS_BOTTOM = "figure_top_bullets_bottom"
    TABLE = "table"
    EQUATION = "equation"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"


class PlannedSlide(BaseModel):
    """One planned slide before final content generation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: SlideRole
    goal: str = Field(min_length=1)
    target_evidence: list[str] = Field(default_factory=list)
    suggested_layout: SlideLayout = SlideLayout.BULLETS
    expected_content_type: str = Field(default="bullets", min_length=1)


class DeckPlan(BaseModel):
    """Narrative plan for the final Beamer deck."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1"
    duration_minutes: int = Field(ge=1)
    audience: str = Field(min_length=1)
    slide_count: int = Field(ge=0)
    slides: list[PlannedSlide] = Field(default_factory=list)

    @model_validator(mode="after")
    def slide_count_matches_list_when_nonzero(self) -> "DeckPlan":
        if self.slides and self.slide_count != len(self.slides):
            raise ValueError("slide_count must match the number of planned slides")
        return self
