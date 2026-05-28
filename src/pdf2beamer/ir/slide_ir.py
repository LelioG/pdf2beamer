"""Pydantic models for final slide content before deterministic Beamer rendering."""

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

_SUPPORTED_VISUAL_TYPES = {"figure", "table", "equation", "image", "unknown"}


class SlideBullet(BaseModel):
    """A concise slide bullet with source grounding."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SlideVisual(BaseModel):
    """A structured visual reference for a slide."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    type: str = Field(default="unknown", validation_alias=AliasChoices("type", "kind"))
    path: str | None = None
    caption: str | None = None
    content: str | None = None
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> str:
        normalized = _enum_or_str(value).lower().strip()
        return normalized if normalized in _SUPPORTED_VISUAL_TYPES else "unknown"

    @field_validator("path", mode="before")
    @classmethod
    def path_to_string(cls, value: object) -> str | None:
        if value is None:
            return None
        return str(value)


class Slide(BaseModel):
    """One fully structured slide before rendering."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    title: str = Field(min_length=1)
    main_message: str = Field(min_length=1)
    layout: str
    bullets: list[SlideBullet] = Field(default_factory=list)
    visuals: list[SlideVisual] = Field(default_factory=list)
    speaker_notes: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("role", "layout", mode="before")
    @classmethod
    def enum_to_string(cls, value: object) -> str:
        return _enum_or_str(value).lower().strip()


class SlideIR(BaseModel):
    """Final structured deck representation consumed by the Beamer renderer."""

    model_config = ConfigDict(extra="forbid")

    paper_title: str | None = None
    audience: str = "unknown"
    duration_minutes: int = Field(default=0, ge=0)
    slides: list[Slide] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def get_slide(self, slide_id: str) -> Slide | None:
        """Return a slide by id, if present."""

        for slide in self.slides:
            if slide.id == slide_id:
                return slide
        return None

    def get_slides_by_role(self, role: str) -> list[Slide]:
        """Return slides matching a normalized role."""

        normalized = role.lower().strip()
        return [slide for slide in self.slides if slide.role == normalized]

    def save_json(self, path: str | Path) -> None:
        """Serialize SlideIR to JSON."""

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "SlideIR":
        """Load SlideIR from JSON."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _enum_or_str(value: object) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


# Compatibility alias for the initial scaffold.
SlideSource = dict[str, Any]
