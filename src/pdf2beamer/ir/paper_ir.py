"""PaperIR models built by the fusion layer."""

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.ingest.models import BoundingBox


class SourceRef(BaseModel):
    """Traceability from PaperIR elements back to extraction outputs."""

    model_config = ConfigDict(extra="forbid")

    page_index: int | None = Field(default=None, ge=0)
    bbox: BoundingBox | None = None
    block_ids: list[str] = Field(default_factory=list)
    extractor_sources: list[str] = Field(default_factory=list)
    source_item_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class PaperMetadata(BaseModel):
    """Document-level metadata for a fused scientific paper."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    page_count: int = Field(ge=0)
    pdf_path: Path
    metadata: dict[str, str] = Field(default_factory=dict)


class ParagraphIR(BaseModel):
    """Paragraph-level PaperIR element."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1)
    section_id: str | None = None
    page_index: int | None = Field(default=None, ge=0)
    source: SourceRef
    confidence: float = Field(ge=0.0, le=1.0)


class SectionIR(BaseModel):
    """Section-level PaperIR element with owned paragraphs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    level: int = Field(ge=1)
    paragraphs: list[ParagraphIR] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)
    source: SourceRef
    confidence: float = Field(ge=0.0, le=1.0)


class FigureIR(BaseModel):
    """Fused figure or extracted image reference."""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: Path | None = None
    caption: str | None = None
    page_index: int | None = Field(default=None, ge=0)
    bbox: BoundingBox | None = None
    linked_section_id: str | None = None
    source: SourceRef
    confidence: float = Field(ge=0.0, le=1.0)


class TableIR(BaseModel):
    """Fused table reference from structured extraction."""

    model_config = ConfigDict(extra="forbid")

    id: str
    caption: str | None = None
    text: str | None = None
    page_index: int | None = Field(default=None, ge=0)
    linked_section_id: str | None = None
    source: SourceRef
    confidence: float = Field(ge=0.0, le=1.0)


class EquationIR(BaseModel):
    """Equation-like element detected from native PDF text extraction."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1)
    latex: str | None = None
    page_index: int | None = Field(default=None, ge=0)
    bbox: BoundingBox | None = None
    linked_section_id: str | None = None
    source: SourceRef
    confidence: float = Field(ge=0.0, le=1.0)


class PaperIR(BaseModel):
    """Main inspectable intermediate representation of a scientific paper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.2"
    metadata: PaperMetadata
    abstract: str | None = None
    sections: list[SectionIR] = Field(default_factory=list)
    paragraphs: list[ParagraphIR] = Field(default_factory=list)
    figures: list[FigureIR] = Field(default_factory=list)
    tables: list[TableIR] = Field(default_factory=list)
    equations: list[EquationIR] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def save_json(self, path: str | Path) -> None:
        """Serialize the PaperIR to a JSON file."""

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "PaperIR":
        """Load PaperIR from a JSON file."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# Compatibility aliases for the initial scaffolded model names.
Paragraph = ParagraphIR
Section = SectionIR
Figure = FigureIR
Table = TableIR
Equation = EquationIR
