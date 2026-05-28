"""Pydantic models for low-level native PDF extraction."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field


class BoundingBox(BaseModel):
    """A rectangular region in PDF page coordinates."""

    model_config = ConfigDict(extra="ignore")

    x0: float
    y0: float
    x1: float
    y1: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def width(self) -> float:
        """Bounding box width."""

        return max(0.0, self.x1 - self.x0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def height(self) -> float:
        """Bounding box height."""

        return max(0.0, self.y1 - self.y0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area(self) -> float:
        """Bounding box area."""

        return self.width * self.height


class TextSpan(BaseModel):
    """A text span reported by PyMuPDF structured extraction."""

    model_config = ConfigDict(extra="forbid")

    text: str
    bbox: BoundingBox | None = None
    font: str | None = None
    size: float | None = None
    flags: int | None = None


class TextBlock(BaseModel):
    """A text block on a PDF page."""

    model_config = ConfigDict(extra="forbid")

    id: str
    page_index: int = Field(ge=0)
    block_index: int = Field(ge=0)
    text: str
    bbox: BoundingBox
    spans: list[TextSpan] = Field(default_factory=list)
    char_count: int = Field(ge=0)


class ImageBlock(BaseModel):
    """Image metadata and optional extracted image asset path."""

    model_config = ConfigDict(extra="forbid")

    id: str
    page_index: int = Field(ge=0)
    image_index: int = Field(ge=0)
    xref: int | None = None
    bbox: BoundingBox | None = None
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    extension: str | None = None
    output_path: Path | None = None


class PageExtraction(BaseModel):
    """Low-level extraction for one PDF page."""

    model_config = ConfigDict(extra="forbid")

    page_index: int = Field(ge=0)
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)
    rotation: int
    text: str
    char_count: int = Field(ge=0)
    text_blocks: list[TextBlock] = Field(default_factory=list)
    images: list[ImageBlock] = Field(default_factory=list)


class PyMuPDFExtraction(BaseModel):
    """Complete low-level PyMuPDF extraction result for a PDF."""

    model_config = ConfigDict(extra="forbid")

    pdf_path: Path
    page_count: int = Field(ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)
    pages: list[PageExtraction] = Field(default_factory=list)
    assets_dir: Path | None = None
    warnings: list[str] = Field(default_factory=list)
