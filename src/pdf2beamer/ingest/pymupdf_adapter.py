"""PyMuPDF low-level extraction adapter.

This module extracts native PDF structure only. It does not perform OCR, use
Docling, call model APIs, or attempt scientific-paper understanding.
"""

from pathlib import Path
from typing import Any

from pdf2beamer.ingest.models import (
    BoundingBox,
    ImageBlock,
    PageExtraction,
    PyMuPDFExtraction,
    TextBlock,
    TextSpan,
)


class PyMuPDFExtractionError(ValueError):
    """Raised when low-level PyMuPDF extraction cannot proceed."""


class PyMuPDFAdapter:
    """Small object-oriented wrapper around :func:`extract_with_pymupdf`."""

    def __init__(self, extract_images: bool = True) -> None:
        self.extract_images = extract_images

    def extract(
        self,
        pdf_path: str | Path,
        assets_dir: str | Path | None = None,
    ) -> PyMuPDFExtraction:
        """Extract a native PDF with PyMuPDF."""

        return extract_with_pymupdf(
            pdf_path=pdf_path,
            assets_dir=assets_dir,
            extract_images=self.extract_images,
        )


def extract_with_pymupdf(
    pdf_path: str | Path,
    assets_dir: str | Path | None = None,
    extract_images: bool = True,
) -> PyMuPDFExtraction:
    """Extract page text, blocks, spans, and image metadata from a native PDF.

    Invalid inputs raise :class:`PyMuPDFExtractionError` with a concise message.
    PyMuPDF is imported lazily so package imports do not require PDF extras.
    """

    path = _validate_pdf_path(pdf_path)
    output_assets_dir = Path(assets_dir) if assets_dir is not None else None
    if extract_images and output_assets_dir is not None:
        output_assets_dir.mkdir(parents=True, exist_ok=True)

    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise PyMuPDFExtractionError(f"PyMuPDF is not available: {exc}") from exc

    warnings: list[str] = []

    try:
        document = fitz.open(path)
    except Exception as exc:
        raise PyMuPDFExtractionError(f"Failed to open PDF with PyMuPDF: {exc}") from exc

    try:
        pages = [
            _extract_page(
                document=document,
                page=page,
                page_index=page_index,
                assets_dir=output_assets_dir,
                extract_images=extract_images,
                warnings=warnings,
            )
            for page_index, page in enumerate(document)
        ]
        metadata = _normalize_metadata(document.metadata or {})
    finally:
        document.close()

    return PyMuPDFExtraction(
        pdf_path=path,
        page_count=len(pages),
        metadata=metadata,
        pages=pages,
        assets_dir=output_assets_dir,
        warnings=warnings,
    )


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise PyMuPDFExtractionError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise PyMuPDFExtractionError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise PyMuPDFExtractionError(f"File is not a PDF: {path}")
    return path


def _extract_page(
    *,
    document: Any,
    page: Any,
    page_index: int,
    assets_dir: Path | None,
    extract_images: bool,
    warnings: list[str],
) -> PageExtraction:
    rect = page.rect
    page_text = page.get_text("text") or ""
    text_dict = page.get_text("dict") or {}
    text_blocks = _extract_text_blocks(text_dict=text_dict, page_index=page_index)
    images = _extract_images(
        document=document,
        page=page,
        page_index=page_index,
        assets_dir=assets_dir,
        extract_images=extract_images,
        warnings=warnings,
    )

    return PageExtraction(
        page_index=page_index,
        width=float(rect.width),
        height=float(rect.height),
        rotation=int(page.rotation),
        text=page_text,
        char_count=len(page_text.strip()),
        text_blocks=text_blocks,
        images=images,
    )


def _extract_text_blocks(*, text_dict: dict[str, Any], page_index: int) -> list[TextBlock]:
    text_blocks: list[TextBlock] = []
    for raw_block in text_dict.get("blocks", []):
        if raw_block.get("type") != 0:
            continue

        spans: list[TextSpan] = []
        line_texts: list[str] = []
        for line in raw_block.get("lines", []):
            span_texts: list[str] = []
            for raw_span in line.get("spans", []):
                text = str(raw_span.get("text", ""))
                span_texts.append(text)
                spans.append(
                    TextSpan(
                        text=text,
                        bbox=_bbox_from_sequence(raw_span.get("bbox")),
                        font=_optional_str(raw_span.get("font")),
                        size=_optional_float(raw_span.get("size")),
                        flags=_optional_int(raw_span.get("flags")),
                    ),
                )
            line_text = "".join(span_texts).strip()
            if line_text:
                line_texts.append(line_text)

        block_text = "\n".join(line_texts).strip()
        if not block_text:
            continue

        block_index = len(text_blocks)
        text_blocks.append(
            TextBlock(
                id=f"p{page_index}_b{block_index}",
                page_index=page_index,
                block_index=block_index,
                text=block_text,
                bbox=_bbox_from_sequence(raw_block.get("bbox"))
                or BoundingBox(
                    x0=0.0,
                    y0=0.0,
                    x1=0.0,
                    y1=0.0,
                ),
                spans=spans,
                char_count=len(block_text),
            ),
        )
    return text_blocks


def _extract_images(
    *,
    document: Any,
    page: Any,
    page_index: int,
    assets_dir: Path | None,
    extract_images: bool,
    warnings: list[str],
) -> list[ImageBlock]:
    images: list[ImageBlock] = []
    for image_index, image_info in enumerate(page.get_images(full=True)):
        xref = _optional_int(image_info[0]) if image_info else None
        width = _optional_int(image_info[2]) if len(image_info) > 2 else None
        height = _optional_int(image_info[3]) if len(image_info) > 3 else None
        bbox = _first_image_bbox(
            page=page, xref=xref, page_index=page_index, image_index=image_index, warnings=warnings
        )
        extension: str | None = None
        output_path: Path | None = None

        if xref is not None and extract_images:
            try:
                extracted = document.extract_image(xref)
                extension = _normalize_extension(extracted.get("ext"))
                if assets_dir is not None:
                    output_path = assets_dir / f"page_{page_index}_image_{image_index}.{extension}"
                    output_path.write_bytes(extracted["image"])
            except Exception as exc:
                warnings.append(
                    f"Failed to extract image p{page_index}_i{image_index}: {exc}",
                )
        elif len(image_info) > 7:
            extension = _optional_str(image_info[7])

        images.append(
            ImageBlock(
                id=f"p{page_index}_i{image_index}",
                page_index=page_index,
                image_index=image_index,
                xref=xref,
                bbox=bbox,
                width=width,
                height=height,
                extension=extension,
                output_path=output_path,
            ),
        )
    return images


def _first_image_bbox(
    *,
    page: Any,
    xref: int | None,
    page_index: int,
    image_index: int,
    warnings: list[str],
) -> BoundingBox | None:
    if xref is None:
        return None
    try:
        rects = page.get_image_rects(xref)
    except Exception as exc:
        warnings.append(f"Failed to locate image p{page_index}_i{image_index}: {exc}")
        return None
    if not rects:
        return None
    rect = rects[0]
    return BoundingBox(x0=float(rect.x0), y0=float(rect.y0), x1=float(rect.x1), y1=float(rect.y1))


def _bbox_from_sequence(value: Any) -> BoundingBox | None:
    if value is None or len(value) != 4:
        return None
    return BoundingBox(
        x0=float(value[0]), y0=float(value[1]), x1=float(value[2]), y1=float(value[3])
    )


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in metadata.items()}


def _normalize_extension(value: Any) -> str:
    extension = str(value or "bin").strip().lower().lstrip(".")
    return extension or "bin"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
