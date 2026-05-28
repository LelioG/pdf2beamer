"""Native PDF diagnostics implemented with PyMuPDF.

This module performs local-only checks to decide whether an input PDF is inside
the project scope: native scientific PDFs with an exploitable text layer. It
does not perform OCR and does not call external services.
"""

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

MEANINGFUL_TEXT_CHARS: Final[int] = 50
MIN_TEXT_LAYER_PAGE_RATIO: Final[float] = 0.5
MIN_NATIVE_AVG_CHARS_PER_PAGE: Final[float] = 100.0
LOW_AVG_TEXT_WARNING_THRESHOLD: Final[float] = 700.0
HIGH_IMAGES_PER_PAGE_WARNING_THRESHOLD: Final[float] = 3.0


class PDFDiagnostics(BaseModel):
    """Structured suitability result for a candidate input PDF."""

    model_config = ConfigDict(extra="forbid")

    pdf_path: Path
    exists: bool
    is_file: bool
    page_count: int = Field(ge=0)
    has_text_layer: bool
    avg_chars_per_page: float = Field(ge=0.0)
    total_chars: int = Field(ge=0)
    image_count: int = Field(ge=0)
    text_page_count: int = Field(ge=0)
    empty_text_page_count: int = Field(ge=0)
    estimated_native_pdf: bool
    is_supported_native_pdf: bool
    rejection_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    file_size_bytes: int | None = Field(default=None, ge=0)
    min_chars_per_page: int | None = Field(default=None, ge=0)
    max_chars_per_page: int | None = Field(default=None, ge=0)
    per_page_char_counts: list[int] = Field(default_factory=list)
    per_page_image_counts: list[int] = Field(default_factory=list)


# Backward-compatible alias for the initial scaffold.
PdfDiagnostics = PDFDiagnostics


def diagnose_pdf(pdf_path: str | Path) -> PDFDiagnostics:
    """Inspect a PDF locally and return a structured support decision.

    The function catches normal invalid-input errors and returns an unsupported
    diagnostics object instead of raising. PyMuPDF is imported lazily so package
    imports remain usable in environments that have not installed PDF extras.
    """

    path = Path(pdf_path)
    exists = path.exists()
    is_file = path.is_file() if exists else False
    file_size_bytes = path.stat().st_size if is_file else None

    if not exists:
        return _unsupported(path, exists=False, is_file=False, reason="File does not exist.")
    if not is_file:
        return _unsupported(
            path,
            exists=True,
            is_file=False,
            file_size_bytes=file_size_bytes,
            reason="Path is not a file.",
        )
    if path.suffix.lower() != ".pdf":
        return _unsupported(
            path,
            exists=True,
            is_file=True,
            file_size_bytes=file_size_bytes,
            reason="File is not a PDF.",
        )

    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        return _unsupported(
            path,
            exists=True,
            is_file=True,
            file_size_bytes=file_size_bytes,
            reason=f"PDF diagnostics failed: PyMuPDF is not available: {exc}",
        )

    try:
        with fitz.open(path) as document:
            page_count = int(document.page_count)
            if page_count == 0:
                return _unsupported(
                    path,
                    exists=True,
                    is_file=True,
                    file_size_bytes=file_size_bytes,
                    reason="PDF has no pages.",
                )

            per_page_char_counts: list[int] = []
            per_page_image_counts: list[int] = []

            for page in document:
                text = page.get_text("text") or ""
                char_count = len(text.strip())
                per_page_char_counts.append(char_count)
                per_page_image_counts.append(len(page.get_images(full=True)))
    except Exception as exc:
        return _unsupported(
            path,
            exists=True,
            is_file=True,
            file_size_bytes=file_size_bytes,
            reason=_open_error_reason(exc),
        )

    return _build_result(
        path=path,
        exists=True,
        is_file=True,
        file_size_bytes=file_size_bytes,
        per_page_char_counts=per_page_char_counts,
        per_page_image_counts=per_page_image_counts,
    )


def _build_result(
    *,
    path: Path,
    exists: bool,
    is_file: bool,
    file_size_bytes: int | None,
    per_page_char_counts: list[int],
    per_page_image_counts: list[int],
) -> PDFDiagnostics:
    page_count = len(per_page_char_counts)
    total_chars = sum(per_page_char_counts)
    image_count = sum(per_page_image_counts)
    text_page_count = sum(1 for count in per_page_char_counts if count >= MEANINGFUL_TEXT_CHARS)
    empty_text_page_count = sum(1 for count in per_page_char_counts if count == 0)
    avg_chars_per_page = total_chars / page_count if page_count else 0.0
    has_text_layer = (
        bool(page_count) and (text_page_count / page_count) >= MIN_TEXT_LAYER_PAGE_RATIO
    )
    estimated_native_pdf = (
        page_count > 0 and has_text_layer and avg_chars_per_page >= MIN_NATIVE_AVG_CHARS_PER_PAGE
    )
    is_supported_native_pdf = exists and is_file and page_count > 0 and estimated_native_pdf
    rejection_reason = (
        None if is_supported_native_pdf else _rejection_reason(page_count, has_text_layer)
    )

    return PDFDiagnostics(
        pdf_path=path,
        exists=exists,
        is_file=is_file,
        page_count=page_count,
        has_text_layer=has_text_layer,
        avg_chars_per_page=avg_chars_per_page,
        total_chars=total_chars,
        image_count=image_count,
        text_page_count=text_page_count,
        empty_text_page_count=empty_text_page_count,
        estimated_native_pdf=estimated_native_pdf,
        is_supported_native_pdf=is_supported_native_pdf,
        rejection_reason=rejection_reason,
        warnings=_warnings(
            page_count=page_count,
            avg_chars_per_page=avg_chars_per_page,
            image_count=image_count,
            empty_text_page_count=empty_text_page_count,
        ),
        file_size_bytes=file_size_bytes,
        min_chars_per_page=min(per_page_char_counts) if per_page_char_counts else None,
        max_chars_per_page=max(per_page_char_counts) if per_page_char_counts else None,
        per_page_char_counts=per_page_char_counts,
        per_page_image_counts=per_page_image_counts,
    )


def _unsupported(
    path: Path,
    *,
    exists: bool,
    is_file: bool,
    reason: str,
    file_size_bytes: int | None = None,
) -> PDFDiagnostics:
    return PDFDiagnostics(
        pdf_path=path,
        exists=exists,
        is_file=is_file,
        page_count=0,
        has_text_layer=False,
        avg_chars_per_page=0.0,
        total_chars=0,
        image_count=0,
        text_page_count=0,
        empty_text_page_count=0,
        estimated_native_pdf=False,
        is_supported_native_pdf=False,
        rejection_reason=reason,
        warnings=[],
        file_size_bytes=file_size_bytes,
        min_chars_per_page=None,
        max_chars_per_page=None,
        per_page_char_counts=[],
        per_page_image_counts=[],
    )


def _rejection_reason(page_count: int, has_text_layer: bool) -> str:
    if page_count == 0:
        return "PDF has no pages."
    if not has_text_layer:
        return "PDF does not contain a sufficient exploitable text layer."
    return "PDF appears to be image-only or scanned, which is outside the project scope."


def _warnings(
    *,
    page_count: int,
    avg_chars_per_page: float,
    image_count: int,
    empty_text_page_count: int,
) -> list[str]:
    warnings: list[str] = []
    if 0 < avg_chars_per_page < LOW_AVG_TEXT_WARNING_THRESHOLD:
        warnings.append("PDF has very low average extracted text per page.")
    if page_count and (image_count / page_count) > HIGH_IMAGES_PER_PAGE_WARNING_THRESHOLD:
        warnings.append("PDF has a high image count compared to page count.")
    if empty_text_page_count:
        warnings.append("Some pages have no extractable text.")
    if page_count == 1:
        warnings.append("PDF has fewer than 2 pages.")
    if page_count > 80:
        warnings.append("PDF has more than 80 pages.")
    return warnings


def _open_error_reason(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return "PDF could not be opened."
    return f"PDF could not be opened: {message}"
