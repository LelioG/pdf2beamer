"""Docling structured extraction adapter.

This module extracts a higher-level logical view of a native scientific PDF
using Docling when it is installed. It does not perform OCR, PyMuPDF extraction,
model inference, retrieval, fusion, slide generation, or Beamer rendering.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DoclingExtractionError(ValueError):
    """Raised when Docling extraction cannot proceed."""


class DoclingNotInstalledError(DoclingExtractionError):
    """Raised when Docling is not installed in the current environment."""


class DoclingTextItem(BaseModel):
    """A text-bearing item extracted from a Docling document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    label: str | None = None
    page_index: int | None = None
    level: int | None = None
    source_ref: str | None = None


class DoclingSection(BaseModel):
    """A conservative section grouping inferred from Docling text items."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    level: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    item_ids: list[str] = Field(default_factory=list)


class DoclingTable(BaseModel):
    """Table object exposed by Docling, when available."""

    model_config = ConfigDict(extra="forbid")

    id: str
    caption: str | None = None
    text: str | None = None
    page_index: int | None = None
    source_ref: str | None = None


class DoclingFigure(BaseModel):
    """Figure or picture object exposed by Docling, when available."""

    model_config = ConfigDict(extra="forbid")

    id: str
    caption: str | None = None
    page_index: int | None = None
    source_ref: str | None = None


class DoclingExtraction(BaseModel):
    """Structured logical extraction returned by the Docling adapter."""

    model_config = ConfigDict(extra="forbid")

    pdf_path: Path
    title: str | None = None
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)
    items: list[DoclingTextItem] = Field(default_factory=list)
    sections: list[DoclingSection] = Field(default_factory=list)
    tables: list[DoclingTable] = Field(default_factory=list)
    figures: list[DoclingFigure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DoclingAdapter:
    """Small object-oriented wrapper around :func:`extract_with_docling`."""

    def extract(self, pdf_path: str | Path) -> DoclingExtraction:
        """Extract a native PDF with Docling."""

        return extract_with_docling(pdf_path)


def extract_with_docling(pdf_path: str | Path) -> DoclingExtraction:
    """Convert a native PDF with Docling and return a structured extraction."""

    path = _validate_pdf_path(pdf_path)
    converter = _instantiate_converter(_load_document_converter())

    try:
        conversion_result = converter.convert(path)
    except Exception as exc:
        raise DoclingExtractionError(f"Failed to convert PDF with Docling: {exc}") from exc

    document = _safe_getattr(conversion_result, "document", conversion_result)
    warnings: list[str] = []

    text = _export_document_text(document, warnings)
    metadata = _extract_metadata(conversion_result=conversion_result, document=document)
    items = _extract_text_items(document=document, fallback_text=text, warnings=warnings)
    sections = _build_sections(items=items, warnings=warnings)
    tables = _extract_tables(document=document, warnings=warnings)
    figures = _extract_figures(document=document, warnings=warnings)
    title = _infer_title(document=document, metadata=metadata, items=items, text=text)

    if not text.strip():
        warnings.append("Docling produced empty document text.")
    if not items:
        warnings.append("Docling did not expose text items; only document text is available.")

    return DoclingExtraction(
        pdf_path=path,
        title=title,
        text=text,
        metadata=metadata,
        items=items,
        sections=sections,
        tables=tables,
        figures=figures,
        warnings=warnings,
    )


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)
    if not path.exists():
        raise DoclingExtractionError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise DoclingExtractionError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise DoclingExtractionError(f"File is not a PDF: {path}")
    return path


def _load_document_converter() -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise DoclingNotInstalledError(
            "Docling is not installed. Install pdf2beamer with the [docling] extra.",
        ) from exc

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.enable_remote_services = False
        if hasattr(pipeline_options, "force_backend_text"):
            pipeline_options.force_backend_text = True
        pipeline_options.do_picture_classification = False
        pipeline_options.do_picture_description = False
        pipeline_options.do_code_enrichment = False
        pipeline_options.do_formula_enrichment = False
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )
    except Exception as exc:
        raise DoclingExtractionError(
            "Docling could not be configured with OCR disabled; refusing to run OCR."
        ) from exc


def _instantiate_converter(converter_or_cls: Any) -> Any:
    if isinstance(converter_or_cls, type):
        return converter_or_cls()
    if hasattr(converter_or_cls, "convert"):
        return converter_or_cls
    return converter_or_cls()


def _export_document_text(document: Any, warnings: list[str]) -> str:
    for method_name in ("export_to_markdown", "export_to_text"):
        method = _safe_getattr(document, method_name)
        if callable(method):
            try:
                text = method()
            except Exception as exc:
                warnings.append(f"Docling {method_name} failed: {exc}")
                continue
            if text is not None:
                return str(text)
    return str(document) if document is not None else ""


def _extract_metadata(*, conversion_result: Any, document: Any) -> dict[str, str]:
    metadata: dict[str, str] = {}
    sources = (
        conversion_result,
        document,
        _safe_getattr(document, "metadata"),
        _safe_getattr(document, "origin"),
    )
    for source in sources:
        raw_metadata = _safe_getattr(source, "metadata") if source is not None else None
        if isinstance(source, dict):
            raw_metadata = source
        if isinstance(raw_metadata, dict):
            metadata.update(
                {
                    str(key): "" if value is None else str(value)
                    for key, value in raw_metadata.items()
                },
            )
    title = _safe_getattr(document, "title") or _safe_getattr(
        _safe_getattr(document, "metadata"),
        "title",
    )
    if title:
        metadata.setdefault("title", str(title))
    return metadata


def _extract_text_items(
    *,
    document: Any,
    fallback_text: str,
    warnings: list[str],
) -> list[DoclingTextItem]:
    raw_items = _iter_document_text_items(document=document, warnings=warnings)
    items: list[DoclingTextItem] = []
    for raw_item, level in raw_items:
        text = _extract_item_text(raw_item)
        if not text:
            continue
        item = DoclingTextItem(
            id=f"dtext_{len(items)}",
            text=text,
            label=_label_to_str(_safe_getattr(raw_item, "label")),
            page_index=_extract_page_index(raw_item),
            level=level if level is not None else _optional_int(_safe_getattr(raw_item, "level")),
            source_ref=_extract_source_ref(raw_item),
        )
        items.append(item)

    if items:
        return items

    fallback_items = _items_from_fallback_text(fallback_text)
    if fallback_items:
        warnings.append(
            "Docling text items were unavailable; created coarse items from exported text.",
        )
    return fallback_items


def _iter_document_text_items(document: Any, warnings: list[str]) -> list[tuple[Any, int | None]]:
    items: list[tuple[Any, int | None]] = []

    iterate_items = _safe_getattr(document, "iterate_items")
    if callable(iterate_items):
        try:
            for raw in iterate_items():
                item, level = _unpack_iterated_item(raw)
                if _looks_text_like(item):
                    items.append((item, level))
        except Exception as exc:
            warnings.append(f"Docling iterate_items failed: {exc}")

    if items:
        return items

    for attr_name in ("texts", "text_items", "paragraphs", "body"):
        raw_collection = _safe_getattr(document, attr_name)
        for item in _as_iterable(raw_collection):
            if _looks_text_like(item):
                items.append((item, _optional_int(_safe_getattr(item, "level"))))
        if items:
            return items

    return items


def _unpack_iterated_item(raw: Any) -> tuple[Any, int | None]:
    if isinstance(raw, tuple):
        item = raw[0] if raw else None
        level = None
        for value in raw[1:]:
            maybe_level = _optional_int(value)
            if maybe_level is not None:
                level = maybe_level
                break
        return item, level
    return raw, None


def _looks_text_like(item: Any) -> bool:
    if item is None:
        return False
    label = _label_to_str(_safe_getattr(item, "label")) or ""
    if _is_non_narrative_label(label):
        return False
    if _extract_item_text(item):
        return True
    lowered = label.lower()
    return any(term in lowered for term in ("text", "heading", "title", "paragraph"))


def _is_non_narrative_item(item: DoclingTextItem) -> bool:
    return _is_non_narrative_label(item.label or "") or _looks_like_exported_table_text(item.text)


def _is_non_narrative_label(label: str) -> bool:
    normalized = label.lower().replace("-", "_")
    return any(
        term in normalized for term in ("table", "picture", "figure", "formula", "code", "caption")
    )


def _looks_like_exported_table_text(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return False
    lowered = compact.lower()
    if "|" in compact and ("---" in compact or lowered.startswith("table ")):
        return True
    numeric_tokens = len(re.findall(r"\b\d+(?:\.\d+)?\b", compact))
    return lowered.startswith("table ") and len(compact) > 220 and numeric_tokens >= 8


def _items_from_fallback_text(text: str) -> list[DoclingTextItem]:
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if len(chunks) <= 1:
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
    return [
        DoclingTextItem(
            id=f"dtext_{index}",
            text=chunk,
            label=None,
            page_index=None,
            level=None,
            source_ref=None,
        )
        for index, chunk in enumerate(chunks)
    ]


def _build_sections(items: list[DoclingTextItem], warnings: list[str]) -> list[DoclingSection]:
    sections: list[DoclingSection] = []
    current_title: str | None = None
    current_level = 1
    current_page_start: int | None = None
    current_page_end: int | None = None
    current_item_ids: list[str] = []
    current_text_parts: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_level, current_page_start, current_page_end
        nonlocal current_item_ids, current_text_parts
        if current_title is None:
            return
        sections.append(
            DoclingSection(
                id=f"dsec_{len(sections)}",
                title=current_title,
                level=current_level,
                text="\n\n".join(current_text_parts).strip(),
                page_start=current_page_start,
                page_end=current_page_end,
                item_ids=current_item_ids,
            ),
        )
        current_title = None
        current_level = 1
        current_page_start = None
        current_page_end = None
        current_item_ids = []
        current_text_parts = []

    for item in items:
        if _is_non_narrative_item(item):
            continue
        if _is_likely_heading(item):
            flush()
            current_title = item.text.strip()
            current_level = item.level or 1
            current_page_start = item.page_index
            current_page_end = item.page_index
            current_item_ids = [item.id]
            current_text_parts = []
            continue

        if current_title is not None:
            current_item_ids.append(item.id)
            current_text_parts.append(item.text.strip())
            if item.page_index is not None:
                current_page_end = item.page_index

    flush()
    if not sections:
        warnings.append("No reliable sections were detected from Docling text items.")
    return sections


def _is_likely_heading(item: DoclingTextItem) -> bool:
    text = item.text.strip()
    if not text:
        return False
    label = (item.label or "").lower().replace("-", "_")
    if label in {"section_header", "heading", "title", "subtitle", "header"}:
        return True

    normalized = _normalize_heading_text(text)
    if normalized in _COMMON_SCIENTIFIC_SECTIONS:
        return True
    if len(text) > 90 or len(text.split()) > 10:
        return False
    if text.endswith((".", ":", ";", ",")):
        return False
    return text.istitle() or text.isupper()


def _extract_tables(document: Any, warnings: list[str]) -> list[DoclingTable]:
    tables: list[DoclingTable] = []
    try:
        raw_tables = _as_iterable(_safe_getattr(document, "tables"))
        if not raw_tables:
            raw_tables = [
                item
                for item in _as_iterable(_safe_getattr(document, "items"))
                if _item_has_label(item, "table")
            ]
        for raw_table in raw_tables:
            tables.append(
                DoclingTable(
                    id=f"dtable_{len(tables)}",
                    caption=_extract_caption(raw_table),
                    text=_extract_item_text(raw_table, document=document),
                    page_index=_extract_page_index(raw_table),
                    source_ref=_extract_source_ref(raw_table),
                ),
            )
    except Exception as exc:
        warnings.append(f"Docling table extraction failed: {exc}")
    return tables


def _extract_figures(document: Any, warnings: list[str]) -> list[DoclingFigure]:
    figures: list[DoclingFigure] = []
    try:
        raw_figures: list[Any] = []
        for attr_name in ("pictures", "figures"):
            raw_figures.extend(_as_iterable(_safe_getattr(document, attr_name)))
        if not raw_figures:
            raw_figures = [
                item
                for item in _as_iterable(_safe_getattr(document, "items"))
                if _item_has_label(item, "picture") or _item_has_label(item, "figure")
            ]
        for raw_figure in raw_figures:
            figures.append(
                DoclingFigure(
                    id=f"dfigure_{len(figures)}",
                    caption=_extract_caption(raw_figure),
                    page_index=_extract_page_index(raw_figure),
                    source_ref=_extract_source_ref(raw_figure),
                ),
            )
    except Exception as exc:
        warnings.append(f"Docling figure extraction failed: {exc}")
    return figures


def _infer_title(
    *,
    document: Any,
    metadata: dict[str, str],
    items: list[DoclingTextItem],
    text: str,
) -> str | None:
    for value in (metadata.get("title"), _safe_getattr(document, "title")):
        if value and str(value).strip():
            return str(value).strip()
    for item in items:
        if (item.label or "").lower() in {"title", "document_title"} and item.text.strip():
            return item.text.strip()
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:300]
    return None


def _extract_item_text(item: Any, document: Any = None) -> str:
    if item is None:
        return ""
    for attr_name in ("text", "orig", "content"):
        value = _safe_getattr(item, attr_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    exported = _call_export(item, document=document)
    return exported.strip() if exported else ""


def _call_export(item: Any, document: Any = None) -> str | None:
    for method_name in ("export_to_markdown", "export_to_text"):
        method = _safe_getattr(item, method_name)
        if callable(method):
            try:
                if document is not None:
                    try:
                        value = method(doc=document)
                    except TypeError:
                        value = method(document)
                else:
                    value = method()
            except Exception:
                continue
            if value is not None:
                return str(value)
    return None


def _extract_caption(item: Any) -> str | None:
    for attr_name in ("caption", "caption_text", "name"):
        value = _safe_getattr(item, attr_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    captions = _safe_getattr(item, "captions")
    for caption in _as_iterable(captions):
        text = _extract_item_text(caption)
        if text:
            return text
    return None


def _extract_page_index(item: Any) -> int | None:
    for attr_name in ("page_index", "page_no", "page"):
        value = _optional_int(_safe_getattr(item, attr_name))
        if value is not None:
            return value - 1 if attr_name in {"page_no", "page"} and value > 0 else value
    prov = _safe_getattr(item, "prov") or _safe_getattr(item, "provenance")
    for entry in _as_iterable(prov):
        value = _optional_int(_safe_getattr(entry, "page_no") or _safe_getattr(entry, "page"))
        if value is not None:
            return value - 1 if value > 0 else value
    return None


def _extract_source_ref(item: Any) -> str | None:
    for attr_name in ("self_ref", "cref", "source_ref", "id"):
        value = _safe_getattr(item, attr_name)
        if value is not None:
            return str(value)
    return None


def _item_has_label(item: Any, expected: str) -> bool:
    label = (_label_to_str(_safe_getattr(item, "label")) or "").lower()
    return expected.lower() in label


def _label_to_str(label: Any) -> str | None:
    if label is None:
        return None
    for attr_name in ("value", "name"):
        value = _safe_getattr(label, attr_name)
        if value is not None:
            return str(value)
    return str(label)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Iterable):
        return list(value)
    return []


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_heading_text(text: str) -> str:
    cleaned = text.strip().rstrip(":")
    while cleaned and (cleaned[0].isdigit() or cleaned[0] in ". )("):
        cleaned = cleaned[1:].strip()
    return " ".join(cleaned.split()).lower()


_COMMON_SCIENTIFIC_SECTIONS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "proposed method",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "limitations",
    "conclusion",
    "references",
}
