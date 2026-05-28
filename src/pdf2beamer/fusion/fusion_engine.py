"""Fusion of Docling and PyMuPDF extraction outputs into PaperIR."""

import re
from pathlib import Path

from pdf2beamer.fusion.confidence import (
    figure_confidence,
    paragraph_confidence,
    section_confidence,
    table_confidence,
)
from pdf2beamer.fusion.source_map import make_source_ref
from pdf2beamer.ingest import (
    DoclingExtraction,
    DoclingFigure,
    DoclingSection,
    PyMuPDFExtraction,
)
from pdf2beamer.ingest.models import ImageBlock, PageExtraction, TextBlock
from pdf2beamer.ir.paper_ir import (
    EquationIR,
    FigureIR,
    PaperIR,
    PaperMetadata,
    ParagraphIR,
    SectionIR,
    TableIR,
)


def build_paper_ir(
    docling_extraction: DoclingExtraction,
    pymupdf_extraction: PyMuPDFExtraction,
) -> PaperIR:
    """Build the first fused PaperIR from Docling and PyMuPDF extractions."""

    warnings = [*docling_extraction.warnings, *pymupdf_extraction.warnings]
    _warn_if_pdf_paths_differ(docling_extraction.pdf_path, pymupdf_extraction.pdf_path, warnings)

    sections = _build_sections(docling_extraction, pymupdf_extraction, warnings)
    paragraphs = [paragraph for section in sections for paragraph in section.paragraphs]
    abstract = _extract_abstract(sections)
    figures = _build_figures(docling_extraction, pymupdf_extraction, sections, warnings)
    tables = _build_tables(docling_extraction, sections)
    equations = []

    metadata = PaperMetadata(
        title=docling_extraction.title or _metadata_title(pymupdf_extraction.metadata),
        authors=_extract_authors(docling_extraction.metadata, pymupdf_extraction.metadata),
        page_count=pymupdf_extraction.page_count,
        pdf_path=pymupdf_extraction.pdf_path,
        metadata={**pymupdf_extraction.metadata, **docling_extraction.metadata},
    )

    return PaperIR(
        metadata=metadata,
        abstract=abstract,
        sections=sections,
        paragraphs=paragraphs,
        figures=figures,
        tables=tables,
        equations=equations,
        warnings=warnings,
    )


def split_text_into_paragraphs(text: str) -> list[str]:
    """Split text into coarse paragraphs without sentence-level fragmentation."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    if "\n\n" in normalized:
        candidates = normalized.split("\n\n")
    else:
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if len(lines) > 1 and all(len(line) >= 40 for line in lines):
            candidates = lines
        else:
            candidates = [" ".join(lines)] if lines else [normalized]

    paragraphs: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.split())
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _build_sections(
    docling: DoclingExtraction,
    pymupdf: PyMuPDFExtraction,
    warnings: list[str],
) -> list[SectionIR]:
    if docling.sections:
        sections = [
            _section_from_docling(section, docling, index)
            for index, section in enumerate(docling.sections)
        ]
        if not any(section.paragraphs for section in sections):
            warnings.append(
                "Docling sections did not contain paragraphs; using PyMuPDF fallback text.",
            )
            return _fallback_sections_from_pymupdf(pymupdf, warnings)
        return sections

    warnings.append("Docling did not provide sections; created fallback section from PyMuPDF text.")
    return _fallback_sections_from_pymupdf(pymupdf, warnings)


def _section_from_docling(
    section: DoclingSection,
    docling: DoclingExtraction,
    index: int,
) -> SectionIR:
    section_id = f"sec_{index}"
    paragraph_texts = split_text_into_paragraphs(section.text)
    paragraph_source = "docling_section"

    if not paragraph_texts:
        paragraph_texts = _paragraph_texts_from_docling_items(section, docling)
        paragraph_source = "docling_item"

    paragraphs: list[ParagraphIR] = []
    for paragraph_index, text in enumerate(paragraph_texts):
        page_index = section.page_start
        confidence = paragraph_confidence(paragraph_source, has_page=page_index is not None)
        paragraphs.append(
            ParagraphIR(
                id=f"p_{index}_{paragraph_index}",
                text=text,
                section_id=section_id,
                page_index=page_index,
                source=make_source_ref(
                    page_index=page_index,
                    extractor_sources=["docling"],
                    source_item_ids=[section.id, *section.item_ids],
                    confidence=confidence,
                ),
                confidence=confidence,
            ),
        )

    confidence = section_confidence(
        has_title=bool(section.title.strip()),
        has_text=bool(paragraphs),
        source="docling",
    )
    return SectionIR(
        id=section_id,
        title=section.title,
        level=max(section.level, 1),
        paragraphs=paragraphs,
        page_start=section.page_start,
        page_end=section.page_end,
        source=make_source_ref(
            page_index=section.page_start,
            extractor_sources=["docling"],
            source_item_ids=[section.id, *section.item_ids],
            confidence=confidence,
        ),
        confidence=confidence,
    )


def _paragraph_texts_from_docling_items(
    section: DoclingSection,
    docling: DoclingExtraction,
) -> list[str]:
    item_by_id = {item.id: item for item in docling.items}
    parts = []
    for item_id in section.item_ids:
        item = item_by_id.get(item_id)
        if item is None:
            continue
        if item.text.strip() == section.title.strip():
            continue
        parts.append(item.text)
    return split_text_into_paragraphs("\n\n".join(parts))


def _fallback_sections_from_pymupdf(
    pymupdf: PyMuPDFExtraction,
    warnings: list[str],
) -> list[SectionIR]:
    all_text = "\n\n".join(page.text for page in pymupdf.pages if page.text.strip())
    paragraph_texts = split_text_into_paragraphs(all_text)
    if not paragraph_texts:
        warnings.append("PyMuPDF fallback did not provide extractable text.")

    section_id = "sec_0"
    page_start = pymupdf.pages[0].page_index if pymupdf.pages else None
    page_end = pymupdf.pages[-1].page_index if pymupdf.pages else None
    block_ids = [block.id for page in pymupdf.pages for block in page.text_blocks]
    confidence = section_confidence(
        has_title=True,
        has_text=bool(paragraph_texts),
        source="pymupdf",
    )

    paragraphs: list[ParagraphIR] = []
    for index, text in enumerate(paragraph_texts):
        page = _page_for_paragraph_text(pymupdf.pages, text)
        page_index = page.page_index if page is not None else page_start
        page_block_ids = [block.id for block in page.text_blocks] if page is not None else block_ids
        para_confidence = paragraph_confidence("pymupdf", has_page=page_index is not None)
        paragraphs.append(
            ParagraphIR(
                id=f"p_0_{index}",
                text=text,
                section_id=section_id,
                page_index=page_index,
                source=make_source_ref(
                    page_index=page_index,
                    block_ids=page_block_ids,
                    extractor_sources=["pymupdf"],
                    confidence=para_confidence,
                ),
                confidence=para_confidence,
            ),
        )

    return [
        SectionIR(
            id=section_id,
            title="Document",
            level=1,
            paragraphs=paragraphs,
            page_start=page_start,
            page_end=page_end,
            source=make_source_ref(
                page_index=page_start,
                block_ids=block_ids,
                extractor_sources=["pymupdf"],
                confidence=confidence,
            ),
            confidence=confidence,
        ),
    ]


def _page_for_paragraph_text(pages: list[PageExtraction], paragraph: str) -> PageExtraction | None:
    needle = paragraph[:80].strip()
    if not needle:
        return None
    for page in pages:
        if needle in " ".join(page.text.split()):
            return page
    return None


def _build_figures(
    docling: DoclingExtraction,
    pymupdf: PyMuPDFExtraction,
    sections: list[SectionIR],
    warnings: list[str],
) -> list[FigureIR]:
    figures: list[FigureIR] = []
    used_docling_ids: set[str] = set()

    for page in pymupdf.pages:
        for image in page.images:
            docling_figure = _match_docling_figure(image, docling.figures)
            if docling_figure is not None:
                used_docling_ids.add(docling_figure.id)
            caption = (
                docling_figure.caption
                if docling_figure is not None
                else _nearby_caption(page, image)
            )
            linked_section_id = _link_section_by_page(image.page_index, sections)
            if linked_section_id is None:
                warnings.append(f"Could not link figure image {image.id} to a section.")
            matched_sources = 1 + (1 if docling_figure is not None else 0)
            confidence = figure_confidence(
                has_image=True,
                has_caption=bool(caption),
                matched_sources=matched_sources,
            )
            source_item_ids = [docling_figure.id] if docling_figure is not None else []
            figures.append(
                FigureIR(
                    id=f"fig_{len(figures)}",
                    path=image.output_path,
                    caption=caption,
                    page_index=image.page_index,
                    bbox=image.bbox,
                    linked_section_id=linked_section_id,
                    source=make_source_ref(
                        page_index=image.page_index,
                        bbox=image.bbox,
                        block_ids=[image.id],
                        extractor_sources=["pymupdf", *(["docling"] if docling_figure else [])],
                        source_item_ids=source_item_ids,
                        confidence=confidence,
                    ),
                    confidence=confidence,
                ),
            )

    for docling_figure in docling.figures:
        if docling_figure.id in used_docling_ids:
            continue
        linked_section_id = _link_section_by_page(docling_figure.page_index, sections)
        if linked_section_id is None:
            warnings.append(f"Could not link Docling figure {docling_figure.id} to a section.")
        confidence = figure_confidence(
            has_image=False,
            has_caption=bool(docling_figure.caption),
            matched_sources=1,
        )
        figures.append(
            FigureIR(
                id=f"fig_{len(figures)}",
                path=None,
                caption=docling_figure.caption,
                page_index=docling_figure.page_index,
                bbox=None,
                linked_section_id=linked_section_id,
                source=make_source_ref(
                    page_index=docling_figure.page_index,
                    extractor_sources=["docling"],
                    source_item_ids=[docling_figure.id],
                    confidence=confidence,
                ),
                confidence=confidence,
            ),
        )

    return figures


def _match_docling_figure(image: ImageBlock, figures: list[DoclingFigure]) -> DoclingFigure | None:
    same_page = [figure for figure in figures if figure.page_index == image.page_index]
    if same_page:
        return same_page[0]
    return None


def _nearby_caption(page: PageExtraction, image: ImageBlock) -> str | None:
    if image.bbox is None:
        return None
    candidates: list[tuple[float, TextBlock]] = []
    for block in page.text_blocks:
        text = block.text.strip()
        if not text.lower().startswith(("fig.", "figure")):
            continue
        vertical_distance = abs(block.bbox.y0 - image.bbox.y1)
        candidates.append((vertical_distance, block))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1].text


def _build_tables(docling: DoclingExtraction, sections: list[SectionIR]) -> list[TableIR]:
    tables: list[TableIR] = []
    for table in docling.tables:
        linked_section_id = _link_section_by_page(table.page_index, sections)
        confidence = table_confidence(has_text=bool(table.text), has_caption=bool(table.caption))
        tables.append(
            TableIR(
                id=f"tbl_{len(tables)}",
                caption=table.caption,
                text=table.text,
                page_index=table.page_index,
                linked_section_id=linked_section_id,
                source=make_source_ref(
                    page_index=table.page_index,
                    extractor_sources=["docling"],
                    source_item_ids=[table.id],
                    confidence=confidence,
                ),
                confidence=confidence,
            ),
        )
    return tables


def _build_equations(pymupdf: PyMuPDFExtraction, sections: list[SectionIR]) -> list[EquationIR]:
    equations: list[EquationIR] = []
    seen: set[str] = set()
    for page in pymupdf.pages:
        for block in page.text_blocks:
            text = _normalize_equation_text(block.text)
            if not _looks_like_equation(text):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            linked_section_id = _link_section_by_page(page.page_index, sections)
            confidence = 0.65 if "=" in text else 0.45
            equations.append(
                EquationIR(
                    id=f"eq_{len(equations)}",
                    text=text,
                    latex=None,
                    page_index=page.page_index,
                    bbox=block.bbox,
                    linked_section_id=linked_section_id,
                    source=make_source_ref(
                        page_index=page.page_index,
                        bbox=block.bbox,
                        block_ids=[block.id],
                        extractor_sources=["pymupdf"],
                        confidence=confidence,
                    ),
                    confidence=confidence,
                ),
            )
    return equations


def _normalize_equation_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _looks_like_equation(text: str) -> bool:
    if len(text) < 8 or len(text) > 220:
        return False
    if text.lower().startswith(("figure", "table", "algorithm")):
        return False
    if not any(symbol in text for symbol in ("=", "∑", "Σ", "\\sum", "\\frac", "→", "←")):
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", text)
    if len(alpha_tokens) > 28:
        return False
    symbol_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
    if symbol_chars / max(len(text), 1) < 0.06:
        return False
    return True


def _link_section_by_page(page_index: int | None, sections: list[SectionIR]) -> str | None:
    if page_index is None:
        return None
    containing = [
        section
        for section in sections
        if section.page_start is not None
        and section.page_end is not None
        and section.page_start <= page_index <= section.page_end
    ]
    if containing:
        return containing[-1].id

    previous = [
        section
        for section in sections
        if section.page_start is not None and section.page_start <= page_index
    ]
    if previous:
        return previous[-1].id
    return None


def _extract_abstract(sections: list[SectionIR]) -> str | None:
    for section in sections:
        if section.title.strip().lower() == "abstract":
            return "\n\n".join(paragraph.text for paragraph in section.paragraphs).strip() or None
    for section in sections:
        if "abstract" in section.title.strip().lower():
            return "\n\n".join(paragraph.text for paragraph in section.paragraphs).strip() or None
    return None


def _metadata_title(metadata: dict[str, str]) -> str | None:
    for key in ("title", "Title"):
        value = metadata.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _extract_authors(*metadata_sources: dict[str, str]) -> list[str]:
    for metadata in metadata_sources:
        value = metadata.get("authors") or metadata.get("author") or metadata.get("Author")
        if value:
            return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    return []


def _warn_if_pdf_paths_differ(path_a: Path, path_b: Path, warnings: list[str]) -> None:
    if path_a == path_b:
        return
    if path_a.resolve(strict=False) != path_b.resolve(strict=False):
        warnings.append(
            f"Docling and PyMuPDF extractions refer to different PDF paths: {path_a} != {path_b}.",
        )
