"""Simple deterministic confidence heuristics for fusion."""


def section_confidence(has_title: bool, has_text: bool, source: str) -> float:
    """Score a fused section using explicit first-version heuristics."""

    if source == "docling":
        if has_title and has_text:
            return 0.90
        if has_title:
            return 0.70
    if source == "pymupdf":
        return 0.50
    return 0.40


def paragraph_confidence(source: str, has_page: bool) -> float:
    """Score a fused paragraph by source and whether page provenance exists."""

    if source == "docling_section":
        return 0.90 if has_page else 0.85
    if source == "docling_item":
        return 0.70 if has_page else 0.65
    if source == "pymupdf":
        return 0.50 if has_page else 0.45
    return 0.40


def figure_confidence(has_image: bool, has_caption: bool, matched_sources: int) -> float:
    """Score a fused figure from image/caption availability."""

    if has_image and has_caption and matched_sources >= 2:
        return 0.90
    if has_image and has_caption:
        return 0.70
    if has_image:
        return 0.50
    if has_caption:
        return 0.40
    return 0.30


def table_confidence(has_text: bool, has_caption: bool) -> float:
    """Score a table from text and caption availability."""

    if has_text:
        return 0.80
    if has_caption:
        return 0.60
    return 0.40
