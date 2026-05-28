"""Source mapping helpers for fusion outputs."""

from pdf2beamer.ingest.models import BoundingBox
from pdf2beamer.ir.paper_ir import SourceRef


def make_source_ref(
    page_index: int | None = None,
    bbox: BoundingBox | None = None,
    block_ids: list[str] | None = None,
    extractor_sources: list[str] | None = None,
    source_item_ids: list[str] | None = None,
    confidence: float | None = None,
) -> SourceRef:
    """Create a normalized SourceRef for a PaperIR element."""

    return SourceRef(
        page_index=page_index,
        bbox=bbox,
        block_ids=list(dict.fromkeys(block_ids or [])),
        extractor_sources=list(dict.fromkeys(extractor_sources or [])),
        source_item_ids=list(dict.fromkeys(source_item_ids or [])),
        confidence=confidence,
    )
