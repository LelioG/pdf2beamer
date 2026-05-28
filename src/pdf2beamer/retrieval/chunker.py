"""PaperIR chunk models and lightweight chunking primitives."""

import re

from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.ir import PaperIR


class PaperChunk(BaseModel):
    """A retrievable text chunk derived from PaperIR."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    chunk_type: str = "paragraph"
    section_id: str | None = None
    section_title: str | None = None
    source_pages: list[int] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


def chunk_paper_ir(paper_ir: PaperIR) -> list[PaperChunk]:
    """Create retrievable paragraph chunks from PaperIR.

    The extraction/fusion layers can produce tiny layout artifacts such as
    commas, dashes, table separators, and isolated labels. Those artifacts are
    harmful for retrieval because they can receive arbitrary embedding/rerank
    scores. Filter them here so downstream planning only sees meaningful text.
    """

    chunks: list[PaperChunk] = []
    section_title_by_id = {section.id: section.title for section in paper_ir.sections}
    paragraphs = paper_ir.paragraphs or [
        paragraph for section in paper_ir.sections for paragraph in section.paragraphs
    ]
    for index, paragraph in enumerate(paragraphs):
        text = _normalize_chunk_text(paragraph.text)
        section_title = section_title_by_id.get(paragraph.section_id or "")
        if not is_informative_chunk_text(text, section_title=section_title):
            continue
        chunks.append(
            PaperChunk(
                id=paragraph.id or f"chunk_{index}",
                text=text,
                chunk_type="paragraph",
                section_id=paragraph.section_id,
                section_title=section_title,
                source_pages=[] if paragraph.page_index is None else [paragraph.page_index],
                metadata={"paper_title": paper_ir.metadata.title or ""},
            ),
        )
    return chunks


_UNINFORMATIVE_SECTION_TERMS = (
    "references",
    "bibliography",
    "acknowledgment",
    "acknowledgement",
    "statement on the use of large language models",
)

_UNINFORMATIVE_EXACT_SECTIONS = {"n", "average", "f1"}


def is_informative_chunk_text(text: str, section_title: str | None = None) -> bool:
    """Return whether text is useful enough for retrieval/planning."""

    normalized = _normalize_chunk_text(text)
    section = (section_title or "").lower().strip()
    if section in _UNINFORMATIVE_EXACT_SECTIONS:
        return False
    if any(term in section for term in _UNINFORMATIVE_SECTION_TERMS):
        return False
    if len(normalized) < 30:
        return False
    lowered = normalized.lower()
    if _looks_like_table_dump(normalized):
        return False
    if lowered in {"this design provides several advantages:"}:
        return False
    if lowered.startswith(("figure ", "fig. ", "table ")) and len(normalized.split()) <= 10:
        return False
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", normalized)
    alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    if len(alpha_tokens) < 5:
        return False
    if normalized.endswith(":") and len(alpha_tokens) <= 8:
        return False
    alpha_chars = sum(1 for char in normalized if char.isalpha())
    if alpha_chars / max(len(normalized), 1) < 0.45:
        return False
    symbol_chars = sum(1 for char in normalized if not char.isalnum() and not char.isspace())
    symbol_ratio = symbol_chars / max(len(normalized), 1)
    if "=" in normalized and symbol_ratio > 0.12:
        return False
    if "=" in normalized and len(alpha_tokens) <= 12:
        return False
    if symbol_ratio > 0.35:
        return False
    if re.search(r"\bet al\.", normalized, flags=re.IGNORECASE) and re.search(
        r"\b20\d{2}\b",
        normalized,
    ):
        return False
    return True


def _looks_like_table_dump(text: str) -> bool:
    lowered = text.lower()
    if "|" in text:
        return True
    if not lowered.startswith(("table ", "tab. ")):
        return False

    numeric_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    year_tokens = re.findall(r"\b(?:19|20)\d{2}\b", text)
    if len(text) > 220 and len(numeric_tokens) >= 8:
        return True
    if len(text) > 180 and len(year_tokens) >= 3:
        return True
    return False


def _normalize_chunk_text(text: str) -> str:
    return " ".join(text.strip().split())
