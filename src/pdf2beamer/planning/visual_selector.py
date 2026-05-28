"""Select relevant, non-repeating visuals for generated slides."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
import re
from pdf2beamer.ir import PaperIR, Slide, SlideIR, SlideVisual
from pdf2beamer.retrieval import RerankResult

_VISUAL_ROLES = {"problem", "method", "architecture", "experiments", "results"}
_MIN_FIGURE_SCORE = 0.55
_MIN_TABLE_SCORE = 0.45
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "paper", "present", "show", "shows", "that", "the", "this", "to",
    "we", "with",
}
_ROLE_TERMS = {
    "problem": {"problem", "challenge", "motivation", "example", "failure", "limitation"},
    "method": {"method", "approach", "algorithm", "model", "framework", "pipeline"},
    "architecture": {"architecture", "module", "component", "system", "framework", "overview"},
    "experiments": {"experiment", "dataset", "setup", "benchmark", "implementation", "training"},
    "results": {"result", "comparison", "benchmark", "performance", "ablation", "evaluation"},
}


@dataclass(frozen=True)
class VisualCandidate:
    id: str
    type: str
    source_ids: tuple[str, ...]
    page_index: int | None
    linked_section_id: str | None
    caption: str | None
    text: str
    path: str | None = None
    content: str | None = None
    confidence: float = 0.0
    content_key: str | None = None


def attach_relevant_visuals(
    slide_ir: SlideIR,
    paper_ir: PaperIR | None,
    contexts: list[RerankResult],
) -> SlideIR:
    """Attach at most one relevant visual per eligible slide without repeats."""

    if paper_ir is None:
        return slide_ir

    candidates = _visual_candidates(paper_ir)
    if not candidates:
        return slide_ir

    used_ids = {visual.id for slide in slide_ir.slides for visual in slide.visuals}
    used_keys = {visual.id for slide in slide_ir.slides for visual in slide.visuals}
    used_pages: set[int] = set()
    updated: list[Slide] = []
    for slide in slide_ir.slides:
        if slide.role == "title" or slide.visuals or slide.role not in _VISUAL_ROLES:
            updated.append(slide)
            continue
        candidate = _best_candidate(slide, candidates, contexts, used_ids, used_keys, used_pages)
        if candidate is None:
            updated.append(slide)
            continue
        used_ids.add(candidate.id)
        if candidate.content_key:
            used_keys.add(candidate.content_key)
        if candidate.page_index is not None:
            used_pages.add(candidate.page_index)
        visual = _to_slide_visual(candidate, slide.role)
        source_ids = list(dict.fromkeys([*slide.source_ids, *visual.source_ids]))
        updated.append(
            slide.model_copy(
                update={
                    "layout": _layout_for_visual(slide.role, visual.type, slide.layout),
                    "visuals": [visual],
                    "source_ids": source_ids,
                },
            ),
        )
    return slide_ir.model_copy(update={"slides": updated})


def _visual_candidates(paper_ir: PaperIR) -> list[VisualCandidate]:
    output: list[VisualCandidate] = []
    seen_keys: set[str] = set()
    for figure in paper_ir.figures:
        if figure.path is None:
            continue
        key = _figure_content_key(figure.path)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        output.append(
            VisualCandidate(
                id=figure.id,
                type="figure",
                source_ids=(figure.id,),
                page_index=figure.page_index,
                linked_section_id=figure.linked_section_id,
                caption=figure.caption,
                text=" ".join(part for part in (figure.caption, figure.linked_section_id) if part),
                path=str(figure.path),
                confidence=figure.confidence,
                content_key=key,
            ),
        )
    for table in paper_ir.tables:
        if not (table.text or table.caption):
            continue
        content = _compact_table_text(table.text or table.caption or "")
        if not content:
            continue
        key = _text_key(content)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        output.append(
            VisualCandidate(
                id=table.id,
                type="table",
                source_ids=(table.id,),
                page_index=table.page_index,
                linked_section_id=table.linked_section_id,
                caption=table.caption,
                text=" ".join(part for part in (table.caption, table.text) if part),
                content=content,
                confidence=table.confidence,
                content_key=key,
            ),
        )
    return output


def _best_candidate(
    slide: Slide,
    candidates: list[VisualCandidate],
    contexts: list[RerankResult],
    used_ids: set[str],
    used_keys: set[str],
    used_pages: set[int],
) -> VisualCandidate | None:
    fresh_page_scored = []
    fallback_scored = []
    for candidate in candidates:
        if candidate.id in used_ids or candidate.content_key in used_keys:
            continue
        if candidate.type == "table" and slide.role not in {"experiments", "results"}:
            continue
        score = _candidate_score(slide, candidate, contexts)
        threshold = _MIN_TABLE_SCORE if candidate.type == "table" else _MIN_FIGURE_SCORE
        if score < threshold:
            continue
        item = (score, candidate)
        if candidate.page_index is not None and candidate.page_index in used_pages:
            fallback_scored.append((score - 0.35, candidate))
        else:
            fresh_page_scored.append(item)
    scored = fresh_page_scored or fallback_scored
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], _page_sort_key(item[1])), reverse=True)
    return scored[0][1]


def _candidate_score(slide: Slide, candidate: VisualCandidate, contexts: list[RerankResult]) -> float:
    slide_tokens = _tokens(" ".join([slide.title, slide.main_message, *[b.text for b in slide.bullets]]))
    visual_tokens = _tokens(" ".join([candidate.caption or "", candidate.text]))
    score = 0.2 + 0.35 * candidate.confidence
    score += 1.2 * _jaccard(slide_tokens, visual_tokens)
    score += _role_term_bonus(slide.role, visual_tokens)
    score += _context_alignment_bonus(candidate, contexts)
    if candidate.caption and len(candidate.caption.split()) >= 4:
        score += 0.15
    if candidate.page_index is not None and candidate.page_index <= 1 and slide.role in {"problem", "method"}:
        score += 0.08
    if candidate.type == "table" and slide.role == "results":
        score += 0.15
    return score


def _context_alignment_bonus(candidate: VisualCandidate, contexts: list[RerankResult]) -> float:
    if candidate.page_index is None and candidate.linked_section_id is None:
        return 0.0
    bonus = 0.0
    for context in contexts:
        if candidate.page_index is not None and candidate.page_index in context.source_pages:
            bonus += 0.22
        if candidate.linked_section_id and candidate.linked_section_id == context.section_id:
            bonus += 0.3
    return min(bonus, 0.8)


def _role_term_bonus(role: str, visual_tokens: set[str]) -> float:
    terms = _ROLE_TERMS.get(role, set())
    if not terms:
        return 0.0
    return min(0.5, 0.12 * len(terms & visual_tokens))


def _to_slide_visual(candidate: VisualCandidate, role: str) -> SlideVisual:
    return SlideVisual(
        id=candidate.id,
        type=candidate.type,
        path=candidate.path,
        caption=_compact_caption(candidate.caption),
        content=candidate.content,
        source_ids=list(candidate.source_ids),
    )


def _layout_for_visual(role: str, visual_type: str, current_layout: str) -> str:
    if visual_type == "table":
        return "table"
    if visual_type in {"figure", "image"}:
        if role in {"architecture", "method"}:
            return "figure_left_bullets_right"
        return "figure_top_bullets_bottom"
    return current_layout


def _figure_content_key(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return f"missing:{path}"
    digest = blake2b(data, digest_size=12).hexdigest()
    return f"figure:{digest}"


def _text_key(text: str) -> str:
    return "text:" + blake2b(" ".join(text.split()).encode("utf-8"), digest_size=12).hexdigest()


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in _STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _page_sort_key(candidate: VisualCandidate) -> int:
    return -1 if candidate.page_index is None else -candidate.page_index


def _compact_caption(caption: str | None) -> str | None:
    if not caption:
        return None
    text = " ".join(caption.split())
    if len(text) <= 140:
        return text
    return text[:137].rstrip(" ,;:.") + "..."


def _compact_table_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines or any("|" in line for line in lines):
        return ""
    return "\n".join(line[:100].rstrip() for line in lines[:4])


def _title_from_role(role: str) -> str:
    titles = {
        "problem": "Problem and Motivation",
        "method": "Method Overview",
        "architecture": "Architecture Overview",
        "experiments": "Experimental Setup",
        "results": "Results",
    }
    return titles.get(role, "Visual Evidence")
