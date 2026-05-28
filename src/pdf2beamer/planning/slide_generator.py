"""Generate sanitized SlideIR from DeckPlan and ArgumentGraph."""

from typing import Any

from pdf2beamer.ir import ArgumentGraph, ArgumentNode, PaperIR, Slide, SlideBullet, SlideIR, SlideVisual
from pdf2beamer.ir.deck_plan import DeckPlan, PlannedSlide
from pdf2beamer.llm import BaseGenerator, build_slide_ir_prompt, generate_validated_json
from pdf2beamer.retrieval import RerankResult

_SUPPORTED_LAYOUTS = {
    "title",
    "bullets",
    "two_columns",
    "figure_left_bullets_right",
    "figure_top_bullets_bottom",
    "table",
    "equation",
    "conclusion",
    "appendix",
}

_INTERNAL_GOAL_MESSAGES = {
    "introduce the paper",
    "introduce paper",
    "explain the main problem",
    "explain main problem",
    "summarize the contribution",
    "summarize contribution",
    "describe the method",
    "describe method",
    "present the key results",
    "present key results",
    "close with the main takeaway",
    "close with main takeaway",
}

_MIN_BULLETS_BY_ROLE = {
    "problem": 3,
    "contribution": 3,
    "method": 3,
    "architecture": 3,
    "experiments": 3,
    "results": 3,
    "limitations": 2,
    "takeaway": 2,
}

_MAX_BULLETS = 5
_MAX_BULLET_CHARS = 160


def generate_slide_ir(
    deck_plan: DeckPlan,
    argument_graph: ArgumentGraph,
    generator: BaseGenerator,
    contexts: list[RerankResult] | None = None,
    paper_ir: PaperIR | None = None,
) -> SlideIR:
    """Generate SlideIR with deterministic sanitization and fallback."""

    warnings: list[str] = []
    prompt = build_slide_ir_prompt(deck_plan, argument_graph, contexts)
    try:
        payload = generate_validated_json(generator, prompt, schema_name="SlideIR")
    except Exception as exc:
        warnings.append(f"SlideIR generation failed; used fallback: {exc}")
        return _fallback_slide_ir(deck_plan, argument_graph, warnings, contexts, paper_ir)

    if not isinstance(payload.get("slides"), list) or not payload.get("slides"):
        warnings.append("Generator returned no usable slides; used fallback SlideIR.")
        return _fallback_slide_ir(deck_plan, argument_graph, warnings, contexts, paper_ir)

    slide_ir = _slide_ir_from_payload(payload, deck_plan, argument_graph, warnings, contexts or [], paper_ir)
    if not slide_ir.slides:
        warnings.append("Generator returned no usable slides; used fallback SlideIR.")
        return _fallback_slide_ir(deck_plan, argument_graph, warnings, contexts, paper_ir)
    return slide_ir


def _slide_ir_from_payload(
    payload: dict[str, Any],
    deck_plan: DeckPlan,
    argument_graph: ArgumentGraph,
    warnings: list[str],
    contexts: list[RerankResult],
    paper_ir: PaperIR | None,
) -> SlideIR:
    raw_slides = payload.get("slides")
    if not isinstance(raw_slides, list):
        warnings.append("Generator output did not contain a slides list.")
        raw_slides = []

    slides: list[Slide] = []
    used_ids: set[str] = set()
    for index, raw_slide in enumerate(raw_slides):
        if not isinstance(raw_slide, dict):
            warnings.append(f"Skipped non-object slide at index {index}.")
            continue
        planned = deck_plan.slides[index] if index < len(deck_plan.slides) else None
        slide = _sanitize_slide(raw_slide, planned, index, used_ids, warnings)
        if slide is not None:
            slides.append(slide)

    slides = _order_and_fill_slides(slides, deck_plan, argument_graph, used_ids, warnings, contexts)
    slides = _attach_paper_visuals(slides, paper_ir, contexts)
    slides = [_polish_slide(slide, argument_graph) for slide in slides]
    if deck_plan.slide_count and len(slides) != deck_plan.slide_count:
        warnings.append(
            "Slide count differs from DeckPlan: "
            f"got {len(slides)}, expected {deck_plan.slide_count}.",
        )

    payload_warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    warnings.extend(str(warning) for warning in payload_warnings)
    return SlideIR(
        paper_title=_clean_paper_title(_optional_str(payload.get("paper_title")) or argument_graph.paper_title),
        audience=_optional_str(payload.get("audience")) or deck_plan.audience,
        duration_minutes=(
            _optional_int(payload.get("duration_minutes")) or deck_plan.duration_minutes
        ),
        slides=slides,
        warnings=warnings,
    )


def _sanitize_slide(
    raw_slide: dict[str, Any],
    planned: PlannedSlide | None,
    index: int,
    used_ids: set[str],
    warnings: list[str],
) -> Slide | None:
    slide_warnings = [str(item) for item in raw_slide.get("warnings", []) if str(item).strip()]
    planned_role = _planned_role(planned)
    role = _optional_str(raw_slide.get("role")) or planned_role or "unknown"
    role = role.lower().strip()

    slide_id = _optional_str(raw_slide.get("id")) or f"slide_{index + 1:02d}"
    if slide_id in used_ids:
        original_id = slide_id
        slide_id = _unique_id(slide_id, used_ids)
        slide_warnings.append(f"Duplicate slide id {original_id!r} renamed to {slide_id!r}.")
    used_ids.add(slide_id)

    title = _optional_str(raw_slide.get("title")) or _title_from_role(role)
    layout = _optional_str(raw_slide.get("layout")) or _planned_layout(planned) or "bullets"
    layout = layout.lower().strip()
    if layout not in _SUPPORTED_LAYOUTS:
        slide_warnings.append(f"Unknown layout {layout!r}; using 'bullets'.")
        layout = "bullets"

    bullets = _sanitize_bullets(raw_slide.get("bullets"), slide_warnings)
    bullets = [bullet for bullet in bullets if _bullet_allowed_for_role(role, bullet.text)]
    main_message = _optional_str(raw_slide.get("main_message"))
    if not main_message:
        main_message = bullets[0].text if bullets else title
        slide_warnings.append("Missing main_message; filled from bullet or title.")
    elif _is_internal_goal_message(main_message):
        main_message = bullets[0].text if bullets else title
        slide_warnings.append("Internal planning goal removed from main_message.")
    if not bullets and role != "title":
        slide_warnings.append("Slide has no bullets.")

    source_ids = _string_list(raw_slide.get("source_ids"))
    for bullet in bullets:
        source_ids.extend(bullet.source_ids)
    source_ids = list(dict.fromkeys(source_ids))
    bullets = _ground_bullets(bullets, source_ids)

    visuals = _sanitize_visuals(raw_slide.get("visuals"), slide_warnings)
    visuals = _caption_generated_figures(visuals, role)
    if not title and not main_message and not bullets and role != "title":
        warnings.append(f"Removed empty slide at index {index}.")
        return None

    return Slide(
        id=slide_id,
        role=role,
        title=title,
        main_message=main_message,
        layout=layout,
        bullets=bullets,
        visuals=visuals,
        speaker_notes=_optional_str(raw_slide.get("speaker_notes")),
        source_ids=source_ids,
        warnings=slide_warnings,
    )


def _caption_generated_figures(visuals: list[SlideVisual], role: str) -> list[SlideVisual]:
    caption = _title_from_role(role)
    return [
        visual.model_copy(update={"caption": caption})
        if visual.type in {"figure", "image"} and not visual.caption
        else visual
        for visual in visuals
    ]


def _sanitize_bullets(value: object, warnings: list[str]) -> list[SlideBullet]:
    if not isinstance(value, list):
        return []
    bullets: list[SlideBullet] = []
    for index, raw_bullet in enumerate(value):
        if isinstance(raw_bullet, str):
            text = raw_bullet.strip()
            source_ids: list[str] = []
            confidence = None
        elif isinstance(raw_bullet, dict):
            text = str(raw_bullet.get("text") or "").strip()
            source_ids = _string_list(raw_bullet.get("source_ids"))
            confidence = _optional_clamped_float(raw_bullet.get("confidence"))
        else:
            warnings.append(f"Skipped invalid bullet at index {index}.")
            continue
        if not text:
            warnings.append(f"Removed empty bullet at index {index}.")
            continue
        if "…" in text or "..." in text or ". . ." in text:
            warnings.append(f"Removed ellipsized bullet at index {index}.")
            continue
        text = _compact_text(text, max_chars=_MAX_BULLET_CHARS)
        if not text or _looks_truncated(text):
            warnings.append(f"Removed truncated bullet at index {index}.")
            continue
        bullets.append(SlideBullet(text=text, source_ids=source_ids, confidence=confidence))
    return bullets


def _bullet_allowed_for_role(role: str, text: str) -> bool:
    if role != "limitations":
        return True
    lowered = text.lower()
    limitation_terms = (
        "limitation",
        "however",
        "cost",
        "failure",
        "degrade",
        "degradation",
        "redundancy",
        "trade-off",
        "tradeoff",
        "hurt",
        "artifact",
        "padding",
    )
    return any(term in lowered for term in limitation_terms)


def _ground_bullets(bullets: list[SlideBullet], fallback_source_ids: list[str]) -> list[SlideBullet]:
    if not fallback_source_ids:
        return bullets
    return [
        bullet if bullet.source_ids else bullet.model_copy(update={"source_ids": fallback_source_ids})
        for bullet in bullets
    ]


def _sanitize_visuals(value: object, warnings: list[str]) -> list[SlideVisual]:
    if not isinstance(value, list):
        return []
    visuals: list[SlideVisual] = []
    for index, raw_visual in enumerate(value):
        if not isinstance(raw_visual, dict):
            warnings.append(f"Skipped invalid visual at index {index}.")
            continue
        visual_id = _optional_str(raw_visual.get("id")) or f"visual_{index + 1:02d}"
        visual_type = _optional_str(raw_visual.get("type") or raw_visual.get("kind")) or "unknown"
        path = _optional_str(raw_visual.get("path"))
        content = _optional_str(raw_visual.get("content") or raw_visual.get("text"))
        caption = _optional_str(raw_visual.get("caption"))
        if visual_type in {"figure", "image"} and not path:
            warnings.append(f"Removed incomplete visual at index {index}: figure/image has no path.")
            continue
        if visual_type == "equation":
            warnings.append(f"Removed unsupported visual at index {index}: equations are disabled.")
            continue
        if visual_type == "table" and not content:
            warnings.append(f"Removed incomplete visual at index {index}: table has no content.")
            continue
        visuals.append(
            SlideVisual(
                id=visual_id,
                type=visual_type,
                path=path,
                caption=caption,
                content=content,
                source_ids=_string_list(raw_visual.get("source_ids")),
            ),
        )
    return visuals


def _order_and_fill_slides(
    slides: list[Slide],
    deck_plan: DeckPlan,
    argument_graph: ArgumentGraph,
    used_ids: set[str],
    warnings: list[str],
    contexts: list[RerankResult],
) -> list[Slide]:
    if not deck_plan.slides:
        return slides

    remaining = slides.copy()
    ordered: list[Slide] = []
    for planned in deck_plan.slides:
        role = _planned_role(planned)
        match_index = next((i for i, slide in enumerate(remaining) if slide.role == role), None)
        if match_index is None:
            warnings.append(f"Generator omitted planned role {role}; added fallback slide.")
            ordered.append(
                _fallback_slide_for_plan(planned, argument_graph, used_ids, len(ordered), contexts),
            )
            continue
        matched = remaining.pop(match_index)
        ordered.append(_ensure_slide_content(matched, planned, argument_graph, warnings, contexts))
    ordered.extend(remaining)
    return _renumber_slides(ordered)


def _attach_paper_visuals(
    slides: list[Slide],
    paper_ir: PaperIR | None,
    contexts: list[RerankResult],
) -> list[Slide]:
    if paper_ir is None:
        return slides

    used_visual_ids = {visual.id for slide in slides for visual in slide.visuals}
    updated: list[Slide] = []
    for slide in slides:
        if slide.role == "title" or slide.visuals:
            updated.append(slide)
            continue
        visual = _select_visual_for_slide(slide, paper_ir, contexts, used_visual_ids)
        if visual is None:
            updated.append(slide)
            continue
        used_visual_ids.add(visual.id)
        source_ids = list(dict.fromkeys([*slide.source_ids, *visual.source_ids]))
        layout = _layout_for_visual(slide.role, visual.type, slide.layout)
        updated.append(
            slide.model_copy(
                update={
                    "layout": layout,
                    "visuals": [visual],
                    "source_ids": source_ids,
                },
            ),
        )
    return updated


def _select_visual_for_slide(
    slide: Slide,
    paper_ir: PaperIR,
    contexts: list[RerankResult],
    used_visual_ids: set[str],
) -> SlideVisual | None:
    if slide.role == "method":
        figure = _best_figure(slide, paper_ir, contexts, used_visual_ids)
        if figure is not None:
            return figure
    if slide.role == "architecture":
        figure = _best_figure(slide, paper_ir, contexts, used_visual_ids)
        if figure is not None:
            return figure
    if slide.role in {"results", "experiments"}:
        figure = _best_figure(slide, paper_ir, contexts, used_visual_ids)
        if figure is not None:
            return figure
        table = _best_table(slide, paper_ir, contexts, used_visual_ids)
        if table is not None:
            return table
    if slide.role == "problem":
        return _best_figure(slide, paper_ir, contexts, used_visual_ids)
    return None


def _best_figure(
    slide: Slide,
    paper_ir: PaperIR,
    contexts: list[RerankResult],
    used_visual_ids: set[str],
) -> SlideVisual | None:
    candidates = [figure for figure in paper_ir.figures if figure.id not in used_visual_ids and figure.path]
    if not candidates:
        return None
    candidates.sort(key=lambda figure: _figure_score(slide, figure, contexts), reverse=True)
    best = candidates[0]
    if _figure_score(slide, best, contexts) < 0.4:
        return None
    return SlideVisual(
        id=best.id,
        type="figure",
        path=str(best.path) if best.path is not None else None,
        caption=_compact_caption(best.caption) or _title_from_role(slide.role),
        content=None,
        source_ids=[best.id],
    )


def _best_table(
    slide: Slide,
    paper_ir: PaperIR,
    contexts: list[RerankResult],
    used_visual_ids: set[str],
) -> SlideVisual | None:
    candidates = [table for table in paper_ir.tables if table.id not in used_visual_ids and (table.text or table.caption)]
    if not candidates:
        return None
    candidates.sort(key=lambda table: _table_score(slide, table, contexts), reverse=True)
    best = candidates[0]
    if _table_score(slide, best, contexts) <= 0.0:
        return None
    return SlideVisual(
        id=best.id,
        type="table",
        path=None,
        caption=_compact_caption(best.caption) or _title_from_role(slide.role),
        content=_compact_table_text(best.text or best.caption or ""),
        source_ids=[best.id],
    )


def _figure_score(slide: Slide, figure: object, contexts: list[RerankResult]) -> float:
    text = f"{getattr(figure, 'caption', '') or ''} {slide.title} {slide.main_message}".lower()
    score = 0.2 * float(getattr(figure, 'confidence', 0.0) or 0.0)
    score += _page_context_bonus(getattr(figure, 'page_index', None), contexts)
    role_terms = {
        "problem": ("manipulation", "example", "splicing", "inpainting"),
        "method": ("glra", "architecture", "framework", "attention", "relay"),
        "architecture": ("glra", "architecture", "framework", "attention", "relay", "structure"),
        "results": ("result", "comparison", "qualitative", "ablation"),
        "experiments": ("dataset", "benchmark", "setup", "comparison"),
    }
    score += sum(0.4 for term in role_terms.get(slide.role, ()) if term in text)
    path = getattr(figure, 'path', None)
    if path is not None:
        score += 0.1
    return score


def _table_score(slide: Slide, table: object, contexts: list[RerankResult]) -> float:
    text = f"{getattr(table, 'caption', '') or ''} {getattr(table, 'text', '') or ''}".lower()
    score = float(getattr(table, 'confidence', 0.0) or 0.0)
    score += _page_context_bonus(getattr(table, 'page_index', None), contexts)
    terms = {
        "results": ("result", "f1", "benchmark", "comparison", "ablation", "performance"),
        "experiments": ("dataset", "benchmark", "training", "setup", "implementation"),
    }.get(slide.role, ())
    score += sum(0.45 for term in terms if term in text)
    return score


def _page_context_bonus(page_index: int | None, contexts: list[RerankResult]) -> float:
    if page_index is None:
        return 0.0
    return sum(0.15 for context in contexts if page_index in context.source_pages)


def _layout_for_visual(role: str, visual_type: str, current_layout: str) -> str:
    if visual_type == "table":
        return "table"
    if visual_type in {"figure", "image"}:
        if role in {"architecture", "method"}:
            return "figure_left_bullets_right"
        return "figure_top_bullets_bottom"
    return current_layout


def _compact_caption(caption: str | None) -> str | None:
    if not caption:
        return None
    return _compact_text(caption, max_chars=140)


def _compact_table_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if any("|" in line for line in lines):
        return ""

    compacted: list[str] = []
    for line in lines[:4]:
        if len(line) > 100:
            line = _compact_text(line, max_chars=100)
        if line:
            compacted.append(line)
    return "\n".join(compacted)


def _polish_slide(slide: Slide, argument_graph: ArgumentGraph) -> Slide:
    if slide.role == "title":
        title = _clean_paper_title(argument_graph.paper_title) or _clean_paper_title(slide.title)
        if not title or title.lower() == "title":
            title = "Untitled Paper"
        return slide.model_copy(
            update={
                "title": title,
                "main_message": title,
                "bullets": [],
                "visuals": [],
                "layout": "title",
            },
        )

    bullets = [bullet for bullet in slide.bullets if not _looks_truncated(bullet.text)]
    return slide.model_copy(update={"bullets": _dedupe_bullets(bullets)})


def _dedupe_bullets(bullets: list[SlideBullet]) -> list[SlideBullet]:
    output: list[SlideBullet] = []
    for bullet in bullets:
        if any(_same_or_contained(bullet.text, existing.text) for existing in output):
            continue
        output.append(bullet)
    return output[:_MAX_BULLETS]


def _same_or_contained(left: str, right: str | None) -> bool:
    if not right:
        return False
    a = _normalized_for_compare(left)
    b = _normalized_for_compare(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter.split()) >= 7 and shorter in longer


def _normalized_for_compare(text: str) -> str:
    return " ".join(part.strip(".,;:()[]{}") for part in text.lower().split())


def _clean_paper_title(title: str | None) -> str | None:
    if not title:
        return None
    cleaned = " ".join(title.replace("\n", " ").split()).strip(" #")
    if not cleaned:
        return None
    if cleaned.isupper() or cleaned.lower().startswith("relayformer:"):
        return _title_case_preserving_acronyms(cleaned)
    return cleaned.replace("Relayformer", "RelayFormer")


def _title_case_preserving_acronyms(text: str) -> str:
    small_words = {"a", "an", "and", "as", "for", "in", "of", "on", "or", "the", "to", "with"}
    acronym_map = {
        "relayformer": "RelayFormer",
        "glra": "GLRA",
        "glr": "GLR",
        "vml": "VML",
    }
    words = text.lower().split()
    output: list[str] = []
    previous_ended_colon = False
    for index, word in enumerate(words):
        prefix = word[: len(word) - len(word.lstrip(":;,.()[]{}"))]
        suffix = word[len(word.rstrip(":;,.()[]{}")) :]
        core = word[len(prefix) : len(word) - len(suffix) if suffix else len(word)]
        if not core:
            output.append(word)
            continue
        if core in acronym_map:
            cased = acronym_map[core]
        elif "-" in core:
            cased = "-".join(_case_title_part(part, force=True) for part in core.split("-"))
        elif 0 < index < len(words) - 1 and core in small_words and not previous_ended_colon:
            cased = core
        else:
            cased = _case_title_part(core, force=True)
        output.append(f"{prefix}{cased}{suffix}")
        previous_ended_colon = suffix.endswith(":") or core.endswith(":")
    return " ".join(output)


def _case_title_part(part: str, force: bool = False) -> str:
    if not part:
        return part
    return part[:1].upper() + part[1:] if force else part


def _renumber_slides(slides: list[Slide]) -> list[Slide]:
    return [slide.model_copy(update={"id": f"slide_{index + 1:02d}"}) for index, slide in enumerate(slides)]


def _ensure_slide_content(
    slide: Slide,
    planned: PlannedSlide,
    argument_graph: ArgumentGraph,
    warnings: list[str],
    contexts: list[RerankResult],
) -> Slide:
    if slide.role == "title":
        return slide

    nodes = _matching_nodes(slide.role, argument_graph)
    fallback_source_ids = list(slide.source_ids)
    if not fallback_source_ids and nodes:
        fallback_source_ids = [nodes[0].id, *nodes[0].evidence_chunk_ids]

    slide_warnings = [warning for warning in slide.warnings if warning != "Slide has no bullets."]

    context_bullets = _context_bullets_for_role(slide.role, contexts, planned.target_evidence)

    if slide.bullets:
        bullets = _ground_bullets(slide.bullets, fallback_source_ids)
        bullets = _enrich_bullets(slide.role, bullets, nodes)
        bullets = _merge_bullets(bullets, context_bullets, _MIN_BULLETS_BY_ROLE.get(slide.role, 3))
        source_ids = list(
            dict.fromkeys([*slide.source_ids, *(sid for bullet in bullets for sid in bullet.source_ids)]),
        )
        main_message = _main_message_from_evidence(slide.main_message, planned, nodes, bullets)
        return slide.model_copy(
            update={
                "main_message": main_message,
                "bullets": bullets,
                "source_ids": source_ids,
                "warnings": slide_warnings,
            },
        )

    if not nodes and not context_bullets:
        main_message = _main_message_from_evidence(slide.main_message, planned, [], [])
        return slide.model_copy(update={"main_message": main_message, "warnings": slide_warnings})

    bullets = _enrich_bullets(slide.role, [], nodes)
    bullets = _merge_bullets(bullets, context_bullets, _MIN_BULLETS_BY_ROLE.get(slide.role, 3))
    source_ids = list(
        dict.fromkeys([*slide.source_ids, *(sid for bullet in bullets for sid in bullet.source_ids)]),
    )
    main_message = _main_message_from_evidence(slide.main_message, planned, nodes, bullets)
    slide_warnings.append("Filled empty generated slide from ArgumentGraph evidence.")
    warnings.append(f"Filled empty slide {slide.id} from ArgumentGraph evidence.")
    return slide.model_copy(
        update={
            "main_message": main_message,
            "bullets": bullets,
            "source_ids": source_ids,
            "warnings": slide_warnings,
        },
    )


def _enrich_bullets(
    role: str,
    bullets: list[SlideBullet],
    nodes: list[ArgumentNode],
) -> list[SlideBullet]:
    target = _MIN_BULLETS_BY_ROLE.get(role, 3)
    if len(bullets) >= target or not nodes:
        return bullets[:_MAX_BULLETS]

    enriched = list(bullets)
    seen = {_normalized_for_dedupe(bullet.text) for bullet in enriched}
    for node in nodes:
        source_ids = [node.id, *node.evidence_chunk_ids]
        for point in _evidence_points(node.text):
            normalized = _normalized_for_dedupe(point)
            if not normalized or normalized in seen:
                continue
            enriched.append(
                SlideBullet(
                    text=point,
                    source_ids=source_ids,
                    confidence=node.confidence,
                ),
            )
            seen.add(normalized)
            if len(enriched) >= target:
                return enriched[:_MAX_BULLETS]
    return enriched[:_MAX_BULLETS]


def _merge_bullets(
    primary: list[SlideBullet],
    secondary: list[SlideBullet],
    target: int,
) -> list[SlideBullet]:
    merged: list[SlideBullet] = []
    seen: set[str] = set()
    for bullet in [*primary, *secondary]:
        normalized = _normalized_for_dedupe(bullet.text)
        if not normalized or normalized in seen:
            continue
        merged.append(bullet)
        seen.add(normalized)
        if len(merged) >= max(target, min(_MAX_BULLETS, target + 1)):
            break
    return merged[:_MAX_BULLETS]


def _context_bullets_for_role(
    role: str,
    contexts: list[RerankResult],
    target_evidence: list[str] | None = None,
) -> list[SlideBullet]:
    target_ids = set(target_evidence or [])
    selected_contexts = [context for context in contexts if context.chunk_id in target_ids]
    if not selected_contexts:
        selected_contexts = contexts
    scored = [
        (context, _context_role_score(role, context))
        for context in selected_contexts
        if context.text.strip() and _context_allowed_for_role(role, context)
    ]
    threshold = 0.0 if target_ids else 0.75
    scored = [(context, score) for context, score in scored if score > threshold]
    scored.sort(key=lambda item: item[1], reverse=True)

    bullets: list[SlideBullet] = []
    seen: set[str] = set()
    for context, _score in scored[:6]:
        for point in _evidence_points(context.text):
            normalized = _normalized_for_dedupe(point)
            if not normalized or normalized in seen:
                continue
            bullets.append(
                SlideBullet(
                    text=_compact_text(point, max_chars=_MAX_BULLET_CHARS),
                    source_ids=[context.chunk_id],
                    confidence=context.combined_score,
                ),
            )
            seen.add(normalized)
            if len(bullets) >= _MAX_BULLETS:
                return bullets
    return bullets


def _context_allowed_for_role(role: str, context: RerankResult) -> bool:
    if role != "limitations":
        return True
    text = context.text.lower()
    section = (context.section_title or "").lower()
    query = context.metadata.get("retrieval_query", "").lower()
    limitation_terms = (
        "limitation",
        "however",
        "cost",
        "failure",
        "degrade",
        "redundancy",
        "trade-off",
        "tradeoff",
        "hurt",
        "artifact",
    )
    return any(term in text or term in section or term in query for term in limitation_terms)


def _context_role_score(role: str, context: RerankResult) -> float:
    text = context.text.lower()
    section = (context.section_title or "").lower()
    query = context.metadata.get("retrieval_query", "").lower()
    seed = context.metadata.get("context_seed", "").lower()
    score = context.combined_score

    terms = {
        "problem": ("problem", "challenge", "task", "demands", "difficulty", "gap"),
        "gap": ("gap", "challenge", "limitation", "existing methods", "however"),
        "contribution": ("contribution", "propose", "framework", "novel", "relayformer"),
        "method": ("method", "propose", "attention", "token", "global-local", "glra"),
        "architecture": ("architecture", "module", "submodule", "token", "attention", "decoder", "glra"),
        "experiments": ("experiment", "dataset", "benchmark", "training", "implementation", "setup"),
        "results": ("result", "outperform", "improve", "f1", "benchmark", "ablation", "performance"),
        "limitations": ("limitation", "however", "cost", "failure", "degrade", "redundancy", "caveat"),
        "takeaway": ("conclusion", "overall", "demonstrate", "unified", "scalable", "effective"),
    }
    for term in terms.get(role, (role,)):
        if term in text:
            score += 0.25
        if term in section:
            score += 0.35
        if term in query or term in seed:
            score += 0.45

    section_boosts = {
        "problem": ("abstract", "introduction"),
        "gap": ("abstract", "introduction"),
        "contribution": ("abstract", "introduction"),
        "method": ("method", "input unification", "global-local"),
        "architecture": ("method", "global-local", "input unification"),
        "experiments": ("experiment", "dataset", "implementation"),
        "results": ("result", "ablation", "flops", "robust"),
        "limitations": ("ablation", "complexity", "limitation", "conclusion"),
        "takeaway": ("conclusion", "abstract"),
    }
    if any(term in section for term in section_boosts.get(role, ())):
        score += 0.75
    if any(term in section for term in ("references", "bibliography")):
        score -= 5.0
    return score


def _main_message_from_evidence(
    current: str | None,
    planned: PlannedSlide,
    nodes: list[ArgumentNode],
    bullets: list[SlideBullet],
) -> str:
    if current and not _is_internal_goal_message(current) and current != planned.goal:
        return current
    if bullets:
        return bullets[0].text
    if nodes:
        return _compact_text(nodes[0].text, max_chars=150)
    return _title_from_role(_planned_role(planned))


def _evidence_points(text: str) -> list[str]:
    compact = " ".join(text.split())
    compact = compact.replace("…", "")
    compact = compact.replace("Fig. ", "Figure ").replace("Fig.", "Figure")
    compact = compact.replace("vs. ", "vs ").replace("e.g. ", "for example ").replace("i.e. ", "that is ")
    raw_parts = []
    for sentence in compact.split(". "):
        raw_parts.extend(part for part in sentence.split("; "))

    points: list[str] = []
    for part in raw_parts:
        cleaned = part.strip(" .")
        if not cleaned or len(cleaned.split()) < 4 or _looks_truncated(cleaned):
            continue
        compacted = _compact_text(cleaned, max_chars=90)
        if len(compacted.split()) < 4 or _looks_truncated(compacted):
            continue
        points.append(compacted)
    return points


def _compact_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact.strip(" ,;:")

    # Prefer complete clauses. If no clean boundary exists, keep the full sentence
    # and let density validation warn instead of showing a broken fragment.
    window = compact[:max_chars].rstrip()
    for separator in (". ", "; ", " - ", ", "):
        pos = window.rfind(separator)
        if pos >= max(48, int(max_chars * 0.6)):
            return window[:pos].strip(" ,;:.-")
    return compact.strip(" ,;:")


def _looks_truncated(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(("[", "(", "/", "-")):
        return True
    last = stripped.split()[-1].strip(".,;:)]}")
    if last.lower() in {
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "to",
        "with",
        "by",
        "for",
        "in",
        "our",
        "local",
        "global",
        "visual",
    }:
        return True
    allowed_short = {"F1", "F2", "VML", "GLR", "GLRA", "n=1", "n=2", "n=3"}
    if last not in allowed_short and last.isalpha() and len(last) <= 3 and last.islower():
        return True
    if last not in allowed_short and last.isalpha() and len(last) <= 4 and not last.islower() and not last.isupper():
        return True
    return False


def _is_internal_goal_message(text: str) -> bool:
    normalized = text.strip().lower().rstrip(".")
    return normalized in _INTERNAL_GOAL_MESSAGES


def _normalized_for_dedupe(text: str) -> str:
    return " ".join(text.lower().split())[:90]


def _fallback_slide_ir(
    deck_plan: DeckPlan,
    argument_graph: ArgumentGraph,
    warnings: list[str],
    contexts: list[RerankResult] | None = None,
    paper_ir: PaperIR | None = None,
) -> SlideIR:
    used_ids: set[str] = set()
    slides = [
        _fallback_slide_for_plan(planned, argument_graph, used_ids, index, contexts or [])
        for index, planned in enumerate(deck_plan.slides)
    ]
    if not slides:
        warnings.append("DeckPlan has no planned slides; created a minimal title slide.")
        slides.append(_minimal_title_slide(argument_graph, used_ids))
    slides = _attach_paper_visuals(slides, paper_ir, contexts or [])
    slides = [_polish_slide(slide, argument_graph) for slide in slides]
    warnings.append("Fallback SlideIR was generated from DeckPlan and ArgumentGraph.")
    return SlideIR(
        paper_title=_clean_paper_title(argument_graph.paper_title),
        audience=deck_plan.audience,
        duration_minutes=deck_plan.duration_minutes,
        slides=slides,
        warnings=warnings,
    )


def _fallback_slide_for_plan(
    planned: PlannedSlide,
    argument_graph: ArgumentGraph,
    used_ids: set[str],
    index: int,
    contexts: list[RerankResult] | None = None,
) -> Slide:
    role = _planned_role(planned)
    slide_id = planned.id or f"slide_{index + 1:02d}"
    slide_id = _unique_id(slide_id, used_ids)
    used_ids.add(slide_id)
    if role == "title":
        title = _clean_paper_title(argument_graph.paper_title) or "Untitled Paper"
        return Slide(
            id=slide_id,
            role="title",
            title=title,
            main_message=title,
            layout="title",
            bullets=[],
            visuals=[],
            speaker_notes=None,
            source_ids=list(dict.fromkeys(planned.target_evidence)),
            warnings=["Fallback title slide generated."],
        )

    nodes = _matching_nodes(role, argument_graph)
    bullets: list[SlideBullet] = []
    bullets = _enrich_bullets(role, bullets, nodes)
    bullets = _merge_bullets(
        bullets,
        _context_bullets_for_role(role, contexts or [], planned.target_evidence),
        _MIN_BULLETS_BY_ROLE.get(role, 3),
    )
    title = _title_for_planned_slide(role, planned, contexts or [])
    main_message = _main_message_from_evidence(None, planned, nodes, bullets)
    source_ids = list(
        dict.fromkeys(source_id for bullet in bullets for source_id in bullet.source_ids),
    )
    return Slide(
        id=slide_id,
        role=role,
        title=title,
        main_message=main_message,
        layout=_planned_layout(planned) or "bullets",
        bullets=bullets,
        visuals=[],
        speaker_notes=f"Discuss the {role} using the cited evidence.",
        source_ids=source_ids,
        warnings=["Fallback slide generated."],
    )


def _minimal_title_slide(argument_graph: ArgumentGraph, used_ids: set[str]) -> Slide:
    slide_id = _unique_id("slide_01", used_ids)
    used_ids.add(slide_id)
    title = _clean_paper_title(argument_graph.paper_title) or "Untitled Paper"
    return Slide(
        id=slide_id,
        role="title",
        title=title,
        main_message=title,
        layout="title",
        bullets=[],
        visuals=[],
        speaker_notes=None,
        source_ids=[],
        warnings=["Minimal title slide generated."],
    )


def _matching_nodes(role: str, argument_graph: ArgumentGraph) -> list[ArgumentNode]:
    role_map = {
        "problem": ["problem"],
        "gap": ["gap"],
        "contribution": ["contribution"],
        "method": ["method"],
        "architecture": ["method"],
        "experiments": ["experiment"],
        "results": ["result"],
        "limitations": ["limitation"],
        "takeaway": ["takeaway"],
        "intuition": ["background", "problem"],
        "title": ["takeaway", "contribution"],
    }
    types = role_map.get(role, [role])
    nodes = [node for node in argument_graph.nodes if node.type in types]
    return nodes


def _planned_role(planned: PlannedSlide | None) -> str:
    if planned is None:
        return "unknown"
    return _enum_or_str(planned.role).lower().strip()


def _planned_layout(planned: PlannedSlide | None) -> str | None:
    if planned is None:
        return None
    layout = _enum_or_str(planned.suggested_layout).lower().strip()
    return layout if layout in _SUPPORTED_LAYOUTS else "bullets"


def _title_for_planned_slide(
    role: str,
    planned: PlannedSlide,
    contexts: list[RerankResult],
) -> str:
    target_ids = set(planned.target_evidence)
    context = next((item for item in contexts if item.chunk_id in target_ids), None)
    if context is not None:
        section = (context.section_title or "").strip()
        if section and len(section) <= 80:
            return section.title()
        if context.metadata.get("retrieval_query"):
            return _title_from_query(role, context.metadata["retrieval_query"])
    return _title_from_role(role)


def _title_from_query(role: str, query: str) -> str:
    normalized = query.strip().lower()
    if "experimental" in normalized:
        return "Experimental Details"
    if "result" in normalized:
        return "Additional Results"
    if "method" in normalized:
        return "Method Details"
    if "limitation" in normalized:
        return "Trade-offs and Caveats"
    return _title_from_role(role)


def _title_from_role(role: str) -> str:
    titles = {
        "problem": "Problem: Why VML Is Hard",
        "contribution": "Core Contributions",
        "method": "Method Overview",
        "architecture": "Architecture: Global-Local Relay Attention",
        "experiments": "Experimental Setup",
        "results": "Results and Ablations",
        "limitations": "Caveats and Trade-offs",
        "takeaway": "Takeaway",
    }
    if role in titles:
        return titles[role]
    return " ".join(part.capitalize() for part in role.replace("_", " ").split()) or "Slide"


def _unique_id(candidate: str, used_ids: set[str]) -> str:
    if candidate not in used_ids:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used_ids:
        suffix += 1
    return f"{candidate}_{suffix}"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_clamped_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def _enum_or_str(value: object) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)
