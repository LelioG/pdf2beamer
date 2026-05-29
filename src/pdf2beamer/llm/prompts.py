"""Prompt builders for structured local generation."""

from pdf2beamer.ir import ArgumentGraph
from pdf2beamer.ir.deck_plan import DeckPlan
from pdf2beamer.retrieval import RerankResult


def build_argument_graph_prompt(
    paper_title: str | None,
    contexts: list[RerankResult],
) -> str:
    """Build a deterministic JSON-only prompt for argument extraction."""

    title = paper_title or "Untitled paper"
    lines = [
        "You are extracting a scientific ArgumentGraph from provided chunks.",
        "Use only the provided chunks as evidence.",
        "Do not invent claims, methods, numerical results, or citations.",
        "Every important claim must include evidence_chunk_ids when possible.",
        "Return JSON only. Do not include prose, markdown, or code fences.",
        "Required JSON shape:",
        '{"paper_title": string|null, "nodes": [...], "edges": [...] }',
        "Node types: problem, gap, contribution, method, experiment, result, limitation,",
        "takeaway, background, unknown.",
        "Relations: motivates, refined_by, addressed_by, implemented_by, evaluated_by,",
        "validated_by, limited_by, supports, summarizes, related_to.",
        f"Paper title: {title}",
        "Contexts:",
    ]
    for context in contexts:
        pages = ", ".join(str(page) for page in context.source_pages)
        lines.extend(
            [
                f"[chunk_id: {context.chunk_id}]",
                f"[pages: {pages}]",
                f"[section: {context.section_title or context.section_id or 'unknown'}]",
                "[text]",
                context.text.strip(),
                "[/text]",
            ],
        )
    return "\n".join(lines).strip() + "\n"


def build_slide_ir_prompt(
    deck_plan: DeckPlan,
    argument_graph: ArgumentGraph,
    contexts: list[RerankResult] | None = None,
) -> str:
    """Build a deterministic JSON-only prompt for SlideIR generation."""

    lines = [
        "You are generating SlideIR JSON for a Beamer presentation renderer.",
        "Use only the provided ArgumentGraph nodes and optional contexts.",
        "Do not invent claims, numbers, citations, figures, or results.",
        "Preserve source ids from argument nodes and context chunks.",
        "Write concise slide titles and one clear evidence-backed main_message per slide.",
        "Each non-title slide should contain 3 to 5 grounded bullets when enough evidence exists.",
        "Do not copy planned goals such as Describe the method or Summarize the contribution.",
        "Write short bullets and include speaker notes.",
        "Return JSON only. Do not include prose, markdown, or code fences.",
        "Never return LaTeX. Never return Beamer.",
        "Expected JSON shape:",
        '{"paper_title": string|null, "audience": string, "duration_minutes": number,',
        ' "slides": [{"id": string, "role": string, "title": string,',
        ' "main_message": string, "layout": string, "bullets": [], "visuals": [],',
        ' "speaker_notes": string|null, "source_ids": [], "warnings": []}], "warnings": []}',
        (
            'Visuals may include {"id": string, "type": "figure|table", '
            '"path": string|null, "caption": string|null, '
            '"content": string|null, "source_ids": []}.'
        ),
        (
            "Do not invent visual paths, table contents, or visual contents; "
            "use only provided source ids."
        ),
        f"Paper title: {argument_graph.paper_title or 'Untitled paper'}",
        f"Audience: {deck_plan.audience}",
        f"Duration minutes: {deck_plan.duration_minutes}",
        "Planned slides. Goals are internal planning hints and are intentionally omitted:",
    ]
    for planned in deck_plan.slides:
        lines.extend(
            [
                f"[planned_slide_id: {planned.id}]",
                f"[role: {_prompt_value(planned.role)}]",
                f"[layout: {_prompt_value(planned.suggested_layout)}]",
                f"[target_evidence: {', '.join(planned.target_evidence)}]",
            ],
        )

    lines.append("Argument nodes:")
    for node in argument_graph.nodes:
        pages = ", ".join(str(page) for page in node.source_pages)
        evidence = ", ".join(node.evidence_chunk_ids)
        lines.extend(
            [
                f"[node_id: {node.id}]",
                f"[node_type: {node.type}]",
                f"[evidence_chunk_ids: {evidence}]",
                f"[pages: {pages}]",
                "[node_text]",
                node.text,
                "[/node_text]",
            ],
        )

    if contexts:
        lines.append("Contexts:")
        for context in contexts:
            pages = ", ".join(str(page) for page in context.source_pages)
            lines.extend(
                [
                    f"[chunk_id: {context.chunk_id}]",
                    f"[pages: {pages}]",
                    f"[section: {context.section_title or context.section_id or 'unknown'}]",
                    "[text]",
                    context.text.strip(),
                    "[/text]",
                ],
            )
    return "\n".join(lines).strip() + "\n"


def _prompt_value(value: object) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)
