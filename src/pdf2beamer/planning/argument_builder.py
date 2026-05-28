"""Build ArgumentGraph objects from reranked evidence contexts."""

from typing import Any

from pdf2beamer.ir import ArgumentEdge, ArgumentGraph, ArgumentNode
from pdf2beamer.ir.argument_graph import ArgumentNodeType, ArgumentRelationType
from pdf2beamer.llm import BaseGenerator, build_argument_graph_prompt, generate_validated_json
from pdf2beamer.retrieval import RerankResult, is_informative_chunk_text

_REQUIRED_NODE_TYPES = ("problem", "contribution", "method", "result")
_ALLOWED_NODE_TYPES = {item.value for item in ArgumentNodeType}
_ALLOWED_RELATION_TYPES = {item.value for item in ArgumentRelationType}

_ROLE_KEYWORDS = {
    "problem": ("problem", "challenge", "difficult", "aim", "task", "limitation"),
    "contribution": ("propose", "contribution", "framework", "relayformer", "key idea"),
    "method": ("method", "architecture", "attention", "local", "global", "relay"),
    "result": ("experiment", "result", "benchmark", "outperform", "performance", "achieve"),
    "takeaway": ("overall", "demonstrate", "effective", "scalable", "unified", "conclusion"),
}


def build_argument_graph(
    paper_title: str | None,
    contexts: list[RerankResult],
    generator: BaseGenerator,
) -> ArgumentGraph:
    """Build and sanitize an ArgumentGraph from reranked chunks."""

    warnings: list[str] = []
    if not contexts:
        warnings.append("No contexts provided for argument graph construction.")

    prompt = build_argument_graph_prompt(paper_title=paper_title, contexts=contexts)
    try:
        payload = generate_validated_json(
            generator=generator,
            prompt=prompt,
            schema_name="ArgumentGraph",
        )
    except Exception as exc:
        warnings.append(f"ArgumentGraph generation failed; used fallback: {exc}")
        graph = _fallback_graph(paper_title=paper_title, contexts=contexts, warnings=warnings)
    else:
        graph = _graph_from_payload(
            payload=payload,
            paper_title=paper_title,
            contexts=contexts,
            warnings=warnings,
        )
        if not graph.nodes and contexts:
            warnings.append("ArgumentGraph generation returned no nodes; used fallback.")
            graph = _fallback_graph(
                paper_title=paper_title,
                contexts=contexts,
                warnings=warnings,
            )

    for node_type in _REQUIRED_NODE_TYPES:
        if not graph.has_node_type(node_type):
            graph.warnings.append(f"ArgumentGraph has no {node_type} node.")
    return graph


def _fallback_graph(
    *,
    paper_title: str | None,
    contexts: list[RerankResult],
    warnings: list[str],
) -> ArgumentGraph:
    role_queries = [
        ("problem", "problem"),
        ("contribution", "contribution"),
        ("method", "method"),
        ("result", "result"),
        ("takeaway", "takeaway"),
    ]
    nodes: list[ArgumentNode] = []
    used_chunks: set[str] = set()
    for node_type, keyword in role_queries:
        context = _best_context_for_keyword(contexts, node_type, keyword, used_chunks)
        if context is None:
            continue
        used_chunks.add(context.chunk_id)
        nodes.append(
            ArgumentNode(
                id=f"{node_type}_1",
                type=node_type,
                text=_compact_text(context.text),
                evidence_chunk_ids=[context.chunk_id],
                source_pages=context.source_pages,
                confidence=max(0.3, min(0.7, context.combined_score)),
            ),
        )
    edges: list[ArgumentEdge] = []
    for source, target, relation in (
        ("problem_1", "contribution_1", "addressed_by"),
        ("contribution_1", "method_1", "implemented_by"),
        ("method_1", "result_1", "evaluated_by"),
        ("result_1", "takeaway_1", "summarizes"),
    ):
        if any(node.id == source for node in nodes) and any(node.id == target for node in nodes):
            edges.append(
                ArgumentEdge(source=source, target=target, relation=relation, confidence=0.5),
            )
    warnings.append("Fallback ArgumentGraph was generated from reranked contexts.")
    return ArgumentGraph(paper_title=paper_title, nodes=nodes, edges=edges, warnings=warnings)


def _best_context_for_keyword(
    contexts: list[RerankResult],
    node_type: str,
    keyword: str,
    used_chunks: set[str],
) -> RerankResult | None:
    candidates = [
        context
        for context in contexts
        if context.chunk_id not in used_chunks
        and is_informative_chunk_text(context.text, section_title=context.section_title)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda context: _context_role_score(context, node_type, keyword),
        reverse=True,
    )
    return candidates[0]


def _context_role_score(context: RerankResult, node_type: str, keyword: str) -> tuple[float, int]:
    text = context.text.lower()
    section = (context.section_title or "").lower()
    query = context.metadata.get("retrieval_query", "").lower()
    keywords = _ROLE_KEYWORDS.get(node_type, (keyword,))
    keyword_hits = sum(1 for item in keywords if item in text or item in section or item in query)

    section_bonus = 0.0
    if "abstract" in section:
        section_bonus += 0.35
    if "introduction" in section:
        section_bonus += 0.25
    if node_type == "result" and any(
        item in section for item in ("result", "experiment", "evaluation", "ablation", "robust")
    ):
        section_bonus += 0.45
    if node_type in {"problem", "contribution"} and any(
        item in section for item in ("experiment", "result", "reference")
    ):
        section_bonus -= 0.35
    if any(item in section for item in ("references", "bibliography")):
        section_bonus -= 2.0

    query_bonus = 0.0
    if keyword in query or node_type in query:
        query_bonus += 0.45
    if node_type == "result" and "key result" in query:
        query_bonus += 0.25
    if node_type == "takeaway" and "conclusion" in query:
        query_bonus += 0.25

    length_bonus = min(len(text.split()) / 120.0, 0.25)
    score = context.combined_score + keyword_hits * 0.2 + section_bonus + query_bonus + length_bonus
    return score, -len(text)


def _compact_text(text: str, max_chars: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _graph_from_payload(
    *,
    payload: dict[str, Any],
    paper_title: str | None,
    contexts: list[RerankResult],
    warnings: list[str],
) -> ArgumentGraph:
    context_by_id = {context.chunk_id: context for context in contexts}
    nodes, id_map = _sanitize_nodes(payload.get("nodes", []), context_by_id, warnings)
    edges = _sanitize_edges(payload.get("edges", []), id_map, {node.id for node in nodes}, warnings)
    title = payload.get("paper_title") if isinstance(payload.get("paper_title"), str) else None
    return ArgumentGraph(
        paper_title=title or paper_title,
        nodes=nodes,
        edges=edges,
        warnings=warnings,
    )


def _sanitize_nodes(
    raw_nodes: Any,
    context_by_id: dict[str, RerankResult],
    warnings: list[str],
) -> tuple[list[ArgumentNode], dict[str, str]]:
    if not isinstance(raw_nodes, list):
        warnings.append("Generator output did not contain a nodes list.")
        return [], {}

    nodes: list[ArgumentNode] = []
    used_ids: set[str] = set()
    id_map: dict[str, str] = {}

    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            warnings.append(f"Skipped non-object node at index {index}.")
            continue
        text = str(raw_node.get("text") or "").strip()
        if not text:
            warnings.append(f"Skipped empty node at index {index}.")
            continue

        original_id = str(raw_node.get("id") or f"node_{index}").strip() or f"node_{index}"
        node_id = _unique_id(original_id, used_ids)
        if node_id != original_id:
            warnings.append(f"Duplicate node id {original_id!r} renamed to {node_id!r}.")
        id_map.setdefault(original_id, node_id)
        used_ids.add(node_id)

        evidence_ids = _string_list(raw_node.get("evidence_chunk_ids"))
        for evidence_id in evidence_ids:
            if evidence_id not in context_by_id:
                warnings.append(f"Node {node_id} references missing evidence chunk {evidence_id}.")

        source_pages = _int_list(raw_node.get("source_pages"))
        if not source_pages:
            source_pages = _pages_from_evidence(evidence_ids, context_by_id)

        node_type = _normalize_node_type(raw_node.get("type"))
        confidence = _clamp(raw_node.get("confidence"), default=0.5)
        nodes.append(
            ArgumentNode(
                id=node_id,
                type=node_type,
                text=text,
                evidence_chunk_ids=evidence_ids,
                source_pages=source_pages,
                confidence=confidence,
            ),
        )
    return nodes, id_map


def _sanitize_edges(
    raw_edges: Any,
    id_map: dict[str, str],
    node_ids: set[str],
    warnings: list[str],
) -> list[ArgumentEdge]:
    if not isinstance(raw_edges, list):
        warnings.append("Generator output did not contain an edges list.")
        return []

    edges: list[ArgumentEdge] = []
    for index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, dict):
            warnings.append(f"Skipped non-object edge at index {index}.")
            continue
        source = str(raw_edge.get("source") or raw_edge.get("source_node_id") or "").strip()
        target = str(raw_edge.get("target") or raw_edge.get("target_node_id") or "").strip()
        source = id_map.get(source, source)
        target = id_map.get(target, target)
        if source not in node_ids or target not in node_ids:
            warnings.append(f"Removed edge with missing endpoint: {source} -> {target}.")
            continue
        relation = _normalize_relation(raw_edge.get("relation") or raw_edge.get("relation_type"))
        edges.append(
            ArgumentEdge(
                source=source,
                target=target,
                relation=relation,
                confidence=_clamp(raw_edge.get("confidence"), default=0.5),
            ),
        )
    return edges


def _unique_id(candidate: str, used_ids: set[str]) -> str:
    if candidate not in used_ids:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used_ids:
        suffix += 1
    return f"{candidate}_{suffix}"


def _normalize_node_type(value: object) -> str:
    normalized = _enum_or_str(value).lower().strip()
    return normalized if normalized in _ALLOWED_NODE_TYPES else ArgumentNodeType.UNKNOWN.value


def _normalize_relation(value: object) -> str:
    normalized = _enum_or_str(value).lower().strip()
    if normalized in _ALLOWED_RELATION_TYPES:
        return normalized
    return ArgumentRelationType.RELATED_TO.value


def _pages_from_evidence(
    evidence_ids: list[str],
    context_by_id: dict[str, RerankResult],
) -> list[int]:
    pages: list[int] = []
    for evidence_id in evidence_ids:
        context = context_by_id.get(evidence_id)
        if context is None:
            continue
        pages.extend(context.source_pages)
    return list(dict.fromkeys(pages))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    ints: list[int] = []
    for item in value:
        try:
            ints.append(int(item))
        except (TypeError, ValueError):
            continue
    return list(dict.fromkeys(ints))


def _clamp(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _enum_or_str(value: object) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)
