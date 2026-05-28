"""Deterministic DeckPlan construction from ArgumentGraph and retrieved contexts."""

from pdf2beamer.ir import ArgumentGraph
from pdf2beamer.ir.deck_plan import DeckPlan, PlannedSlide, SlideLayout, SlideRole
from pdf2beamer.retrieval import RerankResult

_BASE_ROLE_ORDER = [
    SlideRole.TITLE,
    SlideRole.PROBLEM,
    SlideRole.CONTRIBUTION,
    SlideRole.METHOD,
    SlideRole.ARCHITECTURE,
    SlideRole.EXPERIMENTS,
    SlideRole.RESULTS,
    SlideRole.LIMITATIONS,
    SlideRole.TAKEAWAY,
]
_DETAIL_ROLES = {
    SlideRole.METHOD,
    SlideRole.ARCHITECTURE,
    SlideRole.EXPERIMENTS,
    SlideRole.RESULTS,
    SlideRole.LIMITATIONS,
}


def build_deck_plan(
    argument_graph: ArgumentGraph,
    duration_minutes: int,
    audience: str,
    max_slides: int | None = None,
    contexts: list[RerankResult] | None = None,
) -> DeckPlan:
    """Create a deterministic deck plan.

    There is no fixed internal slide cap. If many retrieved contexts carry
    distinct technical content, they become extra detail slides. The only cap is
    the explicit user/config ``max_slides`` value.
    """

    specs = _base_specs(argument_graph, duration_minutes=duration_minutes)
    specs = _insert_context_detail_specs(specs, contexts or [])
    if max_slides is not None:
        specs = specs[:max_slides]

    slides = [
        PlannedSlide(
            id=f"slide_{index + 1:02d}",
            role=role,
            goal=_goal_for_role(role),
            target_evidence=target_evidence
            or [node.id for node in argument_graph.nodes if _node_matches_role(node.type, role)],
            suggested_layout=_layout_for_role(role),
            expected_content_type=content_type or _content_type_for_role(role),
        )
        for index, (role, target_evidence, content_type) in enumerate(specs)
    ]
    return DeckPlan(
        duration_minutes=duration_minutes,
        audience=audience,
        slide_count=len(slides),
        slides=slides,
    )


def _base_specs(
    argument_graph: ArgumentGraph,
    duration_minutes: int,
) -> list[tuple[SlideRole, list[str], str | None]]:
    has_method = any(node.type == "method" for node in argument_graph.nodes)
    has_result = any(node.type in {"result", "experiment"} for node in argument_graph.nodes)
    has_limitation = any(node.type == "limitation" for node in argument_graph.nodes)

    roles = [SlideRole.TITLE]
    for role in _BASE_ROLE_ORDER[1:]:
        if any(_node_matches_role(node.type, role) for node in argument_graph.nodes):
            roles.append(role)
        elif role == SlideRole.ARCHITECTURE and has_method:
            roles.append(role)
        elif role == SlideRole.EXPERIMENTS and has_result:
            roles.append(role)
        elif role == SlideRole.LIMITATIONS and (has_limitation or duration_minutes >= 8):
            roles.append(role)

    if len(roles) <= 1:
        roles = [SlideRole.TITLE, SlideRole.TAKEAWAY]
    return [(role, [], None) for role in roles]


def _insert_context_detail_specs(
    specs: list[tuple[SlideRole, list[str], str | None]],
    contexts: list[RerankResult],
) -> list[tuple[SlideRole, list[str], str | None]]:
    if not contexts:
        return specs

    detail_specs: list[tuple[SlideRole, list[str], str | None]] = []
    seen_chunks: set[str] = set()
    for context in contexts:
        role = _role_for_context(context)
        if role not in _DETAIL_ROLES or context.chunk_id in seen_chunks:
            continue
        seen_chunks.add(context.chunk_id)
        detail_specs.append((role, [context.chunk_id], f"detail:{context.chunk_id}"))

    if not detail_specs:
        return specs

    output: list[tuple[SlideRole, list[str], str | None]] = []
    inserted = False
    for spec in specs:
        role = spec[0]
        if role == SlideRole.TAKEAWAY and not inserted:
            output.extend(detail_specs)
            inserted = True
        output.append(spec)
    if not inserted:
        output.extend(detail_specs)
    return output


def _role_for_context(context: RerankResult) -> SlideRole:
    query = context.metadata.get("retrieval_query", "").lower()
    seed = context.metadata.get("context_seed", "").lower()
    section = (context.section_title or "").lower()
    text = context.text.lower()
    joined = " ".join((query, seed, section, text[:300]))

    if any(term in joined for term in ("limitation", "degrade", "redundancy", "trade-off", "tradeoff", "hurt")):
        return SlideRole.LIMITATIONS
    if any(term in joined for term in ("experimental setup", "implementation", "dataset", "training")):
        return SlideRole.EXPERIMENTS
    if any(term in joined for term in ("key results", "result", "ablation", "benchmark", "performance", "f1")):
        return SlideRole.RESULTS
    if any(term in joined for term in ("architecture", "global-local", "glra", "relay attention", "token")):
        return SlideRole.ARCHITECTURE
    if any(term in joined for term in ("proposed method", "method", "approach")):
        return SlideRole.METHOD
    return SlideRole.APPENDIX


def _node_matches_role(node_type: str, role: SlideRole) -> bool:
    mapping = {
        SlideRole.PROBLEM: {"problem"},
        SlideRole.GAP: {"gap"},
        SlideRole.CONTRIBUTION: {"contribution"},
        SlideRole.INTUITION: {"background", "problem"},
        SlideRole.METHOD: {"method"},
        SlideRole.ARCHITECTURE: {"method"},
        SlideRole.EXPERIMENTS: {"experiment", "result"},
        SlideRole.RESULTS: {"result"},
        SlideRole.LIMITATIONS: {"limitation"},
        SlideRole.TAKEAWAY: {"takeaway"},
        SlideRole.TITLE: {"takeaway", "contribution"},
    }
    return node_type in mapping.get(role, {role.value})


def _goal_for_role(role: SlideRole) -> str:
    return {
        SlideRole.TITLE: "Introduce the paper.",
        SlideRole.PROBLEM: "Explain the main problem.",
        SlideRole.GAP: "Explain the research gap.",
        SlideRole.CONTRIBUTION: "Summarize the contribution.",
        SlideRole.INTUITION: "Explain the intuition.",
        SlideRole.METHOD: "Describe the method.",
        SlideRole.ARCHITECTURE: "Detail the architecture.",
        SlideRole.EXPERIMENTS: "Explain the experimental setup.",
        SlideRole.RESULTS: "Present the key results.",
        SlideRole.LIMITATIONS: "Discuss limitations and caveats.",
        SlideRole.TAKEAWAY: "Close with the main takeaway.",
        SlideRole.APPENDIX: "Add supporting details.",
    }.get(role, f"Cover the {role.value}.")


def _layout_for_role(role: SlideRole) -> SlideLayout:
    if role == SlideRole.TITLE:
        return SlideLayout.TITLE
    if role == SlideRole.TAKEAWAY:
        return SlideLayout.CONCLUSION
    if role in {SlideRole.ARCHITECTURE, SlideRole.EXPERIMENTS, SlideRole.RESULTS}:
        return SlideLayout.TWO_COLUMNS
    return SlideLayout.BULLETS


def _content_type_for_role(role: SlideRole) -> str:
    return {
        SlideRole.ARCHITECTURE: "technical_details",
        SlideRole.EXPERIMENTS: "setup_and_protocol",
        SlideRole.RESULTS: "metrics_and_ablation",
        SlideRole.LIMITATIONS: "caveats",
    }.get(role, "bullets")
