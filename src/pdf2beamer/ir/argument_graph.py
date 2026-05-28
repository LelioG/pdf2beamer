"""Pydantic models for scientific argument graphs."""

from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from pdf2beamer._compat import StrEnum


class ArgumentNodeType(StrEnum):
    """Supported scientific logic node types."""

    PROBLEM = "problem"
    GAP = "gap"
    CONTRIBUTION = "contribution"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULT = "result"
    LIMITATION = "limitation"
    TAKEAWAY = "takeaway"
    BACKGROUND = "background"
    UNKNOWN = "unknown"

    # Compatibility aliases for the initial scaffold.
    EXPERIMENTS = "experiment"
    RESULTS = "result"
    LIMITATIONS = "limitation"


class ArgumentRelationType(StrEnum):
    """Supported directed edge relation types."""

    MOTIVATES = "motivates"
    REFINED_BY = "refined_by"
    ADDRESSED_BY = "addressed_by"
    IMPLEMENTED_BY = "implemented_by"
    EVALUATED_BY = "evaluated_by"
    VALIDATED_BY = "validated_by"
    LIMITED_BY = "limited_by"
    SUPPORTS = "supports"
    SUMMARIZES = "summarizes"
    RELATED_TO = "related_to"


_ALLOWED_NODE_TYPES = {item.value for item in ArgumentNodeType}
_ALLOWED_RELATION_TYPES = {item.value for item in ArgumentRelationType}


class ArgumentNode(BaseModel):
    """A grounded scientific-logic claim."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    text: str = Field(min_length=1)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> str:
        normalized = _enum_or_str(value).lower().strip()
        return normalized if normalized in _ALLOWED_NODE_TYPES else ArgumentNodeType.UNKNOWN.value


class ArgumentEdge(BaseModel):
    """Directed relation between two argument nodes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(validation_alias=AliasChoices("source", "source_node_id"))
    target: str = Field(validation_alias=AliasChoices("target", "target_node_id"))
    relation: str = Field(validation_alias=AliasChoices("relation", "relation_type"))
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("relation", mode="before")
    @classmethod
    def normalize_relation(cls, value: object) -> str:
        normalized = _enum_or_str(value).lower().strip()
        if normalized in _ALLOWED_RELATION_TYPES:
            return normalized
        return ArgumentRelationType.RELATED_TO.value

    @property
    def source_node_id(self) -> str:
        """Compatibility alias for the initial scaffold."""

        return self.source

    @property
    def target_node_id(self) -> str:
        """Compatibility alias for the initial scaffold."""

        return self.target

    @property
    def relation_type(self) -> str:
        """Compatibility alias for the initial scaffold."""

        return self.relation


class ArgumentGraph(BaseModel):
    """Structured representation of a paper's scientific logic."""

    model_config = ConfigDict(extra="forbid")

    paper_title: str | None = None
    nodes: list[ArgumentNode] = Field(default_factory=list)
    edges: list[ArgumentEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def get_nodes_by_type(self, node_type: str) -> list[ArgumentNode]:
        """Return all nodes matching a normalized type."""

        normalized = node_type.lower().strip()
        return [node for node in self.nodes if node.type == normalized]

    def has_node_type(self, node_type: str) -> bool:
        """Return whether at least one node of a given type exists."""

        return bool(self.get_nodes_by_type(node_type))

    def save_json(self, path: str | Path) -> None:
        """Serialize the graph to JSON."""

        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ArgumentGraph":
        """Load a graph from JSON."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _enum_or_str(value: object) -> str:
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)
