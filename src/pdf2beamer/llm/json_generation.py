"""Helpers for JSON-only model generation."""

from typing import Any

from pdf2beamer.llm.base import BaseGenerator
from pdf2beamer.llm.output_parser import extract_json_object


def generate_validated_json(
    generator: BaseGenerator,
    prompt: str,
    schema_name: str | None = None,
) -> dict[str, Any]:
    """Call a generator and return a validated JSON object dictionary."""

    try:
        result = generator.generate_json(prompt, schema_name=schema_name)
    except TypeError:
        result = generator.generate_json(prompt)  # type: ignore[call-arg]

    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return extract_json_object(result)
    raise ValueError(f"Generator returned unsupported JSON payload type: {type(result).__name__}.")
