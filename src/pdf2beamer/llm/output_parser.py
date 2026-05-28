"""Parsing helpers for structured generation outputs."""

import json
from typing import Any


def extract_json_object(
    text: str,
    required_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Parse or extract the best JSON object from model text.

    The parser never evaluates code. If the model emits reasoning/prose before a
    final JSON object, this picks the largest valid object, optionally requiring
    top-level keys such as ``slides`` or ``nodes``.
    """

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _extract_from_surrounding_text(text, required_keys=required_keys)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    if required_keys and not required_keys.issubset(parsed.keys()):
        raise ValueError("Could not extract a valid JSON object from generator output.")
    return parsed


def _extract_from_surrounding_text(
    text: str,
    required_keys: set[str] | None = None,
) -> object:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, object]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if required_keys and not required_keys.issubset(parsed.keys()):
            continue
        candidates.append((end, index, parsed))

    if not candidates:
        raise ValueError("Could not extract a valid JSON object from generator output.")

    # Prefer the largest object. This avoids returning small nested examples from
    # the model's reasoning when a complete final JSON object follows later.
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
    return candidates[0][2]
