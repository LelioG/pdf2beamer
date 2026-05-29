"""Deterministic LaTeX escaping helpers."""

_UNICODE_REPLACEMENTS = {
    "✓": r"\checkmark{}",
    "✔": r"\checkmark{}",
    "✗": r"$\times$",
    "✘": r"$\times$",
    "×": r"$\times$",
    "→": r"$\rightarrow$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "≠": r"$\neq$",
    "≈": r"$\approx$",
    "±": r"$\pm$",
    "−": "-",
    "–": "-",
    "—": "-",
    "‑": "-",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "∈": r"$\in$",
    "∑": r"$\sum$",
    "∞": r"$\infty$",
}

_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape common unsafe LaTeX characters in plain text."""

    return "".join(_escape_char(char) for char in str(text))


def _escape_char(char: str) -> str:
    if char in _UNICODE_REPLACEMENTS:
        return _UNICODE_REPLACEMENTS[char]
    return _REPLACEMENTS.get(char, char)


def escape_latex_preserving_math(text: str) -> str:
    """Escape text while preserving simple balanced inline math spans."""

    text = str(text)
    if text.count("$") == 0:
        return escape_latex(text)
    if text.count("$") % 2 == 1:
        return escape_latex(text)

    parts = text.split("$")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            rendered.append(escape_latex(part))
        else:
            rendered.append(f"${part}$")
    return "".join(rendered)
