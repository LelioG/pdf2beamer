"""Local LaTeX compilation wrapper."""

from pathlib import Path
import shutil
import subprocess

from pydantic import BaseModel, ConfigDict, Field


class LatexCompileResult(BaseModel):
    """Structured result from a local LaTeX compiler invocation."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    tex_path: Path
    pdf_path: Path | None = None
    log_path: Path | None = None
    command: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def compile_latex(
    tex_path: str | Path,
    engine: str = "latexmk",
    timeout_seconds: int = 120,
) -> LatexCompileResult:
    """Compile a local LaTeX file with latexmk or tectonic."""

    path = Path(tex_path)
    if not path.exists():
        return LatexCompileResult(
            success=False,
            tex_path=path,
            errors=[f"TeX file not found: {path}"],
        )
    command = _command(engine, path.name)
    executable = shutil.which(command[0])
    if executable is None:
        return LatexCompileResult(
            success=False,
            tex_path=path,
            command=command,
            errors=[f"{command[0]} not found"],
        )
    try:
        completed = subprocess.run(
            command,
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return LatexCompileResult(
            success=False,
            tex_path=path,
            command=command,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            errors=[f"LaTeX compilation timed out after {timeout_seconds} seconds."],
        )
    pdf_path = path.with_suffix(".pdf") if path.with_suffix(".pdf").exists() else None
    log_path = path.with_suffix(".log") if path.with_suffix(".log").exists() else None
    text = "\n".join([completed.stdout or "", completed.stderr or "", _read_log(log_path)])
    errors, warnings = _parse_messages(text)
    success = completed.returncode == 0 and pdf_path is not None
    if completed.returncode != 0 and not errors:
        errors.append(f"LaTeX compiler exited with status {completed.returncode}.")
    return LatexCompileResult(
        success=success,
        tex_path=path,
        pdf_path=pdf_path,
        log_path=log_path,
        command=command,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        warnings=warnings,
        errors=errors,
    )


def _command(engine: str, filename: str) -> list[str]:
    normalized = engine.lower().strip()
    if normalized == "tectonic":
        return ["tectonic", filename]
    return ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", filename]


def _read_log(log_path: Path | None) -> str:
    if log_path is None or not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def _parse_messages(text: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        has_error_token = any(
            token in stripped for token in ("Emergency stop", "Fatal error", "Error")
        )
        if has_error_token or stripped.startswith("!"):
            errors.append(stripped)
        elif any(token in stripped for token in ("Warning", "Overfull", "Underfull")):
            warnings.append(stripped)
    return errors[:50], warnings[:50]
