"""Beamer theme configuration."""

from pydantic import BaseModel, ConfigDict


class BeamerTheme(BaseModel):
    """Small deterministic Beamer theme configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "clean"
    beamer_theme: str = "Madrid"
    color_theme: str | None = None
    font_theme: str | None = None
    aspect_ratio: str = "169"
    show_navigation: bool = False


def get_theme(name: str) -> BeamerTheme:
    """Return a supported theme by name."""

    normalized = name.strip().lower()
    if normalized == "classic":
        return BeamerTheme(name="classic", beamer_theme="Madrid", aspect_ratio="169")
    if normalized == "minimal":
        return BeamerTheme(name="minimal", beamer_theme="default", aspect_ratio="169")
    return BeamerTheme(name="clean", beamer_theme="Madrid", aspect_ratio="169")
