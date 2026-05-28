"""Deterministic Beamer renderer from SlideIR."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.beamer.latex_escape import escape_latex, escape_latex_preserving_math
from pdf2beamer.beamer.theme import BeamerTheme, get_theme
from pdf2beamer.ir import Slide, SlideIR, SlideVisual


_INTERNAL_MESSAGES = {
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


class BeamerRenderResult(BaseModel):
    """Result of rendering a Beamer project."""

    model_config = ConfigDict(extra="forbid")

    tex_path: Path
    assets_dir: Path | None = None
    warnings: list[str] = Field(default_factory=list)


class BeamerRenderer:
    """Render SlideIR deterministically into a complete Beamer document."""

    def __init__(self, template_dir: Path | None = None, theme: BeamerTheme | None = None) -> None:
        self.template_dir = template_dir
        self.theme = theme or get_theme("clean")

    def render(
        self,
        slide_ir: SlideIR,
        output_dir: str | Path,
        tex_filename: str = "main.tex",
    ) -> BeamerRenderResult:
        """Render SlideIR to a main.tex file."""

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        slides_tex = [self._render_slide(slide, out_dir, warnings) for slide in slide_ir.slides]
        tex = self._document(slide_ir, slides_tex)
        tex_path = out_dir / tex_filename
        tex_path.write_text(tex, encoding="utf-8")
        return BeamerRenderResult(
            tex_path=tex_path,
            assets_dir=out_dir / "assets",
            warnings=warnings,
        )

    def _document(self, slide_ir: SlideIR, slides_tex: list[str]) -> str:
        title = escape_latex_preserving_math(slide_ir.paper_title or "Untitled Paper")
        lines = [
            rf"\documentclass[aspectratio={self.theme.aspect_ratio}]{{beamer}}",
            rf"\usetheme{{{self.theme.beamer_theme}}}",
        ]
        if self.theme.color_theme:
            lines.append(rf"\usecolortheme{{{self.theme.color_theme}}}")
        if self.theme.font_theme:
            lines.append(rf"\usefonttheme{{{self.theme.font_theme}}}")
        if not self.theme.show_navigation:
            lines.append(r"\setbeamertemplate{navigation symbols}{}")
        lines.extend(
            [
                r"\usepackage{graphicx}",
                r"\usepackage{booktabs}",
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
                r"\usepackage{hyperref}",
                r"\definecolor{PdfBeamerBlue}{HTML}{174A7C}",
                r"\definecolor{PdfBeamerAccent}{HTML}{2D7D6F}",
                r"\definecolor{PdfBeamerSoft}{HTML}{EEF4F8}",
                r"\setbeamercolor{frametitle}{fg=white,bg=PdfBeamerBlue}",
                r"\setbeamercolor{title}{fg=PdfBeamerBlue}",
                r"\setbeamercolor{structure}{fg=PdfBeamerAccent}",
                r"\setbeamercolor{block title}{fg=white,bg=PdfBeamerBlue}",
                r"\setbeamercolor{block body}{bg=PdfBeamerSoft,fg=black}",
                r"\setbeamerfont{frametitle}{series=\bfseries,size=\large}",
                r"\setbeamerfont{title}{series=\bfseries,size=\Large}",
                r"\setbeamersize{text margin left=8mm,text margin right=8mm}",
                r"\setbeamertemplate{itemize item}{\raise0.2ex\hbox{\scriptsize$\triangleright$}}",
                r"\setbeamertemplate{itemize subitem}{\raise0.2ex\hbox{\scriptsize$\circ$}}",
                r"\setbeamertemplate{footline}{%",
                r"  \leavevmode%",
                r"  \hbox{%",
                r"    \begin{beamercolorbox}[wd=.78\paperwidth,ht=2.5ex,dp=1.2ex,leftskip=2ex]{author in head/foot}%",
                r"      \usebeamerfont{author in head/foot}\insertshorttitle",
                r"    \end{beamercolorbox}%",
                r"    \begin{beamercolorbox}[wd=.22\paperwidth,ht=2.5ex,dp=1.2ex,center]{date in head/foot}%",
                r"      \usebeamerfont{date in head/foot}\insertframenumber/\inserttotalframenumber",
                r"    \end{beamercolorbox}%",
                r"  }%",
                r"}",
                rf"\title{{{title}}}",
                rf"\hypersetup{{pdftitle={{{title}}}}}",
                r"\begin{document}",
                *slides_tex,
                r"\end{document}",
                "",
            ],
        )
        return "\n".join(lines)

    def _render_slide(self, slide: Slide, output_dir: Path, warnings: list[str]) -> str:
        layout = slide.layout
        if layout == "title":
            return self._title_slide(slide)
        if layout in {"conclusion", "appendix"}:
            return self._conclusion_slide(slide)
        if layout == "two_columns":
            return self._two_column_slide(slide)
        if layout == "figure_left_bullets_right":
            return self._figure_left_slide(slide, output_dir, warnings)
        if layout == "figure_top_bullets_bottom":
            return self._figure_top_slide(slide, output_dir, warnings)
        if layout == "table":
            return self._table_slide(slide, output_dir, warnings)
        if layout != "bullets":
            warnings.append(f"Unknown layout {layout!r}; rendered as bullets for slide {slide.id}.")
        return self._bullet_slide(slide)

    def _title_slide(self, slide: Slide) -> str:
        title = escape_latex_preserving_math(slide.title)
        message = self._visible_message(slide)
        lines = [
            r"\begin{frame}[plain]",
            r"\vspace*{8mm}",
            rf"{{\usebeamerfont{{title}}\usebeamercolor[fg]{{title}} {title}\par}}",
            r"\vspace{5mm}",
            r"\rule{0.22\linewidth}{0.7pt}\par",
        ]
        if message:
            lines.extend(
                [
                    r"\vspace{5mm}",
                    rf"{{\large {message}\par}}",
                ],
            )
        lines.extend([r"\vfill", self._comments(slide), r"\end{frame}"])
        return "\n".join(lines)

    def _bullet_slide(self, slide: Slide) -> str:
        title = escape_latex_preserving_math(slide.title)
        lines = [r"\begin{frame}[t]", rf"\frametitle{{{title}}}", r"\small"]
        message = self._visible_message(slide)
        if message:
            lines.extend([self._message_block(message), r"\vspace{0.25em}"])
        bullet_lines = self._bullet_lines(slide, itemsep="0.45em")
        if bullet_lines:
            lines.extend(bullet_lines)
        elif not message:
            lines.append(r"\vfill")
        lines.extend([self._comments(slide), r"\end{frame}"])
        return "\n".join(lines)

    def _two_column_slide(self, slide: Slide) -> str:
        title = escape_latex_preserving_math(slide.title)
        midpoint = max(1, (len(slide.bullets) + 1) // 2)
        left = slide.bullets[:midpoint]
        right = slide.bullets[midpoint:]
        lines = [r"\begin{frame}[t]", rf"\frametitle{{{title}}}", r"\small"]
        message = self._visible_message(slide)
        if message:
            lines.extend([self._message_block(message), r"\vspace{0.25em}"])
        lines.append(r"\begin{columns}[T,onlytextwidth]")
        for bullets in (left, right):
            lines.extend([r"\column{0.48\textwidth}", r"\begin{itemize}", r"\setlength\itemsep{0.45em}"])
            for bullet in bullets:
                lines.append(rf"\item {escape_latex_preserving_math(bullet.text)}")
            lines.append(r"\end{itemize}")
        lines.extend([r"\end{columns}", self._comments(slide), r"\end{frame}"])
        return "\n".join(lines)

    def _figure_left_slide(self, slide: Slide, output_dir: Path, warnings: list[str]) -> str:
        title = escape_latex_preserving_math(slide.title)
        lines = [r"\begin{frame}[t]", rf"\frametitle{{{title}}}", r"\scriptsize"]
        message = self._visible_message(slide)
        if message:
            lines.extend([self._message_block(message), r"\vspace{0.15em}"])
        visual = slide.visuals[0] if slide.visuals else None
        lines.append(r"\begin{columns}[T,onlytextwidth]")
        lines.append(r"\column{0.46\textwidth}")
        if visual is None:
            warnings.append(f"Slide {slide.id} layout expects a visual but none was provided.")
            lines.append(r"% missing visual")
        else:
            lines.extend(self._visual_lines(visual, output_dir, warnings, max_height=r"0.52\textheight"))
        lines.append(r"\column{0.50\textwidth}")
        lines.extend(self._bullet_lines(slide, itemsep="0.28em"))
        lines.extend([r"\end{columns}", self._comments(slide), r"\end{frame}"])
        return "\n".join(lines)

    def _figure_top_slide(self, slide: Slide, output_dir: Path, warnings: list[str]) -> str:
        title = escape_latex_preserving_math(slide.title)
        lines = [r"\begin{frame}[t]", rf"\frametitle{{{title}}}", r"\scriptsize"]
        message = self._visible_message(slide)
        if message:
            lines.extend([self._message_block(message), r"\vspace{0.15em}"])
        visual = slide.visuals[0] if slide.visuals else None
        if visual is None:
            warnings.append(f"Slide {slide.id} layout expects a visual but none was provided.")
            lines.append(r"% missing visual")
        else:
            lines.extend(self._visual_lines(visual, output_dir, warnings, max_height=r"0.28\textheight"))
        lines.extend(self._bullet_lines(slide, itemsep="0.25em"))
        lines.extend([self._comments(slide), r"\end{frame}"])
        return "\n".join(lines)

    def _table_slide(self, slide: Slide, output_dir: Path, warnings: list[str]) -> str:
        title = escape_latex_preserving_math(slide.title)
        lines = [r"\begin{frame}[t]", rf"\frametitle{{{title}}}", r"\scriptsize"]
        message = self._visible_message(slide)
        if message:
            lines.extend([self._message_block(message), r"\vspace{0.15em}"])
        visual = slide.visuals[0] if slide.visuals else None
        if visual is not None:
            lines.extend(self._visual_lines(visual, output_dir, warnings, max_height=r"0.32\textheight"))
        lines.extend(self._bullet_lines(slide, itemsep="0.25em"))
        lines.extend([self._comments(slide), r"\end{frame}"])
        return "\n".join(lines)

    def _bullet_lines(self, slide: Slide, itemsep: str) -> list[str]:
        if not slide.bullets:
            return []
        lines = [r"\begin{itemize}", rf"\setlength\itemsep{{{itemsep}}}"]
        for bullet in slide.bullets:
            lines.append(rf"\item {escape_latex_preserving_math(bullet.text)}")
        lines.append(r"\end{itemize}")
        return lines

    def _conclusion_slide(self, slide: Slide) -> str:
        title = escape_latex_preserving_math(slide.title)
        lines = [r"\begin{frame}[t]", rf"\frametitle{{{title}}}", r"\small"]
        message = self._visible_message(slide)
        if message:
            lines.extend([self._message_block(message), r"\vspace{0.4em}"])
        if slide.bullets:
            lines.extend([r"\begin{itemize}", r"\setlength\itemsep{0.55em}"])
            for bullet in slide.bullets:
                lines.append(rf"\item {escape_latex_preserving_math(bullet.text)}")
            lines.append(r"\end{itemize}")
        lines.extend([r"\vfill", self._comments(slide), r"\end{frame}"])
        return "\n".join(lines)

    def _visual_lines(
        self,
        visual: SlideVisual,
        output_dir: Path,
        warnings: list[str],
        max_height: str = "0.42\\textheight",
    ) -> list[str]:
        caption = escape_latex_preserving_math(visual.caption or "")
        if visual.type == "equation":
            return self._equation_lines(visual)
        if visual.type == "table":
            return self._table_lines(visual)
        if visual.path:
            path = Path(visual.path)
            full_path, display_path = self._resolve_visual_path(path, output_dir)
            if full_path.exists():
                return [
                    r"\begin{center}",
                    rf"\includegraphics[width=0.96\linewidth,height={max_height},keepaspectratio]{{\detokenize{{{str(display_path)}}}}}",
                    rf"\par{{\scriptsize {caption}}}" if caption else r"",
                    r"\end{center}",
                ]
            warnings.append(f"Visual path not found during rendering: {visual.path}.")
        return [
            rf"% visual omitted: {escape_latex(visual.id)}",
            rf"{{\scriptsize {caption}}}" if caption else r"% no caption",
        ]

    def _resolve_visual_path(self, path: Path, output_dir: Path) -> tuple[Path, Path]:
        if path.is_absolute():
            try:
                return path, path.relative_to(output_dir.resolve())
            except ValueError:
                return path, path
        output_relative = output_dir / path
        if output_relative.exists():
            return output_relative, path
        if path.exists():
            try:
                return path, path.relative_to(output_dir)
            except ValueError:
                return path, path
        return output_relative, path

    def _equation_lines(self, visual: SlideVisual) -> list[str]:
        content = escape_latex_preserving_math(visual.content or visual.caption or "")
        caption = escape_latex_preserving_math(visual.caption or "")
        lines = [r"\begin{block}{Equation}", rf"\centering\small\texttt{{{content}}}", r"\end{block}"]
        if caption:
            lines.append(rf"{{\scriptsize {caption}}}")
        return lines

    def _table_lines(self, visual: SlideVisual) -> list[str]:
        caption = escape_latex_preserving_math(visual.caption or "Table")
        content = visual.content or ""
        lines = [r"\begin{block}{" + caption + "}", r"\scriptsize"]
        for line in content.splitlines()[:8]:
            escaped = escape_latex_preserving_math(line)
            if escaped:
                lines.append(escaped + r"\\")
        lines.append(r"\end{block}")
        return lines

    def _visible_message(self, slide: Slide) -> str:
        message = (slide.main_message or "").strip()
        if not message:
            return ""
        if any("Fallback" in warning for warning in slide.warnings):
            return ""
        if message.lower().rstrip(".") in _INTERNAL_MESSAGES:
            return ""
        if message.strip() == slide.title.strip():
            return ""
        if _looks_like_fragment(message):
            return ""
        for bullet in slide.bullets:
            if _same_or_contained(message, bullet.text):
                return ""
        return escape_latex_preserving_math(message)

    def _message_block(self, message: str) -> str:
        return "\n".join(
            [
                r"\begin{block}{}",
                rf"\normalsize {message}",
                r"\end{block}",
            ],
        )

    def _comments(self, slide: Slide) -> str:
        comments = []
        if slide.source_ids:
            comments.append("% sources: " + ", ".join(slide.source_ids))
        # Speaker notes are intentionally not emitted into main.tex for now.
        # They can contain generator planning language and should not affect rendered decks.
        return "\n".join(comments)


def _looks_like_fragment(text: str) -> bool:
    words = text.split()
    if len(words) < 8 and text.rstrip().endswith(("modalities", "diversity", "benchmarks")):
        return True
    last = words[-1].strip(".,;:()[]{}") if words else ""
    return last.lower() in {"local", "global", "visual"}


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
