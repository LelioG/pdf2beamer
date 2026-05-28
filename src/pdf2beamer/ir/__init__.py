"""Intermediate representation models."""

from pdf2beamer.ir.argument_graph import ArgumentEdge, ArgumentGraph, ArgumentNode
from pdf2beamer.ir.deck_plan import DeckPlan
from pdf2beamer.ir.paper_ir import (
    EquationIR,
    FigureIR,
    PaperIR,
    PaperMetadata,
    ParagraphIR,
    SectionIR,
    SourceRef,
    TableIR,
)
from pdf2beamer.ir.quality_report import QualityReport
from pdf2beamer.ir.slide_ir import Slide, SlideBullet, SlideIR, SlideVisual

__all__ = [
    "ArgumentGraph",
    "ArgumentNode",
    "ArgumentEdge",
    "DeckPlan",
    "FigureIR",
    "PaperIR",
    "PaperMetadata",
    "ParagraphIR",
    "QualityReport",
    "SectionIR",
    "SlideIR",
    "SlideVisual",
    "SlideBullet",
    "Slide",
    "SourceRef",
    "TableIR",
    "EquationIR",
]
