"""Argument, deck, and slide planning modules."""

from pdf2beamer.planning.argument_builder import build_argument_graph
from pdf2beamer.planning.deck_planner import build_deck_plan
from pdf2beamer.planning.slide_generator import generate_slide_ir

__all__ = ["build_argument_graph", "build_deck_plan", "generate_slide_ir"]
