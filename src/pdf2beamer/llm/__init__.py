"""Local model interfaces and structured-generation helpers."""

from pdf2beamer.llm.base import BaseGenerator, FakeGenerator
from pdf2beamer.llm.json_generation import generate_validated_json
from pdf2beamer.llm.output_parser import extract_json_object
from pdf2beamer.llm.prompts import build_argument_graph_prompt, build_slide_ir_prompt
from pdf2beamer.llm.gguf_generator import LocalGGUFGenerator, LocalNemotronGenerator

__all__ = [
    "BaseGenerator",
    "FakeGenerator",
    "LocalGGUFGenerator",
    "LocalNemotronGenerator",
    "build_argument_graph_prompt",
    "build_slide_ir_prompt",
    "extract_json_object",
    "generate_validated_json",
]
