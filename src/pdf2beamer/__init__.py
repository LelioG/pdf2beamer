"""Local-first PDF-to-Beamer conversion package."""

from pdf2beamer.beamer import BeamerRenderer, compile_latex
from pdf2beamer.config import PipelineConfig
from pdf2beamer.diagnostics import PDFDiagnostics, diagnose_pdf
from pdf2beamer.errors import (
    InvalidModelConfigurationError,
    LocalModelInferenceError,
    LocalModelLoadError,
    OptionalDependencyNotInstalledError,
    Pdf2BeamerError,
)
from pdf2beamer.model_factory import create_embedder, create_generator, create_reranker
from pdf2beamer.pipeline import GenerationResult, PdfToBeamerPipeline
from pdf2beamer.quality import QualityReport
from pdf2beamer.validators import validate_slide_ir

__all__ = [
    "BeamerRenderer",
    "GenerationResult",
    "InvalidModelConfigurationError",
    "LocalModelInferenceError",
    "LocalModelLoadError",
    "OptionalDependencyNotInstalledError",
    "PDFDiagnostics",
    "Pdf2BeamerError",
    "PdfToBeamerPipeline",
    "PipelineConfig",
    "QualityReport",
    "compile_latex",
    "create_embedder",
    "create_generator",
    "create_reranker",
    "diagnose_pdf",
    "validate_slide_ir",
]
