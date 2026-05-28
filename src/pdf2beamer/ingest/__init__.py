"""Extraction adapters for native PDFs."""

from pdf2beamer.ingest.docling_adapter import (
    DoclingAdapter,
    DoclingExtraction,
    DoclingExtractionError,
    DoclingFigure,
    DoclingNotInstalledError,
    DoclingSection,
    DoclingTable,
    DoclingTextItem,
    extract_with_docling,
)
from pdf2beamer.ingest.models import (
    BoundingBox,
    ImageBlock,
    PageExtraction,
    PyMuPDFExtraction,
    TextBlock,
    TextSpan,
)
from pdf2beamer.ingest.pymupdf_adapter import (
    PyMuPDFAdapter,
    PyMuPDFExtractionError,
    extract_with_pymupdf,
)

__all__ = [
    "BoundingBox",
    "DoclingAdapter",
    "DoclingExtraction",
    "DoclingExtractionError",
    "DoclingFigure",
    "DoclingNotInstalledError",
    "DoclingSection",
    "DoclingTable",
    "DoclingTextItem",
    "ImageBlock",
    "PageExtraction",
    "PyMuPDFAdapter",
    "PyMuPDFExtraction",
    "PyMuPDFExtractionError",
    "TextBlock",
    "TextSpan",
    "extract_with_docling",
    "extract_with_pymupdf",
]
