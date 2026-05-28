# pdf2beamer

`pdf2beamer` is a local-first Python package for converting native scientific
PDF papers into editable, compilable Beamer presentations.

The project is intentionally structured around inspectable intermediate
representations:

```text
PDF -> PaperIR -> ArgumentGraph -> DeckPlan -> SlideIR -> Beamer
```

The package defines strict Pydantic v2 data models and keeps real Docling, PyMuPDF, Nemotron generation, validation, rendering, and compilation behind local integration points.

## Constraints

- No external API calls.
- No OCR or scanned-PDF fallback.
- Local Nemotron generation plus Qwen embedding/reranking adapters are dependency-injected.
- LLM components generate structured JSON only.
- Beamer is rendered deterministically from `SlideIR`.

## Public API Sketch

```python
from pdf2beamer import PdfToBeamerPipeline, PipelineConfig

config = PipelineConfig(
    model_path="./models/nemotron-3-nano-4b-gguf/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    embedding_model_path="./models/Qwen3-Embedding-0.6B",
    reranker_model_path="./models/Qwen3-Reranker-0.6B",
    duration_minutes=10,
    audience="technical",
    theme="clean",
)

pipeline = PdfToBeamerPipeline(config)
result = pipeline.generate("paper.pdf", "out/")
```

## Local Models

Real model backends are optional. The base package imports and tests without
installing model dependencies, and the library never downloads model files at
runtime.

Install local model dependencies when you want to use real backends:

```bash
pip install -e ".[pdf,models]"
```

Expected local files, auto-detected by `--real-models` when present:

- Generation: `models/nemotron-3-nano-4b-gguf/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`
- Embedding: `models/Qwen3-Embedding-0.6B`
- Reranking: `models/Qwen3-Reranker-0.6B`

You can override paths with `--model`, `--embedding`, or `--reranker`.

Model files are local assets and should not be committed. Store them under
`models/`; `.gitignore` excludes common model-weight formats.

Fake-model command for lightweight local development:

```bash
pdf2beamer generate paper.pdf \
  --use-fake-models \
  --duration 10 \
  --output out/
```

Real local-model command:

```bash
pdf2beamer generate paper.pdf \
  --real-models \
  --duration 10 \
  --audience technical \
  --output out/
```

### Structured GGUF Output

The GGUF generator is exposed as `LocalNemotronGenerator` and loads a local Nemotron instruct GGUF through `llama-cpp-python`.
For `ArgumentGraph` and `SlideIR`, it first tries Instructor's local
`llama-cpp-python` integration:

```python
instructor.patch(
    create=llama.create_chat_completion_openai_v1,
    mode=instructor.Mode.JSON_SCHEMA,
)
```

This path returns Pydantic response models directly and retries validation
failures. If Instructor or the OpenAI-compatible llama.cpp method is unavailable,
the generator falls back to llama.cpp `response_format` JSON schema/object mode,
then to strict JSON parsing.

CLI controls:

```bash
pdf2beamer generate paper.pdf \
  --real-models \
  --instructor \
  --instructor-max-retries 2 \
  --no-compile \
  --output out/
```

Disable Instructor and use llama.cpp response format fallback:

```bash
pdf2beamer generate paper.pdf --real-models --no-instructor --output out/
```



1. `00_pdf_diagnostics.json` - verifie que le PDF a du texte exploitable et liste les avertissements.
2. `01_pymupdf_extraction.json` - extraction native: pages, blocs de texte, images, bounding boxes.
3. `02_docling_logical_extraction_fallback.json` - extraction logique au format Docling. 
4. `03_paper_ir.json` - representation fusionnee du papier: metadata, sections, paragraphes, figures, tables.
5. `04_chunks.json` - decoupage de `PaperIR` en morceaux recuperables.
6. `05_embedding_index.json` - index vectoriel local des chunks.
7. `06_retrieved_contexts.json` - contextes selectionnes pour probleme, contribution, methode, resultats, limites, conclusion.
8. `07_argument_graph.json` - graphe des claims: probleme, contribution, methode, resultats, takeaway.
9. `08_deck_plan.json` - plan de presentation: roles des slides, objectifs, evidences ciblees.
10. `09_slide_ir.json` - contenu structure des slides avant rendu LaTeX.
11. `10_validation_report.json` - validation densite, grounding, figures et LaTeX.
12. `11_main.tex` - source Beamer genere.
13. `12_quality_report.json` - synthese qualite finale.
14. `assets/` - images extraites du PDF et reutilisables dans les slides.
