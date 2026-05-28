"""Command line interface for pdf2beamer."""

from pathlib import Path

import typer

from pdf2beamer.config import PipelineConfig
from pdf2beamer.pipeline import PdfToBeamerPipeline

app = typer.Typer(help="Convert native scientific PDFs into Beamer decks.")


@app.callback()
def main() -> None:
    """Convert native scientific PDFs into Beamer decks."""


@app.command()
def generate(
    pdf_path: Path = typer.Argument(..., metavar="PDF_PATH"),
    output: Path = typer.Option(Path("out"), "--output", "-o", help="Output directory."),
    model: Path | None = typer.Option(None, "--model", help="Local GGUF generator path."),
    embedding: Path | None = typer.Option(None, "--embedding", help="Local embedding model path."),
    reranker: Path | None = typer.Option(None, "--reranker", help="Local reranker model path."),
    duration: int = typer.Option(10, "--duration", min=1),
    audience: str = typer.Option("technical", "--audience"),
    theme: str = typer.Option("clean", "--theme"),
    max_slides: int | None = typer.Option(None, "--max-slides", min=1),
    compile_pdf: bool = typer.Option(True, "--compile/--no-compile"),
    latex_engine: str = typer.Option("latexmk", "--latex-engine"),
    debug: bool = typer.Option(False, "--debug"),
    save_intermediate: bool = typer.Option(True, "--save-intermediate/--no-save-intermediate"),
    use_fake_models: bool = typer.Option(
        True,
        "--use-fake-models/--real-models",
        help="Use deterministic fake models or local real model backends.",
    ),
    generator_backend: str = typer.Option("fake", "--generator-backend"),
    embedder_backend: str = typer.Option("fake", "--embedder-backend"),
    reranker_backend: str = typer.Option("fake", "--reranker-backend"),
    gpu: int | None = typer.Option(None, "--gpu", help="Use one CUDA GPU id for all real backends."),
    n_ctx: int = typer.Option(8192, "--n-ctx", min=512),
    n_gpu_layers: int = typer.Option(-1, "--n-gpu-layers"),
    llama_main_gpu: int | None = typer.Option(None, "--llama-main-gpu", min=0),
    temperature: float = typer.Option(0.2, "--temperature", min=0.0, max=2.0),
    top_p: float = typer.Option(0.9, "--top-p", min=0.0, max=1.0),
    max_new_tokens: int = typer.Option(2048, "--max-new-tokens", min=1),
    llama_verbose: bool = typer.Option(False, "--llama-verbose"),
    llama_use_instructor: bool = typer.Option(True, "--instructor/--no-instructor"),
    instructor_max_retries: int = typer.Option(2, "--instructor-max-retries", min=0),
    embedding_instruction: str | None = typer.Option(None, "--embedding-instruction"),
    embedding_batch_size: int = typer.Option(8, "--embedding-batch-size", min=1),
    embedding_device: str | None = typer.Option(None, "--embedding-device"),
    reranker_instruction: str | None = typer.Option(None, "--reranker-instruction"),
    reranker_batch_size: int = typer.Option(8, "--reranker-batch-size", min=1),
    reranker_device: str | None = typer.Option(None, "--reranker-device"),
    reranker_max_length: int = typer.Option(2048, "--reranker-max-length", min=256, max=8192),
    retrieval_top_k: int = typer.Option(8, "--retrieval-top-k", min=1, max=50),
    rerank_top_k: int = typer.Option(3, "--rerank-top-k", min=1, max=20),
    max_context_chars: int = typer.Option(1200, "--max-context-chars", min=200, max=8000),
) -> None:
    """Generate a Beamer project from a native scientific PDF."""

    if gpu is not None:
        if llama_main_gpu is None:
            llama_main_gpu = gpu
        if embedding_device is None:
            embedding_device = f"cuda:{gpu}"
        if reranker_device is None:
            reranker_device = f"cuda:{gpu}"

    if not use_fake_models:
        if generator_backend == "fake":
            generator_backend = "llama_cpp"
        if embedder_backend == "fake":
            embedder_backend = "sentence_transformers_qwen"
        if reranker_backend == "fake":
            reranker_backend = "transformers_qwen"

    config = PipelineConfig(
        model_path=model,
        embedding_model_path=embedding,
        reranker_model_path=reranker,
        duration_minutes=duration,
        audience=audience,
        theme=theme,
        max_slides=max_slides,
        compile_pdf=compile_pdf,
        latex_engine=latex_engine,
        save_intermediate=save_intermediate,
        debug=debug,
        use_fake_models=use_fake_models,
        fail_on_error=False,
        generator_backend=generator_backend,
        embedder_backend=embedder_backend,
        reranker_backend=reranker_backend,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        llama_main_gpu=llama_main_gpu,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        llama_verbose=llama_verbose,
        llama_use_instructor=llama_use_instructor,
        instructor_max_retries=instructor_max_retries,
        embedding_instruction=embedding_instruction,
        embedding_batch_size=embedding_batch_size,
        embedding_device=embedding_device,
        reranker_instruction=reranker_instruction,
        reranker_batch_size=reranker_batch_size,
        reranker_device=reranker_device,
        reranker_max_length=reranker_max_length,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        max_context_chars=max_context_chars,
    )
    result = PdfToBeamerPipeline(config).generate(pdf_path, output)
    typer.echo(f"Output directory: {result.output_dir}")
    typer.echo(f"main.tex: {result.beamer_tex}")
    if result.slides_pdf:
        typer.echo(f"slides.pdf: {result.slides_pdf}")
    if result.quality_report_path:
        typer.echo(f"quality_report.json: {result.quality_report_path}")
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")
    for error in result.errors:
        typer.echo(f"error: {error}", err=True)
    if result.errors and not result.beamer_tex.exists():
        raise typer.Exit(code=1)
    if result.errors and result.compile_success is False:
        raise typer.Exit(code=1)
