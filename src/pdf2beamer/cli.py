"""Command line interface for pdf2beamer."""

from pathlib import Path
from typing import Annotated

import typer

from pdf2beamer.config import PipelineConfig
from pdf2beamer.model_download import default_model_specs, download_default_models
from pdf2beamer.pipeline import PdfToBeamerPipeline

app = typer.Typer(help="Convert native scientific PDFs into Beamer decks.")


@app.callback()
def main() -> None:
    """Convert native scientific PDFs into Beamer decks."""


@app.command("download-models")
def download_models(
    destination: Annotated[
        Path,
        typer.Argument(help="Directory where the models/ layout will be created."),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download files even if Hugging Face has cached them."),
    ] = False,
    token: Annotated[
        str | None,
        typer.Option("--token", help="Hugging Face token for gated/private models."),
    ] = None,
) -> None:
    """Download the default local models expected by --real-models."""

    try:
        paths = download_default_models(destination, force=force, token=token)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Downloaded default models:")
    for spec, path in zip(default_model_specs(destination), paths, strict=True):
        typer.echo(f"- {spec.name}: {path}")


@app.command()
def generate(
    pdf_path: Annotated[Path, typer.Argument(..., metavar="PDF_PATH")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory."),
    ] = Path("out"),
    model: Annotated[
        Path | None,
        typer.Option("--model", help="Local GGUF generator path."),
    ] = None,
    embedding: Annotated[
        Path | None,
        typer.Option("--embedding", help="Local embedding model path."),
    ] = None,
    reranker: Annotated[
        Path | None,
        typer.Option("--reranker", help="Local reranker model path."),
    ] = None,
    duration: Annotated[int, typer.Option("--duration", min=1)] = 10,
    audience: Annotated[str, typer.Option("--audience")] = "technical",
    theme: Annotated[str, typer.Option("--theme")] = "clean",
    max_slides: Annotated[int | None, typer.Option("--max-slides", min=1)] = None,
    compile_pdf: Annotated[bool, typer.Option("--compile/--no-compile")] = True,
    latex_engine: Annotated[str, typer.Option("--latex-engine")] = "latexmk",
    debug: Annotated[bool, typer.Option("--debug")] = False,
    save_intermediate: Annotated[
        bool,
        typer.Option("--save-intermediate/--no-save-intermediate"),
    ] = True,
    use_fake_models: Annotated[
        bool,
        typer.Option(
            "--use-fake-models/--real-models",
            help="Use deterministic fake models or local real model backends.",
        ),
    ] = True,
    generator_backend: Annotated[str, typer.Option("--generator-backend")] = "fake",
    embedder_backend: Annotated[str, typer.Option("--embedder-backend")] = "fake",
    reranker_backend: Annotated[str, typer.Option("--reranker-backend")] = "fake",
    gpu: Annotated[
        int | None,
        typer.Option("--gpu", help="Use one CUDA GPU id for all real backends."),
    ] = None,
    n_ctx: Annotated[int, typer.Option("--n-ctx", min=512)] = 8192,
    n_gpu_layers: Annotated[int, typer.Option("--n-gpu-layers")] = -1,
    llama_main_gpu: Annotated[int | None, typer.Option("--llama-main-gpu", min=0)] = None,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0, max=2.0)] = 0.2,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 0.9,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = 2048,
    llama_verbose: Annotated[bool, typer.Option("--llama-verbose")] = False,
    llama_use_instructor: Annotated[bool, typer.Option("--instructor/--no-instructor")] = True,
    instructor_max_retries: Annotated[
        int,
        typer.Option("--instructor-max-retries", min=0),
    ] = 2,
    embedding_instruction: Annotated[
        str | None,
        typer.Option("--embedding-instruction"),
    ] = None,
    embedding_batch_size: Annotated[
        int,
        typer.Option("--embedding-batch-size", min=1),
    ] = 8,
    embedding_device: Annotated[str | None, typer.Option("--embedding-device")] = None,
    reranker_instruction: Annotated[str | None, typer.Option("--reranker-instruction")] = None,
    reranker_batch_size: Annotated[
        int,
        typer.Option("--reranker-batch-size", min=1),
    ] = 8,
    reranker_device: Annotated[str | None, typer.Option("--reranker-device")] = None,
    reranker_max_length: Annotated[
        int,
        typer.Option("--reranker-max-length", min=256, max=8192),
    ] = 2048,
    retrieval_top_k: Annotated[int, typer.Option("--retrieval-top-k", min=1, max=50)] = 8,
    rerank_top_k: Annotated[int, typer.Option("--rerank-top-k", min=1, max=20)] = 3,
    max_context_chars: Annotated[
        int,
        typer.Option("--max-context-chars", min=200, max=8000),
    ] = 1200,
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
