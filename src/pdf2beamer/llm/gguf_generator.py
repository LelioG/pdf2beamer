"""Local-only Nemotron GGUF generation adapter backed by llama-cpp-python."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pdf2beamer.errors import (
    LocalModelInferenceError,
    LocalModelLoadError,
    OptionalDependencyNotInstalledError,
)
from pdf2beamer.llm.base import BaseGenerator
from pdf2beamer.llm.output_parser import extract_json_object


class _ArgumentGraphResponse(BaseModel):
    """Instructor response model for ArgumentGraph JSON payloads."""

    model_config = ConfigDict(extra="allow")

    paper_title: str | None = None
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class _SlideIRResponse(BaseModel):
    """Instructor response model for SlideIR JSON payloads."""

    model_config = ConfigDict(extra="allow")

    paper_title: str | None = None
    audience: str | None = None
    duration_minutes: int | None = None
    slides: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LocalGGUFGenerator(BaseGenerator):
    """Generate structured JSON with a local GGUF instruct model via llama.cpp."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_new_tokens: int = 2048,
        verbose: bool = False,
        main_gpu: int | None = None,
        use_instructor: bool = True,
        instructor_max_retries: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.main_gpu = main_gpu
        self.use_instructor = use_instructor
        self.instructor_max_retries = instructor_max_retries
        self._instructor_create = None
        self._llama = self._load_model()

    def generate_text(self, prompt: str) -> str:
        """Generate raw text locally without calling hosted services."""

        return self._complete(prompt, temperature=self.temperature, schema_name=None)

    def generate_json(self, prompt: str, schema_name: str | None = None) -> dict[str, Any]:
        """Generate and parse one JSON object using llama.cpp JSON mode when available."""

        if self.use_instructor:
            instructor_payload = self._generate_json_with_instructor(prompt, schema_name)
            if instructor_payload is not None:
                return instructor_payload

        schema_hint = f" for schema {schema_name}" if schema_name else ""
        prompts = [
            (
                f"{prompt}\n\n<|no_reasoning|>\n"
                f"Return JSON only{schema_hint}. Start with {{ and end with }}. "
                "Do not include Markdown fences, explanations, LaTeX, Beamer, or prose."
            ),
            (
                "You are a strict JSON serializer. Output exactly one valid JSON object, "
                "with double-quoted keys and strings. No comments. No Markdown.\n\n"
                f"Task:\n{prompt}\n\nJSON object:"
            ),
        ]
        last_output = ""
        for index, json_prompt in enumerate(prompts):
            output = self._complete(
                json_prompt,
                temperature=0.0 if index else self.temperature,
                schema_name=schema_name,
            )
            last_output = output
            try:
                return extract_json_object(output, required_keys=_required_json_keys(schema_name))
            except ValueError:
                continue
        if self.verbose and last_output:
            preview = last_output[:1000].replace("\n", " ")
            raise LocalModelInferenceError(
                "Local GGUF output could not be parsed as a JSON object. "
                f"Output preview: {preview}"
            )
        raise LocalModelInferenceError("Local GGUF output could not be parsed as a JSON object.")

    def _generate_json_with_instructor(
        self,
        prompt: str,
        schema_name: str | None,
    ) -> dict[str, Any] | None:
        response_model = _instructor_response_model(schema_name)
        if response_model is None:
            return None
        create = self._get_instructor_create()
        if create is None:
            return None
        try:
            with _native_output_context(self.verbose):
                response = create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "<|no_reasoning|>\n"
                                "Return structured JSON only. No Markdown, LaTeX, "
                                "Beamer, or explanatory prose."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_model=response_model,
                    max_retries=self.instructor_max_retries,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
        except Exception:
            if self.verbose:
                print("Instructor structured generation failed; falling back to JSON extraction.")
            return None
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        return None

    def _get_instructor_create(self) -> Any | None:
        if self._instructor_create is not None:
            return self._instructor_create
        if not hasattr(self._llama, "create_chat_completion_openai_v1"):
            return None
        try:
            instructor = importlib.import_module("instructor")
        except ImportError:
            return None
        try:
            self._instructor_create = instructor.patch(
                create=self._llama.create_chat_completion_openai_v1,
                mode=instructor.Mode.JSON_SCHEMA,
            )
        except Exception:
            if self.verbose:
                print("Instructor patch failed; falling back to llama.cpp JSON mode.")
            return None
        return self._instructor_create

    def _complete(self, prompt: str, temperature: float, schema_name: str | None) -> str:
        try:
            with _native_output_context(self.verbose):
                if hasattr(self._llama, "create_chat_completion"):
                    completion = self._chat_completion(prompt, temperature, schema_name)
                else:
                    completion = self._llama(
                        prompt,
                        max_tokens=self.max_new_tokens,
                        temperature=temperature,
                        top_p=self.top_p,
                        stop=["</s>", "<|im_end|>"],
                    )
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            raise LocalModelInferenceError(f"Local GGUF generation failed: {exc}") from exc
        return _extract_completion_text(completion)

    def _chat_completion(
        self,
        prompt: str,
        temperature: float,
        schema_name: str | None,
    ) -> Any:
        base_kwargs = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "<|no_reasoning|>\n"
                        "Thinking/reasoning mode is disabled when the model supports it. "
                        "You return structured JSON only. Never return Markdown, LaTeX, "
                        "Beamer, or explanatory prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": temperature,
            "top_p": self.top_p,
            "stop": ["<|im_end|>"],
        }
        response_formats = [
            _json_schema_response_format(schema_name),
            {"type": "json_object"},
            None,
        ]
        last_error: Exception | None = None
        for response_format in response_formats:
            kwargs = dict(base_kwargs)
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                return self._llama.create_chat_completion(**kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return self._llama.create_chat_completion(**base_kwargs)

    def _load_model(self) -> Any:
        if not self.model_path.exists():
            raise LocalModelLoadError(f"Local GGUF model file does not exist: {self.model_path}")
        if not self.model_path.is_file():
            raise LocalModelLoadError(f"Local GGUF model path is not a file: {self.model_path}")
        try:
            llama_cpp = importlib.import_module("llama_cpp")
        except ImportError as exc:
            raise OptionalDependencyNotInstalledError(
                "llama-cpp-python is not installed. Install pdf2beamer with the [models] extra."
            ) from exc
        try:
            with _native_output_context(self.verbose):
                return llama_cpp.Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    main_gpu=0 if self.main_gpu is None else self.main_gpu,
                    verbose=self.verbose,
                    logits_all=True,
                )
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            raise LocalModelLoadError(f"Failed to load local GGUF model: {exc}") from exc


class LocalNemotronGenerator(LocalGGUFGenerator):
    """Generate structured JSON with the configured local Nemotron GGUF model."""


@contextmanager
def _suppress_native_output():
    """Suppress stdout/stderr written by native libraries such as llama.cpp."""

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _native_output_context(verbose: bool):
    if verbose:
        return nullcontext()
    return _suppress_native_output()


def _required_json_keys(schema_name: str | None) -> set[str] | None:
    if schema_name == "ArgumentGraph":
        return {"nodes", "edges"}
    if schema_name == "SlideIR":
        return {"slides"}
    return None


def _instructor_response_model(schema_name: str | None) -> type[BaseModel] | None:
    if schema_name == "ArgumentGraph":
        return _ArgumentGraphResponse
    if schema_name == "SlideIR":
        return _SlideIRResponse
    return None


def _json_schema_response_format(schema_name: str | None) -> dict[str, Any] | None:
    schema = _json_schema_for(schema_name)
    if schema is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name or "StructuredOutput",
            "schema": schema,
            "strict": False,
        },
    }


def _json_schema_for(schema_name: str | None) -> dict[str, Any] | None:
    if schema_name == "ArgumentGraph":
        return {
            "type": "object",
            "properties": {
                "paper_title": {"type": ["string", "null"]},
                "nodes": {"type": "array", "items": {"type": "object"}},
                "edges": {"type": "array", "items": {"type": "object"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["nodes", "edges"],
        }
    if schema_name == "SlideIR":
        return {
            "type": "object",
            "properties": {
                "paper_title": {"type": ["string", "null"]},
                "audience": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "slides": {"type": "array", "items": {"type": "object"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["slides"],
        }
    return None


def _extract_completion_text(completion: Any) -> str:
    """Handle common llama-cpp-python completion response shapes."""

    if isinstance(completion, str):
        return completion
    if not isinstance(completion, dict):
        raise LocalModelInferenceError("llama.cpp returned an unsupported completion object.")
    choices = completion.get("choices")
    if not choices:
        raise LocalModelInferenceError("llama.cpp completion did not contain choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise LocalModelInferenceError("llama.cpp completion choice has an unsupported shape.")
    text = first.get("text")
    if isinstance(text, str):
        return text
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    raise LocalModelInferenceError("llama.cpp completion did not contain generated text.")
