"""Local Qwen reranker adapter backed by transformers."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

from pdf2beamer.errors import (
    LocalModelInferenceError,
    LocalModelLoadError,
    OptionalDependencyNotInstalledError,
)
from pdf2beamer.retrieval.reranker import BaseReranker

_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LocalQwenReranker(BaseReranker):
    """Score query-document pairs with a local Qwen3-Reranker model folder."""

    def __init__(
        self,
        model_path: str | Path,
        instruction: str | None = None,
        batch_size: int = 8,
        device: str | None = None,
        max_length: int = 8192,
    ) -> None:
        self.model_path = Path(model_path)
        self.instruction = instruction or _DEFAULT_INSTRUCTION
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._true_token_id: int | None = None
        self._false_token_id: int | None = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._load_model()

    def score(self, query: str, text: str) -> float:
        """Score one query-text pair in [0, 1]."""

        return self.score_batch(query, [text])[0]

    def score_batch(self, query: str, texts: list[str]) -> list[float]:
        """Score query-text pairs with batched local causal-LM scoring."""

        if not texts:
            return []
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            pairs = [_format_instruction(self.instruction, query, text) for text in batch_texts]
            try:
                encoded = self._process_inputs(pairs)
                with self._torch.no_grad():
                    output = self._model(**encoded)
                scores.extend(
                    _yes_no_probabilities(
                        output.logits,
                        self._true_token_id,
                        self._false_token_id,
                        self._torch,
                    )
                )
            except Exception as exc:  # pragma: no cover - backend-specific failure path
                raise LocalModelInferenceError(f"Local Qwen reranking failed: {exc}") from exc
        return scores

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise LocalModelLoadError(
                f"Local Qwen reranker model directory does not exist: {self.model_path}"
            )
        if not self.model_path.is_dir():
            raise LocalModelLoadError(
                f"Local Qwen reranker model path is not a directory: {self.model_path}"
            )
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise OptionalDependencyNotInstalledError(
                "torch and transformers are required. Install pdf2beamer with the [models] extra."
            ) from exc
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self.model_path),
                padding_side="left",
                local_files_only=True,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            if self.device is not None and hasattr(model, "to"):
                model = model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
            true_token_id = tokenizer.convert_tokens_to_ids("yes")
            false_token_id = tokenizer.convert_tokens_to_ids("no")
            prefix_tokens = tokenizer.encode(_SYSTEM_PROMPT, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(_SUFFIX, add_special_tokens=False)
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            raise LocalModelLoadError(f"Failed to load local Qwen reranker model: {exc}") from exc
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._true_token_id = int(true_token_id)
        self._false_token_id = int(false_token_id)
        self._prefix_tokens = [int(token) for token in prefix_tokens]
        self._suffix_tokens = [int(token) for token in suffix_tokens]

    def _process_inputs(self, pairs: list[str]) -> Any:
        max_pair_length = max(1, self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens))
        inputs = self._tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_pair_length,
        )
        input_ids = inputs["input_ids"]
        for index, ids in enumerate(input_ids):
            input_ids[index] = self._prefix_tokens + list(ids) + self._suffix_tokens
        padded = self._tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return _move_to_device(padded, _model_device(self._model, self.device))


def _format_instruction(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def _model_device(model: Any, fallback: str | None) -> str | None:
    device = getattr(model, "device", None)
    return str(device) if device is not None else fallback


def _move_to_device(encoded: Any, device: str | None) -> Any:
    if device is None:
        return encoded
    if hasattr(encoded, "to"):
        return encoded.to(device)
    if isinstance(encoded, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
    return encoded


def _yes_no_probabilities(
    logits: Any,
    true_token_id: int,
    false_token_id: int,
    torch_module: Any,
) -> list[float]:
    try:
        final_logits = logits[:, -1, :]
        true_values = final_logits[:, true_token_id]
        false_values = final_logits[:, false_token_id]
        stacked = torch_module.stack([false_values, true_values], dim=1)
        log_probs = torch_module.nn.functional.log_softmax(stacked, dim=1)
        probs = log_probs[:, 1].exp()
        return [_clamp_01(float(score)) for score in probs.tolist()]
    except Exception:
        return _yes_no_probabilities_from_lists(logits, true_token_id, false_token_id)


def _yes_no_probabilities_from_lists(
    logits: Any,
    true_token_id: int,
    false_token_id: int,
) -> list[float]:
    rows = logits.tolist() if hasattr(logits, "tolist") else logits
    scores: list[float] = []
    for row in rows:
        final = row[-1]
        true_value = float(final[true_token_id])
        false_value = float(final[false_token_id])
        scores.append(_clamp_01(_two_class_probability(false_value, true_value)))
    return scores


def _two_class_probability(false_value: float, true_value: float) -> float:
    max_value = max(false_value, true_value)
    false_exp = math.exp(false_value - max_value)
    true_exp = math.exp(true_value - max_value)
    total = false_exp + true_exp
    return 0.0 if total == 0.0 else true_exp / total


def _clamp_01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
