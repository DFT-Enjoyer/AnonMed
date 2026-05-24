from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from anonmed.ml.config import ModelConfig
from anonmed.ml.core.types import (
    AnnotationSet,
    AnnotationSetLine,
    ParticipantKind,
    Role,
    Span,
    TextDocument,
    TextLine,
)
from anonmed.ml.models.base import PIIModel
from anonmed.ml.registry import build_model

RunnerModel = str | PIIModel | set[str]


class ModelRunner:
    def __init__(self, model: RunnerModel = "example", **kwargs: Any) -> None:
        self._model: RunnerModel = model
        self._kwargs: dict[str, Any] = dict(kwargs)
        self._pii_model: PIIModel | None = None

    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        resolved_model: str | PIIModel = _normalize_model_argument(self._model)
        kwargs: dict[str, Any] = dict(self._kwargs)
        pii_model: PIIModel = self._resolve_pii_model(resolved_model, kwargs.pop("model_params", None))
        if kwargs:
            unexpected_params: str = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected ModelRunner params for PIIModel inference: {unexpected_params}")
        document: TextDocument = _document_from_text(text)
        prediction: AnnotationSet = pii_model.predict(document)
        return _mask_text_with_annotation(text=text, annotation=prediction)

    def _resolve_pii_model(self, model: str | PIIModel, model_params: object) -> PIIModel:
        if isinstance(model, PIIModel):
            if model_params:
                raise TypeError("model_params cannot be used when model is a PIIModel instance")
            return model
        if self._pii_model is None:
            self._pii_model = _build_pii_model(model, model_params)
        return self._pii_model


def _normalize_model_argument(model: RunnerModel) -> str | PIIModel:
    if isinstance(model, set):
        if len(model) != 1:
            raise ValueError("model set must contain exactly one model name")
        model_name: str = next(iter(model))
        return model_name
    return model


def _build_pii_model(model: str, model_params: object) -> PIIModel:
    if not isinstance(model, str) or model == "":
        raise TypeError(f"model must be a non-empty str or PIIModel, got {type(model).__name__}")
    params: dict[str, Any] = dict(model_params) if isinstance(model_params, dict) else {}
    if model_params is not None and not isinstance(model_params, dict):
        raise TypeError("model_params must be dict[str, Any] when provided")
    return build_model(ModelConfig(id=model, params=params))


def _document_from_text(text: str) -> TextDocument:
    role = Role(name="text", kind=ParticipantKind.UNKNOWN)
    line = TextLine(idx=0, role=role, text=text)
    return TextDocument(lines=(line,), sample_id=None)


def _mask_text_with_annotation(text: str, annotation: AnnotationSet) -> str:
    line: AnnotationSetLine | None = _first_line(annotation.lines)
    if line is None:
        return text
    spans: list[Span] = sorted(
        [span for span in line.spans if 0 <= span.begin < span.end <= len(text)],
        key=lambda span: (span.begin, span.end),
        reverse=True,
    )
    masked_text: str = text
    next_start: int = len(text) + 1
    for span in spans:
        if span.end > next_start:
            continue
        replacement: str = f"[{span.label}]"
        masked_text = f"{masked_text[:span.begin]}{replacement}{masked_text[span.end:]}"
        next_start = span.begin
    return masked_text


def _first_line(lines: Iterable[AnnotationSetLine]) -> AnnotationSetLine | None:
    for line in lines:
        return line
    return None


__all__: list[str] = ["ModelRunner", "RunnerModel"]
