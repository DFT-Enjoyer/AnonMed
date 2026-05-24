from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ModelRunnerResult:
    text: str
    annotation: AnnotationSet
    spans: tuple[Span, ...]
    masked_text: str


class ModelRunner:
    def __init__(self, model: RunnerModel = "example", **kwargs: Any) -> None:
        self._model: RunnerModel = model
        self._kwargs: dict[str, Any] = dict(kwargs)
        self._pii_model: PIIModel | None = None

    def __call__(self, text: str) -> str:
        return self.mask(text)

    def run(self, text: str) -> ModelRunnerResult:
        annotation: AnnotationSet = self.predict(text)
        spans: tuple[Span, ...] = _spans_from_annotation(annotation, text_length=len(text))
        masked_text: str = _mask_text_with_spans(text=text, spans=spans)
        return ModelRunnerResult(
            text=text,
            annotation=annotation,
            spans=spans,
            masked_text=masked_text,
        )

    def predict(self, text: str) -> AnnotationSet:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        resolved_model: str | PIIModel = _normalize_model_argument(self._model)
        kwargs: dict[str, Any] = dict(self._kwargs)
        pii_model: PIIModel = self._resolve_pii_model(resolved_model, kwargs.pop("model_params", None))
        if kwargs:
            unexpected_params: str = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected ModelRunner params for PIIModel inference: {unexpected_params}")
        document: TextDocument = _document_from_text(text)
        return pii_model.predict(document)

    def spans(self, text: str) -> tuple[Span, ...]:
        return self.run(text).spans

    def mask(self, text: str) -> str:
        return self.run(text).masked_text

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
    return _mask_text_with_spans(
        text=text,
        spans=_spans_from_annotation(annotation, text_length=len(text)),
    )


def _spans_from_annotation(annotation: AnnotationSet, *, text_length: int) -> tuple[Span, ...]:
    line: AnnotationSetLine | None = _first_line(annotation.lines)
    if line is None:
        return ()
    spans: list[Span] = sorted(
        [span for span in line.spans if 0 <= span.begin < span.end <= text_length],
        key=lambda span: (span.begin, span.end),
    )
    return tuple(spans)


def _mask_text_with_spans(text: str, spans: Iterable[Span]) -> str:
    masked_text: str = text
    next_start: int = len(text) + 1
    sorted_spans: list[Span] = sorted(spans, key=lambda span: (span.begin, span.end), reverse=True)
    for span in sorted_spans:
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


__all__: list[str] = ["ModelRunner", "ModelRunnerResult", "RunnerModel"]
