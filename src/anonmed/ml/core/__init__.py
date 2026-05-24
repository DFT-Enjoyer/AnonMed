from __future__ import annotations

from importlib import import_module
from typing import Any

from .types import (
    AnnotationSet,
    AnnotationSetLine,
    Case,
    EvaluationReport,
    MetricResult,
    MetricValue,
    ParticipantKind,
    Role,
    Span,
    TextDocument,
    TextLine,
)

__all__ = [
    "AnnotationSet",
    "AnnotationSetLine",
    "Case",
    "DatasetSnapshotWriter",
    "EvaluationReport",
    "MetricResult",
    "MetricValue",
    "ParticipantKind",
    "Role",
    "Span",
    "TextDocument",
    "TextLine",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DatasetSnapshotWriter": ("anonmed.ml.core.snapshot", "DatasetSnapshotWriter"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
