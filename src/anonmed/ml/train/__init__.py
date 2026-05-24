from __future__ import annotations

from importlib import import_module
from typing import Any


__all__: list[str] = [
    "DEFAULT_ENTITY_LABELS",
    "DEFAULT_LABEL_SCHEMA",
    "PIDRTrainingResult",
    "PIDRTrainingSettings",
    "build_parser",
    "case_to_token_features",
    "compute_token_metrics",
    "main",
    "select_training_spans",
    "train_pidr",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_ENTITY_LABELS": ("anonmed.ml.train.pidr", "DEFAULT_ENTITY_LABELS"),
    "DEFAULT_LABEL_SCHEMA": ("anonmed.ml.train.pidr", "DEFAULT_LABEL_SCHEMA"),
    "PIDRTrainingResult": ("anonmed.ml.train.pidr", "PIDRTrainingResult"),
    "PIDRTrainingSettings": ("anonmed.ml.train.pidr", "PIDRTrainingSettings"),
    "build_parser": ("anonmed.ml.train.pidr", "build_parser"),
    "case_to_token_features": ("anonmed.ml.train.pidr", "case_to_token_features"),
    "compute_token_metrics": ("anonmed.ml.train.pidr", "compute_token_metrics"),
    "main": ("anonmed.ml.train.pidr", "main"),
    "select_training_spans": ("anonmed.ml.train.pidr", "select_training_spans"),
    "train_pidr": ("anonmed.ml.train.pidr", "train_pidr"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value: Any = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
