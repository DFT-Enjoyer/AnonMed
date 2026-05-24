from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["build_dashboard_manifest", "write_dashboard"]

_LAZY_EXPORTS = {
    "build_dashboard_manifest": ("anonmed.ml.visualization.dashboard", "build_dashboard_manifest"),
    "write_dashboard": ("anonmed.ml.visualization.dashboard", "write_dashboard"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
