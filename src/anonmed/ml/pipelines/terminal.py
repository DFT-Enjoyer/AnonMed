from __future__ import annotations

import sys
from typing import Mapping


def print_metrics_block(
    metrics: Mapping[str, object],
    *,
    title: str = "METRICS",
) -> None:
    if not metrics:
        return
    border = "=" * 72
    print(_color(border, "cyan"))
    print(_color(title, "bold"))
    print(_color(border, "cyan"))
    for name, values in metrics.items():
        print(_color(f"{name}: {values}", "green"))
    print(_color(border, "cyan"))


def _color(text: str, style: str) -> str:
    if not sys.stdout.isatty():
        return text
    codes = {
        "bold": "1",
        "cyan": "36",
        "green": "32",
    }
    code = codes.get(style)
    return text if code is None else f"\033[{code}m{text}\033[0m"


__all__ = ["print_metrics_block"]
