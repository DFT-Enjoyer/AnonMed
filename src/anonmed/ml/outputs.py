from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

from anonmed.ml.config import PipelineConfig

RUN_TIMEZONE = timezone(timedelta(hours=3))


def build_run_instance_dir(config: PipelineConfig, now: datetime | None = None) -> Path:
    timestamp = _run_datetime(now).strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_name = _safe_path_part(config.run.name)
    return Path(config.outputs.instance_dir) / run_name / timestamp


def _run_datetime(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(RUN_TIMEZONE)
    if now.tzinfo is None:
        return now.replace(tzinfo=RUN_TIMEZONE)
    return now.astimezone(RUN_TIMEZONE)


def _safe_path_part(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return normalized or "run"


__all__ = ["RUN_TIMEZONE", "build_run_instance_dir"]
