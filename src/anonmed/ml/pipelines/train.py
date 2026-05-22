from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from anonmed.ml.config import PipelineConfig, load_pipeline_config
from anonmed.ml.core.snapshot import DatasetSnapshotWriter
from anonmed.ml.core.types import EvaluationReport
from anonmed.ml.factory import evaluate
from anonmed.ml.models.base import PIIModel, TrainablePIIModel
from anonmed.ml.registry import PipelineComponents, build_pipeline_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AnonMed ML pipeline from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report payload as JSON after the run finishes.",
    )
    return parser


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    components: PipelineComponents = build_pipeline_components(config)
    model: PIIModel = components.model

    if config.training.enabled:
        if not isinstance(model, TrainablePIIModel):
            message = (
                f"Model {config.model.id!r} is not trainable. "
                "Set training.enabled=false or register a trainable model."
            )
            raise TypeError(message)
        fit_result: object = model.fit(components.dataset)
        if isinstance(fit_result, PIIModel):
            model = fit_result

    report: EvaluationReport | None = None
    if config.evaluation.enabled:
        report = evaluate(components.dataset, model, components.metrics)

    artifact_dir = Path(config.outputs.artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "run": asdict(config.run),
        "dataset": asdict(config.dataset),
        "model": asdict(config.model),
        "metrics": [asdict(metric) for metric in config.metrics],
        "training": asdict(config.training),
        "evaluation": asdict(config.evaluation),
        "samples_count": 0 if report is None else report.samples_count,
        "metric_results": {} if report is None else report.metrics,
    }

    report_path = artifact_dir / config.outputs.report_filename
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["artifacts"] = {"report": str(report_path)}

    snapshot_writer = DatasetSnapshotWriter()
    if config.outputs.write_dataset_snapshot_json:
        snapshot_path = artifact_dir / "dataset_snapshot.json"
        snapshot_writer.write_json(components.dataset, snapshot_path)
        payload["artifacts"]["dataset_snapshot_json"] = str(snapshot_path)
    if config.outputs.write_dataset_snapshot_parquet:
        snapshot_path = artifact_dir / "dataset_snapshot.parquet"
        snapshot_writer.write_parquet(components.dataset, snapshot_path)
        payload["artifacts"]["dataset_snapshot_parquet"] = str(snapshot_path)

    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)
    payload = run_pipeline(config)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"report: {payload['artifacts']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_pipeline"]
