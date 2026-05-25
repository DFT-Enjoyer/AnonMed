from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from anonmed.ml.config import PipelineConfig, load_pipeline_config
from anonmed.ml.core.snapshot import DatasetSnapshotWriter
from anonmed.ml.evaluation import EvaluationResult
from anonmed.ml.factory import evaluate_with_predictions
from anonmed.ml.models.base import PIIModel
from anonmed.ml.outputs import build_run_instance_dir
from anonmed.ml.pipelines.GLiNER2_TrH_Tests import _error_examples, _format_error_examples
from anonmed.ml.pipelines.terminal import print_metrics_block
from anonmed.ml.registry import PipelineComponents, build_pipeline_components


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "GLiNER2.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the regular GLiNER2 evaluation pipeline.")
    parser.add_argument(
        "--config",
        default=str(_default_config_path()),
        help="Path to the GLiNER2 evaluation config.",
    )
    parser.add_argument(
        "--error-examples",
        type=int,
        default=5,
        help="Number of FP/FN examples printed and saved. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-document prediction progress.",
    )
    parser.add_argument("--json", action="store_true", help="Print the report payload as JSON.")
    return parser


def run_pipeline(
    config: PipelineConfig,
    *,
    show_progress: bool = True,
    error_examples_limit: int = 5,
) -> dict[str, Any]:
    components: PipelineComponents = build_pipeline_components(config)
    model: PIIModel = components.model
    evaluation_result: EvaluationResult = evaluate_with_predictions(
        components.dataset,
        model,
        components.metrics,
        show_progress=show_progress,
    )
    error_examples = _error_examples(
        components.dataset.cases,
        evaluation_result.predictions,
        limit=error_examples_limit,
    )

    instance_dir = build_run_instance_dir(config)
    instance_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "run": asdict(config.run),
        "dataset": asdict(config.dataset),
        "model": asdict(config.model),
        "metrics": [asdict(metric) for metric in config.metrics],
        "training": asdict(config.training),
        "evaluation": asdict(config.evaluation),
        "samples_count": evaluation_result.report.samples_count,
        "metric_results": evaluation_result.report.metrics,
        "error_examples": error_examples,
    }

    report_path = instance_dir / config.outputs.report_filename
    payload["instance"] = {"run_dir": str(instance_dir), "report": str(report_path)}

    snapshot_writer = DatasetSnapshotWriter()
    if config.outputs.write_dataset_snapshot_json:
        snapshot_path = instance_dir / "dataset_snapshot.json"
        snapshot_writer.write_json(components.dataset, snapshot_path)
        payload["instance"]["dataset_snapshot_json"] = str(snapshot_path)
    if config.outputs.write_dataset_snapshot_parquet:
        snapshot_path = instance_dir / "dataset_snapshot.parquet"
        snapshot_writer.write_parquet(components.dataset, snapshot_path)
        payload["instance"]["dataset_snapshot_parquet"] = str(snapshot_path)

    evaluation_snapshot_path = instance_dir / "evaluation_snapshot.json"
    snapshot_writer.write_evaluation_json(
        components.dataset,
        evaluation_result.predictions,
        evaluation_snapshot_path,
    )
    payload["instance"]["evaluation_snapshot_json"] = str(evaluation_snapshot_path)

    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)
    payload = run_pipeline(
        config,
        show_progress=not args.no_progress and not args.json,
        error_examples_limit=args.error_examples,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"report: {payload['instance']['report']}")
        print_metrics_block(payload["metric_results"], title="GLiNER2 METRICS")
        print(_format_error_examples(payload.get("error_examples", ())))
    return 0


__all__ = ["build_parser", "main", "run_pipeline"]


if __name__ == "__main__":
    raise SystemExit(main())
