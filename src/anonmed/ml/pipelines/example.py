import sys
from pathlib import Path

from anonmed.ml.config import load_pipeline_config
from anonmed.ml.factory import create_dataset_snapshot_writer, evaluate
from anonmed.ml.outputs import build_run_instance_dir
from anonmed.ml.pipelines.terminal import print_metrics_block
from anonmed.ml.registry import build_pipeline_components


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python run_evaluation.py <config.yaml>")
        sys.exit(1)
    config_path: Path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_pipeline_config(config_path)
    components = build_pipeline_components(config)

    print(f"Running evaluation for: {config.run.name}")
    print(f"Dataset: {config.dataset.id}, samples: {len(components.dataset.cases)}")
    print(f"Model: {config.model.id}")
    print(f"Metrics: {[m.name for m in components.metrics]}")

    report = evaluate(components.dataset, components.model, components.metrics, show_progress=True)

    output_dir = build_run_instance_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / config.outputs.report_filename
    report.to_json(report_path)

    if config.outputs.write_dataset_snapshot_json or config.outputs.write_dataset_snapshot_parquet:
        writer = create_dataset_snapshot_writer()
        snapshot_path = output_dir / "dataset_snapshot.json"
        writer.write_json(components.dataset, snapshot_path)

    print(f"Report saved to {report_path}")
    print_metrics_block(report.metrics)


if __name__ == "__main__":
    main()


__all__: list[str] = ["main"]
