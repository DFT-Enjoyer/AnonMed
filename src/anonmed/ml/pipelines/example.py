import sys
from pathlib import Path
from anonmed.ml.config import load_pipeline_config
from anonmed.ml.registry import build_pipeline_components
from anonmed.ml.factory import evaluate, create_dataset_snapshot_writer
from anonmed.ml.metrics.base import Metric
from anonmed.ml.outputs import build_run_instance_dir

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_evaluation.py <config.yaml>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
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
    for name, values in report.metrics.items():
        print(f"{name}: {values}")

if __name__ == "__main__":
    main()
