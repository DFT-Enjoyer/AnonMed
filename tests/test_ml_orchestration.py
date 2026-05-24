from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from anonmed.ml.config import pipeline_config_from_mapping
from anonmed.ml.factory import evaluate
from anonmed.ml.datasets.example import build_example_dataset
from anonmed.ml.metrics.example import ExampleCountMetric
from anonmed.ml.models.example import ExamplePIIModel
from anonmed.ml.registry import RegistryError, build_dataset


class MLOrchestrationTests(unittest.TestCase):
    def test_anonmed_ml_imports_and_example_evaluation_runs(self) -> None:
        dataset = build_example_dataset()
        report = evaluate(dataset, ExamplePIIModel(), [ExampleCountMetric()])

        self.assertEqual(report.samples_count, 1)
        self.assertEqual(
            report.metrics["example_count"],
            {"predictions_count": 1, "cases_count": 1},
        )

    def test_unknown_dataset_id_is_rejected(self) -> None:
        config = pipeline_config_from_mapping(
            {
                "dataset": "missing",
                "model": "example",
                "metrics": ["example_count"],
            }
        )
        with self.assertRaisesRegex(RegistryError, "Unknown dataset id: missing"):
            build_dataset(config.dataset)

    def test_train_module_runs_from_yaml_config(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config_path = repository_root / "configs" / "ml" / "train.example.yaml"
        completed_process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "anonmed.ml.pipelines.train",
                "--config",
                str(config_path),
                "--json",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed_process.returncode,
            0,
            msg=completed_process.stderr or completed_process.stdout,
        )
        self.assertIn('"samples_count": 1', completed_process.stdout)

    def test_no_top_level_ml_imports_remain_in_source(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        source_root = repository_root / "src" / "anonmed"
        offenders: list[str] = []
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("from ml.") or stripped.startswith("import ml."):
                    offenders.append(f"{path.relative_to(repository_root)}: {stripped}")

        self.assertEqual(offenders, [])
        self.assertFalse((repository_root / "src" / "ml").exists())

    def test_top_level_ml_package_is_not_importable_from_repository_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        completed_process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import importlib.util; raise SystemExit(importlib.util.find_spec('ml') is not None)",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed_process.returncode,
            0,
            msg=completed_process.stderr or completed_process.stdout,
        )


__all__: list[str] = []
