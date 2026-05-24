from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import unittest

from anonmed.ml.config import pipeline_config_from_mapping
from anonmed.ml.factory import evaluate
from anonmed.ml.data.example import build_example_dataset
from anonmed.ml.metrics.example import ExampleCountMetric
from anonmed.ml.models.example import ExamplePIIModel
from anonmed.ml.outputs import build_run_instance_dir
from anonmed.ml.registry import RegistryError, build_dataset


class MLOrchestrationTests(unittest.TestCase):
    def test_public_ml_api_imports_from_repository_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        completed_process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "from anonmed.ml import "
                    "AnnotationSet, Dataset, Metric, PIIModel, TextDocument, evaluate; "
                    "from anonmed.ml.core import Case, DatasetSnapshotWriter, Span; "
                    "import anonmed.ml.registry; "
                    "import anonmed.ml.data.russian_pii_66k; "
                    "import anonmed.ml.models.natasha_per"
                ),
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

    def test_run_instance_dir_uses_fixed_utc_plus_three_timestamp(self) -> None:
        config = pipeline_config_from_mapping(
            {
                "run": {"name": "example"},
                "dataset": "example",
                "model": "example",
                "metrics": ["example_count"],
                "outputs": {"instance_dir": "instance"},
            }
        )
        run_dir = build_run_instance_dir(
            config,
            now=datetime(2026, 5, 24, 9, 12, 39, 398642, tzinfo=timezone.utc),
        )

        self.assertEqual(
            run_dir,
            Path("instance") / "example" / "2026-05-24_12-12-39_398642",
        )

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
        self.assertIn('"instance":', completed_process.stdout)
        payload = json.loads(completed_process.stdout)
        report_path = payload["instance"]["report"]
        self.assertRegex(
            report_path,
            r"^instance/example/\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}/report\.json$",
        )
        self.assertEqual(payload["instance"]["run_dir"], str(Path(report_path).parent))
        self.assertNotIn("instance/ml", completed_process.stdout)
        self.assertNotIn('"artifacts":', completed_process.stdout)

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
