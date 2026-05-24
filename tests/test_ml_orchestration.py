from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from anonmed.ml.config import pipeline_config_from_mapping
from anonmed.ml.core.types import Role, TextDocument, TextLine
from anonmed.ml.factory import evaluate
from anonmed.ml.data.example import build_example_dataset
from anonmed.ml.metrics.example import ExampleCountMetric
from anonmed.ml.models.GLiNER2 import DEFAULT_ENTITY_DESCRIPTION, GLiNER2Model
from anonmed.ml.models.example import ExamplePIIModel
from anonmed.ml.outputs import build_run_instance_dir
from anonmed.ml.pipelines.GLiNER2_TrH_Tests import (
    DEFAULT_PROMPTS,
    _best_feasible_trial,
    _resolve_trials_count,
    _threshold_values,
    build_parser,
)
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
                    "AnnotationSet, Dataset, GLiNER2Model, Metric, PIIModel, "
                    "TextDocument, evaluate; "
                    "from anonmed.ml.core import Case, DatasetSnapshotWriter, Span; "
                    "import anonmed.ml.registry; "
                    "import anonmed.ml.data.russian_pii_66k; "
                    "import anonmed.ml.models.GLiNER2; "
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

    def test_gliner2_model_maps_person_spans_to_per_annotations(self) -> None:
        class FakeSchemaBuilder:
            def __init__(self, extractor: "FakeGLiNER2Extractor") -> None:
                self.extractor = extractor

            def entities(self, schema: dict[str, object]) -> dict[str, object]:
                self.extractor.schema = schema
                return schema

        class FakeGLiNER2Extractor:
            def create_schema(self) -> FakeSchemaBuilder:
                return FakeSchemaBuilder(self)

            def extract(
                self,
                text: str,
                schema: dict[str, object],
                **kwargs: object,
            ) -> dict[str, object]:
                self.text = text
                self.extract_schema = schema
                self.kwargs = kwargs
                entity_text = "Иванов Иван Иванович"
                begin = text.index(entity_text)
                return {
                    "entities": {
                        "person": [
                            {
                                "text": entity_text,
                                "start": begin,
                                "end": begin + len(entity_text),
                            }
                        ]
                    }
                }

        extractor = FakeGLiNER2Extractor()
        model = GLiNER2Model(extractor=extractor)
        role = Role(name="text")
        text = "Пациент Иванов Иван Иванович пришел на прием."
        document = TextDocument(
            lines=(TextLine(idx=0, role=role, text=text),),
            sample_id="sample-1",
        )

        prediction = model.predict(document)

        self.assertEqual(
            extractor.schema,
            {
                "person": {
                    "description": DEFAULT_ENTITY_DESCRIPTION,
                    "dtype": "list",
                    "threshold": 0.5,
                }
            },
        )
        self.assertIs(extractor.extract_schema, extractor.schema)
        self.assertEqual(extractor.kwargs, {"include_spans": True, "threshold": 0.5})
        self.assertEqual(prediction.idx, "sample-1")
        self.assertEqual(len(prediction.lines), 1)
        self.assertEqual(len(prediction.lines[0].spans), 1)
        span = prediction.lines[0].spans[0]
        self.assertEqual(span.label, "PER")
        self.assertEqual(span.data, "Иванов Иван Иванович")
        self.assertEqual(text[span.begin:span.end], "Иванов Иван Иванович")

    def test_gliner2_optuna_pipeline_imports_without_optuna_runtime(self) -> None:
        parser = build_parser()

        self.assertGreaterEqual(len(DEFAULT_PROMPTS), 3)
        self.assertEqual(parser.parse_args(["--n-trials", "2"]).n_trials, 2)
        self.assertEqual(parser.parse_args([]).sampler, "grid")
        self.assertFalse(parser.parse_args([]).no_progress)
        self.assertFalse(parser.parse_args([]).document_progress)
        self.assertTrue(parser.parse_args(["--no-progress"]).no_progress)
        self.assertTrue(parser.parse_args(["--document-progress"]).document_progress)

    def test_gliner2_grid_search_defaults_to_full_search_space(self) -> None:
        threshold_values = _threshold_values(0.05, 0.15, 0.05)

        self.assertEqual(threshold_values, [0.05, 0.1, 0.15])
        self.assertEqual(
            _resolve_trials_count(
                n_trials=None,
                sampler_name="grid",
                threshold_values=threshold_values,
                prompts_count=4,
            ),
            12,
        )

    def test_gliner2_threshold_search_prefers_precision_under_recall_constraint(self) -> None:
        trials = [
            SimpleNamespace(
                number=0,
                user_attrs={"precision": 0.99, "recall": 0.75, "f1": 0.85},
            ),
            SimpleNamespace(
                number=1,
                user_attrs={"precision": 0.91, "recall": 0.92, "f1": 0.915},
            ),
            SimpleNamespace(
                number=2,
                user_attrs={"precision": 0.86, "recall": 0.98, "f1": 0.915},
            ),
        ]

        best_trial = _best_feasible_trial(trials, min_recall=0.9)

        self.assertIsNotNone(best_trial)
        self.assertEqual(best_trial.number, 1)

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
        self.assertRegex(
            payload["instance"]["evaluation_snapshot_json"],
            r"^instance/example/.+/evaluation_snapshot\.json$",
        )
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
