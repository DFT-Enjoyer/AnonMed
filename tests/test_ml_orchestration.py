from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from anonmed.ml.config import pipeline_config_from_mapping
from anonmed.ml.core.types import (
    AnnotationSet,
    AnnotationSetLine,
    Case,
    Role,
    Span,
    TextDocument,
    TextLine,
)
from anonmed.ml.factory import evaluate
from anonmed.ml.data.example import build_example_dataset
from anonmed.ml.metrics.example import ExampleCountMetric
from anonmed.ml.models.GLiNER2 import DEFAULT_ENTITY_DESCRIPTION, GLiNER2Model
from anonmed.ml.models.example import ExamplePIIModel
from anonmed.ml.outputs import build_run_instance_dir
from anonmed.ml.pipelines.GLiNER2 import build_parser as build_gliner2_parser
from anonmed.ml.pipelines.GLiNER2_TrH_Tests import (
    DEFAULT_PROMPTS,
    _best_feasible_trial,
    _error_examples,
    _format_error_examples,
    _pipeline_logger,
    _require_metric_names,
    _resolve_trials_count,
    _threshold_values,
    build_parser,
)
from anonmed.ml.pipelines.terminal import print_metrics_block
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

    def test_terminal_metric_block_prints_bordered_metrics(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO

        buffer = StringIO()
        with redirect_stdout(buffer):
            print_metrics_block({"metric": {"value": 1.0}}, title="TEST METRICS")

        output = buffer.getvalue()
        self.assertIn("=" * 72, output)
        self.assertIn("TEST METRICS", output)
        self.assertIn("metric: {'value': 1.0}", output)

    def test_gliner2_threshold_logger_can_be_disabled(self) -> None:
        from contextlib import redirect_stderr
        from io import StringIO

        buffer = StringIO()
        with redirect_stderr(buffer):
            _pipeline_logger(False)("hidden")

        self.assertEqual(buffer.getvalue(), "")

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
        self.assertEqual(parser.parse_args(["--error-examples", "2"]).error_examples, 2)

    def test_regular_gliner2_pipeline_imports_without_runtime_dependencies(self) -> None:
        parser = build_gliner2_parser()

        self.assertEqual(parser.parse_args([]).error_examples, 5)
        self.assertTrue(parser.parse_args(["--no-progress"]).no_progress)

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

    def test_gliner2_threshold_search_requires_soft_precision_and_hard_recall(self) -> None:
        with self.assertRaisesRegex(ValueError, "entity_soft_precision"):
            _require_metric_names([ExampleCountMetric()], ("entity_soft_precision", "entity_hard_recall"))

    def test_gliner2_threshold_search_prefers_soft_precision_with_hard_recall_constraint(
        self,
    ) -> None:
        trials = [
            SimpleNamespace(
                number=0,
                user_attrs={
                    "objective_value": 0.99,
                    "soft_precision": 0.99,
                    "hard_recall": 0.75,
                    "soft_f1": 0.9,
                    "hard_f1": 0.85,
                },
            ),
            SimpleNamespace(
                number=1,
                user_attrs={
                    "objective_value": 0.91,
                    "soft_precision": 0.91,
                    "hard_recall": 0.92,
                    "soft_f1": 0.92,
                    "hard_f1": 0.915,
                },
            ),
            SimpleNamespace(
                number=2,
                user_attrs={
                    "objective_value": 0.95,
                    "soft_precision": 0.95,
                    "hard_recall": 0.91,
                    "soft_f1": 0.9,
                    "hard_f1": 0.88,
                },
            ),
        ]

        best_trial = _best_feasible_trial(trials, min_recall=0.9)

        self.assertIsNotNone(best_trial)
        self.assertEqual(best_trial.number, 2)

    def test_gliner2_error_examples_show_false_positive_and_false_negative(self) -> None:
        role = Role(name="text")
        text = "Иванов встретил Петрова."
        document = TextDocument(
            lines=(TextLine(idx=0, role=role, text=text),),
            sample_id="sample-1",
        )
        target = AnnotationSet(
            lines=(
                AnnotationSetLine(
                    idx=0,
                    role=role,
                    spans=[
                        Span(line_idx=0, begin=0, end=6, label="PER", data="Иванов"),
                    ],
                ),
            ),
            idx="sample-1",
        )
        prediction = AnnotationSet(
            lines=(
                AnnotationSetLine(
                    idx=0,
                    role=role,
                    spans=[
                        Span(line_idx=0, begin=16, end=23, label="PER", data="Петрова"),
                    ],
                ),
            ),
            idx="sample-1",
        )

        examples = _error_examples((Case(document=document, target=target),), (prediction,), limit=1)
        formatted = _format_error_examples(examples)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["false_negatives"][0]["text"], "Иванов")
        self.assertEqual(examples[0]["false_positives"][0]["text"], "Петрова")
        self.assertIn("false negatives", formatted)
        self.assertIn("Иванов", formatted)
        self.assertIn("Петрова", formatted)

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

    def test_gt_asr_dataset_loads_fio_annotations_as_per_spans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "gt_asr.jsonl"
            row = {
                "id": 7,
                "split": "val",
                "value": "пациент иванов иван пришел",
                "annotations": [
                    {
                        "start": 8,
                        "end": 19,
                        "type": "ФИО",
                        "text": "иванов иван",
                    },
                    {
                        "start": 20,
                        "end": 26,
                        "type": "АДРЕС",
                        "text": "пришел",
                    },
                ],
            }
            dataset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            config = pipeline_config_from_mapping(
                {
                    "dataset": {
                        "id": "gt_asr",
                        "params": {"path": str(dataset_path), "split": "val"},
                    },
                    "model": "example",
                    "metrics": ["example_count"],
                }
            )

            dataset = build_dataset(config.dataset)

        self.assertEqual(len(dataset.cases), 1)
        case = dataset.cases[0]
        self.assertEqual(case.document.sample_id, "7")
        spans = case.target.lines[0].spans
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].label, "PER")
        self.assertEqual(spans[0].data, "иванов иван")
        self.assertEqual(spans[0].begin, 8)
        self.assertEqual(spans[0].end, 19)

    def test_final_with_newlines_dataset_loads_name_spans_as_per_spans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "final_with_newlines.jsonl"
            row = {
                "id": 11,
                "text": "строка\nиванов иван пришел",
                "spans": [
                    {
                        "begin": 7,
                        "end": 18,
                        "label": "name",
                        "data": "иванов иван",
                    },
                    {
                        "begin": 19,
                        "end": 25,
                        "label": "passport",
                        "data": "пришел",
                    },
                ],
            }
            dataset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            config = pipeline_config_from_mapping(
                {
                    "dataset": {
                        "id": "final_with_newlines",
                        "params": {"path": str(dataset_path)},
                    },
                    "model": "example",
                    "metrics": ["example_count"],
                }
            )

            dataset = build_dataset(config.dataset)

        self.assertEqual(len(dataset.cases), 1)
        case = dataset.cases[0]
        self.assertEqual(case.document.sample_id, "11")
        self.assertEqual(case.document.lines[0].text, "строка\nиванов иван пришел")
        spans = case.target.lines[0].spans
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].label, "PER")
        self.assertEqual(spans[0].data, "иванов иван")
        self.assertEqual(spans[0].begin, 7)
        self.assertEqual(spans[0].end, 18)

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
