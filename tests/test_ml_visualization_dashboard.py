from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from anonmed.ml.visualization.dashboard import build_dashboard_manifest, write_dashboard


class MLVisualizationDashboardTests(unittest.TestCase):
    def test_manifest_discovers_runs_and_normalizes_report_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "instance"
            _write_run(
                root / "GLiNER2" / "2026-05-24_12-00-00_000001",
                {
                    "run": {"name": "GLiNER2"},
                    "dataset": {"id": "russian_pii_66k", "params": {"sample_size": 10}},
                    "model": {"id": "GLiNER2", "params": {}},
                    "samples_count": 10,
                    "metric_results": {
                        "entity_hard_f1": {"value": 0.8, "tp": 8, "fp": 1, "fn": 3},
                        "coverage_percent": {
                            "coverage_percent": 90.0,
                            "over_coverage_percent": 5.0,
                        },
                    },
                },
            )
            _write_run(
                root / "legacy" / "2026-05-24_12-00-01_000001",
                {
                    "metrics": {
                        "entity_hard_precision": {
                            "value": 0.75,
                            "tp": 3,
                            "fp": 1,
                            "fn": 2,
                        }
                    },
                    "samples_count": 5,
                },
            )
            (root / "broken" / "2026-05-24_12-00-02_000001").mkdir(parents=True)
            (root / "broken" / "2026-05-24_12-00-02_000001" / "report.json").write_text(
                "{",
                encoding="utf-8",
            )

            manifest = build_dashboard_manifest(root)

        self.assertEqual(len(manifest["runs"]), 2)
        self.assertEqual(len(manifest["warnings"]), 1)
        gliner_run = next(run for run in manifest["runs"] if run["run_name"] == "GLiNER2")
        legacy_run = next(run for run in manifest["runs"] if run["run_name"] == "legacy")
        self.assertEqual(gliner_run["dataset"]["id"], "russian_pii_66k")
        self.assertEqual(_metric(gliner_run, "entity_hard_f1")["value"], 0.8)
        self.assertEqual(_metric(gliner_run, "coverage_percent")["value"], 90.0)
        self.assertEqual(_metric(legacy_run, "entity_hard_precision")["value"], 0.75)

    def test_html_excludes_raw_samples_by_default_and_includes_them_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "instance"
            run_dir = root / "example" / "2026-05-24_12-00-00_000001"
            _write_run(
                run_dir,
                {
                    "run": {"name": "example"},
                    "dataset": {"id": "example", "params": {}},
                    "model": {"id": "example", "params": {}},
                    "samples_count": 1,
                    "metric_results": {"entity_hard_f1": {"value": 1.0}},
                    "instance": {
                        "dataset_snapshot_json": str(run_dir / "dataset_snapshot.json"),
                    },
                },
            )
            _write_snapshot(run_dir / "dataset_snapshot.json", "SECRET SAMPLE TEXT")

            safe_manifest = build_dashboard_manifest(root, include_samples=False)
            safe_html = write_dashboard(safe_manifest, root / "safe.html").read_text(
                encoding="utf-8"
            )
            rich_manifest = build_dashboard_manifest(root, include_samples=True)
            rich_html = write_dashboard(rich_manifest, root / "rich.html").read_text(
                encoding="utf-8"
            )

        self.assertIn("AnonMed ML Runs", safe_html)
        self.assertIn("runsTable", safe_html)
        self.assertIn("profileChart", safe_html)
        self.assertIn("metricChart", safe_html)
        self.assertIn("Precision vs Recall", safe_html)
        self.assertIn("color-scheme: dark", safe_html)
        self.assertIn('id="tooltip"', safe_html)
        self.assertIn('data-tooltip=', safe_html)
        self.assertIn('id="chartFullscreen"', safe_html)
        self.assertIn('data-fullscreen-target="profileChart"', safe_html)
        self.assertIn("#ff70b8", safe_html)
        self.assertNotIn("SECRET SAMPLE TEXT", safe_html)
        self.assertIn("SECRET SAMPLE TEXT", rich_html)

    def test_cli_generates_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "instance"
            _write_run(
                root / "example" / "2026-05-24_12-00-00_000001",
                {
                    "samples_count": 1,
                    "metric_results": {"example_count": {"predictions_count": 1}},
                },
            )
            output = root / "dashboard.html"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "anonmed.ml.visualization.dashboard",
                    "--instance-root",
                    str(root),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            self.assertTrue(output.exists())
            self.assertIn("runs: 1", completed.stdout)
            self.assertIn(
                "example/2026-05-24_12-00-00_000001",
                output.read_text(encoding="utf-8"),
            )


def _write_run(run_dir: Path, report: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_snapshot(path: Path, text: str) -> None:
    path.write_text(
        json.dumps(
            {
                "samples_count": 1,
                "cases": [
                    {
                        "document": {
                            "sample_id": "sample-1",
                            "lines": [{"idx": 0, "text": text}],
                        },
                        "target": {
                            "lines": [
                                {
                                    "idx": 0,
                                    "spans": [
                                        {
                                            "line_idx": 0,
                                            "begin": 0,
                                            "end": 6,
                                            "label": "PER",
                                            "data": "SECRET",
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _metric(run: dict[str, object], name: str) -> dict[str, object]:
    metrics = run["metrics"]
    assert isinstance(metrics, list)
    metric = next(metric for metric in metrics if metric["name"] == name)
    assert isinstance(metric, dict)
    return metric


__all__: list[str] = []
