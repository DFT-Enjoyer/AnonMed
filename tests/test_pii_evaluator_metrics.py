from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Mapping, cast
import unittest


class PIIEvaluatorMetricsTests(unittest.TestCase):
    def test_evaluator_report_contains_privacy_alignment_and_span_metrics(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temporary_directory:
            command: list[str] = [
                sys.executable,
                "scripts/evaluate_pii_metrics.py",
                "tmp_eval_sample.jsonl",
                "--json",
                "--artifacts-root",
                temporary_directory,
            ]
            completed_process: subprocess.CompletedProcess[str] = subprocess.run(
                command,
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed_process.returncode, 0, completed_process.stderr)
        report: dict[str, object] = json.loads(completed_process.stdout)
        self.assertIn("span", report)
        self.assertIn("character", report)
        self.assertIn("privacy_output", report)
        self.assertIn("alignment_projection", report)
        self.assertIn("ml_metrics", report)
        self.assertEqual(len(report["pii_types"]), 15)
        privacy_output: Mapping[str, object] = cast(Mapping[str, object], report["privacy_output"])
        self.assertEqual(privacy_output["direct_identifier_leakage_rate"], 0.0)
        self.assertEqual(privacy_output["document_level_privacy_pass_rate"], 1.0)


__all__: list[str] = []
