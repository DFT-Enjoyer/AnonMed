from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from anonmed.anonymization import find_numeric_pii, mask_numeric_pii, run_numeric_pii_pipeline
from anonmed.preprocessing import (
    ASRTextPreprocessingPipeline,
    PunctuationFilterConfig,
    run_asr_normalization,
)


class AnonMedPackageTests(unittest.TestCase):
    def test_project_level_import_works_from_repository_root(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        command: list[str] = [sys.executable, "-c", "import anonmed"]
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            command,
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

    def test_new_pipeline_alias_runs(self) -> None:
        pipeline = ASRTextPreprocessingPipeline()
        result = pipeline.run("ну, номер один два три")
        self.assertEqual(result.cleaned_text, "номер один два три")
        self.assertEqual(result.normalized_text, "номер 123")

    def test_project_level_preprocessing_entrypoint_works(self) -> None:
        result = run_asr_normalization(
            "номер: один два",
            punctuation_config=PunctuationFilterConfig(enabled=False),
        )
        self.assertEqual(result.cleaned_text, "номер: один два")

    def test_anonymization_exports_numeric_pii_helpers(self) -> None:
        matches = find_numeric_pii("телефон 89131234567")
        self.assertEqual(len(matches), 1)
        self.assertEqual(mask_numeric_pii("телефон 89131234567"), "телефон [PHONE]")

    def test_project_level_numeric_pii_pipeline_runs_with_preprocessing(self) -> None:
        result = run_numeric_pii_pipeline(
            "телефон восемь девять один три один два три четыре пять шесть семь"
        )
        self.assertEqual(result.preprocessing_result.normalized_text, "телефон 89131234567")
        self.assertEqual([match.normalized_value for match in result.matches], ["+79131234567"])
        self.assertEqual(result.masked_text, "телефон [PHONE]")

    def test_cli_entrypoint_works_for_new_package_name(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        command: list[str] = [
            sys.executable,
            "-m",
            "anonmed.cli",
            "ну,",
            "номер:",
            "один",
            "два",
            "--run",
        ]
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            command,
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
        self.assertEqual(completed_process.stdout.strip(), "номер 12")


__all__: list[str] = []
