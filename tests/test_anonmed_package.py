from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

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
