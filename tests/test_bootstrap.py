from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class BootstrapTests(unittest.TestCase):
    def test_import_works_from_repository_root(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        command: list[str] = [sys.executable, "-c", "import asr_integer_extractor"]
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


__all__: list[str] = []
