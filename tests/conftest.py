from __future__ import annotations

from pathlib import Path
import sys

_TESTS_ROOT: Path = Path(__file__).resolve().parent
_REPOSITORY_ROOT: Path = _TESTS_ROOT.parent
_SOURCE_ROOT: Path = _REPOSITORY_ROOT / "src"

for candidate_root in (_REPOSITORY_ROOT, _SOURCE_ROOT):
    candidate_root_text: str = str(candidate_root)
    if candidate_root.is_dir() and candidate_root_text not in sys.path:
        sys.path.insert(0, candidate_root_text)


__all__: list[str] = []
