from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT: Path = Path(__file__).resolve().parent
_SOURCE_PACKAGE_ROOT: Path = _PACKAGE_ROOT.parent / "src" / "asr_integer_extractor"

if not _SOURCE_PACKAGE_ROOT.is_dir():
    message: str = f"Expected source package directory at {_SOURCE_PACKAGE_ROOT!s}."
    raise ModuleNotFoundError(message)

__path__.append(str(_SOURCE_PACKAGE_ROOT))

from .extractor import IntegerExtractor, extract_integers, replace_integer_spans, replace_spans
from .models import Candidate, ExtractorConfig, IntegerSpan, LexicalMatch, NumericToken, Token

__all__: list[str] = [
    "Candidate",
    "ExtractorConfig",
    "IntegerExtractor",
    "IntegerSpan",
    "LexicalMatch",
    "NumericToken",
    "Token",
    "extract_integers",
    "replace_integer_spans",
    "replace_spans",
]
