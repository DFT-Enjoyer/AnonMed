from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT: Path = Path(__file__).resolve().parent
_SOURCE_PACKAGE_ROOT: Path = _PACKAGE_ROOT.parent / "src" / "asr_integer_extractor"

if not _SOURCE_PACKAGE_ROOT.is_dir():
    message: str = f"Expected source package directory at {_SOURCE_PACKAGE_ROOT!s}."
    raise ModuleNotFoundError(message)

__path__.append(str(_SOURCE_PACKAGE_ROOT))

from .disfluency import (  # noqa: E402
    CleanedText,
    DisfluencyFilter,
    DisfluencyFilterConfig,
    RemovedSpan,
    remove_disfluencies,
)
from .extractor import (  # noqa: E402
    IntegerExtractor,
    extract_integers,
    replace_integer_spans,
    replace_spans,
)
from .models import (  # noqa: E402
    Candidate,
    ExtractorConfig,
    IntegerSpan,
    LexicalMatch,
    NumericToken,
    Token,
)
from .pipeline import (  # noqa: E402
    ASRNormalizationPipeline,
    ASRNormalizationResult,
    run_asr_normalization,
)
from .punctuation import (  # noqa: E402
    ProtectedPunctuationSpan,
    PunctuationCleanedText,
    PunctuationFilterConfig,
    PunctuationRemovalConfig,
    PunctuationRemover,
    RemovedPunctuationSpan,
    remove_punctuation,
)

__all__: list[str] = [
    "ASRNormalizationPipeline",
    "ASRNormalizationResult",
    "Candidate",
    "CleanedText",
    "DisfluencyFilter",
    "DisfluencyFilterConfig",
    "ExtractorConfig",
    "IntegerExtractor",
    "IntegerSpan",
    "LexicalMatch",
    "NumericToken",
    "ProtectedPunctuationSpan",
    "PunctuationCleanedText",
    "PunctuationFilterConfig",
    "PunctuationRemovalConfig",
    "PunctuationRemover",
    "RemovedPunctuationSpan",
    "RemovedSpan",
    "Token",
    "extract_integers",
    "remove_punctuation",
    "remove_disfluencies",
    "replace_integer_spans",
    "replace_spans",
    "run_asr_normalization",
]
