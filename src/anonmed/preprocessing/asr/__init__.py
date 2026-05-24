from __future__ import annotations

from anonmed.preprocessing.asr.disfluency import (
    CleanedText,
    DisfluencyFilter,
    DisfluencyFilterConfig,
    RemovedSpan,
    remove_disfluencies,
)
from anonmed.preprocessing.asr.number_extractor import (
    IntegerExtractor,
    extract_integers,
    replace_integer_spans,
    replace_spans,
)
from anonmed.preprocessing.asr.pipeline import (
    ASRNormalizationPipeline,
    ASRNormalizationResult,
    run_asr_normalization,
)
from anonmed.preprocessing.asr.punctuation import (
    ProtectedPunctuationSpan,
    PunctuationCleanedText,
    PunctuationFilterConfig,
    PunctuationRemovalConfig,
    PunctuationRemover,
    RemovedPunctuationSpan,
    remove_punctuation,
)
from anonmed.preprocessing.asr.types import (
    Candidate,
    ExtractorConfig,
    IntegerSpan,
    LexicalMatch,
    NumericToken,
    Token,
)

ASRTextPreprocessingPipeline = ASRNormalizationPipeline
ASRTextPreprocessingResult = ASRNormalizationResult

__all__: list[str] = [
    "ASRNormalizationPipeline",
    "ASRNormalizationResult",
    "ASRTextPreprocessingPipeline",
    "ASRTextPreprocessingResult",
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
    "remove_disfluencies",
    "remove_punctuation",
    "replace_integer_spans",
    "replace_spans",
    "run_asr_normalization",
]
