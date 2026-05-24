from __future__ import annotations

from importlib import import_module
from typing import Any

__all__: list[str] = [
    "AlignedNumericPIIMatch",
    "ASRNormalizationPipeline",
    "ASRNormalizationResult",
    "ASRTextPreprocessingPipeline",
    "ASRTextPreprocessingResult",
    "MaskingStrategy",
    "OriginalTextRestorer",
    "NumericPIIMatch",
    "NumericPIIPipelineResult",
    "NumericPIIRule",
    "NumericPIIType",
    "PIICandidate",
    "PostProcessedEntityGroup",
    "PostProcessedPIIMention",
    "PostProcessingMode",
    "PostProcessingResult",
    "RestoredTextResult",
    "build_default_numeric_rules",
    "collect_numeric_pii_candidates",
    "find_numeric_pii",
    "mask_numeric_pii",
    "resolve_pii_candidates",
    "restore_safe_original_text",
    "run_numeric_post_processing",
    "run_numeric_pii_pipeline",
    "run_asr_normalization",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AlignedNumericPIIMatch": ("anonmed.anonymization", "AlignedNumericPIIMatch"),
    "ASRNormalizationPipeline": ("anonmed.preprocessing", "ASRNormalizationPipeline"),
    "ASRNormalizationResult": ("anonmed.preprocessing", "ASRNormalizationResult"),
    "ASRTextPreprocessingPipeline": ("anonmed.preprocessing", "ASRTextPreprocessingPipeline"),
    "ASRTextPreprocessingResult": ("anonmed.preprocessing", "ASRTextPreprocessingResult"),
    "NumericPIIMatch": ("anonmed.anonymization", "NumericPIIMatch"),
    "NumericPIIPipelineResult": ("anonmed.anonymization", "NumericPIIPipelineResult"),
    "NumericPIIRule": ("anonmed.anonymization", "NumericPIIRule"),
    "NumericPIIType": ("anonmed.anonymization", "NumericPIIType"),
    "OriginalTextRestorer": ("anonmed.anonymization", "OriginalTextRestorer"),
    "build_default_numeric_rules": ("anonmed.anonymization", "build_default_numeric_rules"),
    "find_numeric_pii": ("anonmed.anonymization", "find_numeric_pii"),
    "mask_numeric_pii": ("anonmed.anonymization", "mask_numeric_pii"),
    "RestoredTextResult": ("anonmed.anonymization", "RestoredTextResult"),
    "restore_safe_original_text": ("anonmed.anonymization", "restore_safe_original_text"),
    "run_numeric_pii_pipeline": ("anonmed.anonymization", "run_numeric_pii_pipeline"),
    "run_asr_normalization": ("anonmed.preprocessing", "run_asr_normalization"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

