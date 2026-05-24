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
    "MLDetectionConfig",
    "OriginalTextRestorer",
    "NumericPIIMatch",
    "NumericPIIPipelineResult",
    "NumericPIIRule",
    "NumericPIIType",
    "PIIAnonymizationResult",
    "PIIAnonymizer",
    "PIIAnonymizerConfig",
    "PIICandidate",
    "PIIEntityType",
    "PostProcessedEntityGroup",
    "PostProcessedPIIMention",
    "PostProcessingMode",
    "PostProcessingConfig",
    "PostProcessingResult",
    "PreprocessingConfig",
    "RestoredTextResult",
    "ResolvedMLDetectionConfig",
    "ResolvedPIIAnonymizerConfig",
    "ResolvedPostProcessingConfig",
    "ResolvedPreprocessingConfig",
    "ResolvedRuleDetectionConfig",
    "RuleDetectionConfig",
    "anonymize",
    "anonymize_pii",
    "build_default_numeric_rules",
    "candidate_from_numeric_match",
    "collect_numeric_pii_candidates",
    "find_numeric_pii",
    "mask_numeric_pii",
    "resolve_pii_candidates",
    "restore_safe_original_text",
    "run_numeric_post_processing",
    "run_numeric_pii_pipeline",
    "run_pii_post_processing",
    "run_asr_normalization",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AlignedNumericPIIMatch": ("anonmed.anonymization", "AlignedNumericPIIMatch"),
    "ASRNormalizationPipeline": ("anonmed.preprocessing", "ASRNormalizationPipeline"),
    "ASRNormalizationResult": ("anonmed.preprocessing", "ASRNormalizationResult"),
    "ASRTextPreprocessingPipeline": ("anonmed.preprocessing", "ASRTextPreprocessingPipeline"),
    "ASRTextPreprocessingResult": ("anonmed.preprocessing", "ASRTextPreprocessingResult"),
    "MaskingStrategy": ("anonmed.anonymization", "MaskingStrategy"),
    "MLDetectionConfig": ("anonmed.anonymizer", "MLDetectionConfig"),
    "NumericPIIMatch": ("anonmed.anonymization", "NumericPIIMatch"),
    "NumericPIIPipelineResult": ("anonmed.anonymization", "NumericPIIPipelineResult"),
    "NumericPIIRule": ("anonmed.anonymization", "NumericPIIRule"),
    "NumericPIIType": ("anonmed.anonymization", "NumericPIIType"),
    "OriginalTextRestorer": ("anonmed.anonymization", "OriginalTextRestorer"),
    "PIIAnonymizationResult": ("anonmed.anonymizer", "PIIAnonymizationResult"),
    "PIIAnonymizer": ("anonmed.anonymizer", "PIIAnonymizer"),
    "PIIAnonymizerConfig": ("anonmed.anonymizer", "PIIAnonymizerConfig"),
    "PIICandidate": ("anonmed.anonymization", "PIICandidate"),
    "PIIEntityType": ("anonmed.anonymization", "PIIEntityType"),
    "PostProcessedEntityGroup": ("anonmed.anonymization", "PostProcessedEntityGroup"),
    "PostProcessedPIIMention": ("anonmed.anonymization", "PostProcessedPIIMention"),
    "PostProcessingMode": ("anonmed.anonymization", "PostProcessingMode"),
    "PostProcessingConfig": ("anonmed.anonymizer", "PostProcessingConfig"),
    "PostProcessingResult": ("anonmed.anonymization", "PostProcessingResult"),
    "PreprocessingConfig": ("anonmed.anonymizer", "PreprocessingConfig"),
    "ResolvedMLDetectionConfig": ("anonmed.anonymizer", "ResolvedMLDetectionConfig"),
    "ResolvedPIIAnonymizerConfig": ("anonmed.anonymizer", "ResolvedPIIAnonymizerConfig"),
    "ResolvedPostProcessingConfig": ("anonmed.anonymizer", "ResolvedPostProcessingConfig"),
    "ResolvedPreprocessingConfig": ("anonmed.anonymizer", "ResolvedPreprocessingConfig"),
    "ResolvedRuleDetectionConfig": ("anonmed.anonymizer", "ResolvedRuleDetectionConfig"),
    "RuleDetectionConfig": ("anonmed.anonymizer", "RuleDetectionConfig"),
    "anonymize": ("anonmed.anonymizer", "anonymize"),
    "anonymize_pii": ("anonmed.anonymizer", "anonymize_pii"),
    "build_default_numeric_rules": ("anonmed.anonymization", "build_default_numeric_rules"),
    "candidate_from_numeric_match": ("anonmed.anonymization", "candidate_from_numeric_match"),
    "collect_numeric_pii_candidates": ("anonmed.anonymization", "collect_numeric_pii_candidates"),
    "find_numeric_pii": ("anonmed.anonymization", "find_numeric_pii"),
    "mask_numeric_pii": ("anonmed.anonymization", "mask_numeric_pii"),
    "RestoredTextResult": ("anonmed.anonymization", "RestoredTextResult"),
    "resolve_pii_candidates": ("anonmed.anonymization", "resolve_pii_candidates"),
    "restore_safe_original_text": ("anonmed.anonymization", "restore_safe_original_text"),
    "run_numeric_post_processing": ("anonmed.anonymization", "run_numeric_post_processing"),
    "run_numeric_pii_pipeline": ("anonmed.anonymization", "run_numeric_pii_pipeline"),
    "run_pii_post_processing": ("anonmed.anonymization", "run_pii_post_processing"),
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
