from __future__ import annotations

from anonmed.anonymization.numeric_pii import (
    NumericPIIMatch,
    NumericPIIRule,
    NumericPIIType,
    build_default_numeric_rules,
    collect_numeric_pii_candidates,
    find_numeric_pii,
    mask_numeric_pii,
    normalize_numeric_pii_value,
)
from anonmed.anonymization.post_processing import (
    MaskingStrategy,
    PIIEntityType,
    PIICandidate,
    PostProcessedEntityGroup,
    PostProcessedPIIMention,
    PostProcessingMode,
    PostProcessingResult,
    candidate_from_numeric_match,
    resolve_pii_candidates,
    run_numeric_post_processing,
    run_pii_post_processing,
)
from anonmed.anonymization.restoration import (
    OriginalTextRestorer,
    RestoredTextResult,
    restore_safe_original_text,
)
from anonmed.anonymization.pipeline import (
    AlignedNumericPIIMatch,
    NumericPIIPipelineResult,
    run_numeric_pii_pipeline,
)

__all__: list[str] = [
    "AlignedNumericPIIMatch",
    "MaskingStrategy",
    "NumericPIIMatch",
    "NumericPIIRule",
    "NumericPIIType",
    "NumericPIIPipelineResult",
    "OriginalTextRestorer",
    "PIIEntityType",
    "RestoredTextResult",
    "restore_safe_original_text",
    "PIICandidate",
    "PostProcessedEntityGroup",
    "PostProcessedPIIMention",
    "PostProcessingMode",
    "PostProcessingResult",
    "build_default_numeric_rules",
    "candidate_from_numeric_match",
    "collect_numeric_pii_candidates",
    "find_numeric_pii",
    "mask_numeric_pii",
    "normalize_numeric_pii_value",
    "resolve_pii_candidates",
    "run_numeric_post_processing",
    "run_pii_post_processing",
    "run_numeric_pii_pipeline",
]
