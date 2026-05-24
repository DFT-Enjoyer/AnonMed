from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT: Path = Path(__file__).resolve().parent
_SOURCE_PACKAGE_ROOT: Path = _PACKAGE_ROOT.parent / "src" / "anonmed"

if not _SOURCE_PACKAGE_ROOT.is_dir():
    message: str = f"Expected source package directory at {_SOURCE_PACKAGE_ROOT!s}."
    raise ModuleNotFoundError(message)

__path__.append(str(_SOURCE_PACKAGE_ROOT))

from .preprocessing import (  # noqa: E402
    ASRNormalizationPipeline,
    ASRNormalizationResult,
    ASRTextPreprocessingPipeline,
    ASRTextPreprocessingResult,
    run_asr_normalization,
)
from .anonymization import (  # noqa: E402
    MaskingStrategy,
    NumericPIIMatch,
    NumericPIIPipelineResult,
    NumericPIIRule,
    NumericPIIType,
    PIICandidate,
    PostProcessedEntityGroup,
    PostProcessedPIIMention,
    PostProcessingMode,
    PostProcessingResult,
    build_default_numeric_rules,
    collect_numeric_pii_candidates,
    find_numeric_pii,
    mask_numeric_pii,
    resolve_pii_candidates,
    run_numeric_post_processing,
    run_numeric_pii_pipeline,
)

__all__: list[str] = [
    "ASRNormalizationPipeline",
    "ASRNormalizationResult",
    "ASRTextPreprocessingPipeline",
    "ASRTextPreprocessingResult",
    "MaskingStrategy",
    "NumericPIIMatch",
    "NumericPIIPipelineResult",
    "NumericPIIRule",
    "NumericPIIType",
    "PIICandidate",
    "PostProcessedEntityGroup",
    "PostProcessedPIIMention",
    "PostProcessingMode",
    "PostProcessingResult",
    "build_default_numeric_rules",
    "collect_numeric_pii_candidates",
    "find_numeric_pii",
    "mask_numeric_pii",
    "resolve_pii_candidates",
    "run_numeric_post_processing",
    "run_numeric_pii_pipeline",
    "run_asr_normalization",
]
