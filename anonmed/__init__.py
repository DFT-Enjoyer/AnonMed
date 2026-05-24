from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

_PACKAGE_ROOT: Path = Path(__file__).resolve().parent
_SOURCE_PACKAGE_ROOT: Path = _PACKAGE_ROOT.parent / "src" / "anonmed"

if not _SOURCE_PACKAGE_ROOT.is_dir():
    message: str = f"Expected source package directory at {_SOURCE_PACKAGE_ROOT!s}."
    raise ModuleNotFoundError(message)

__path__.append(str(_SOURCE_PACKAGE_ROOT))

__all__: list[str] = [
    "ASRNormalizationPipeline",
    "ASRNormalizationResult",
    "ASRTextPreprocessingPipeline",
    "ASRTextPreprocessingResult",
    "NumericPIIMatch",
    "NumericPIIPipelineResult",
    "NumericPIIRule",
    "NumericPIIType",
    "build_default_numeric_rules",
    "find_numeric_pii",
    "mask_numeric_pii",
    "run_numeric_pii_pipeline",
    "run_asr_normalization",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ASRNormalizationPipeline": ("anonmed.preprocessing", "ASRNormalizationPipeline"),
    "ASRNormalizationResult": ("anonmed.preprocessing", "ASRNormalizationResult"),
    "ASRTextPreprocessingPipeline": ("anonmed.preprocessing", "ASRTextPreprocessingPipeline"),
    "ASRTextPreprocessingResult": ("anonmed.preprocessing", "ASRTextPreprocessingResult"),
    "NumericPIIMatch": ("anonmed.anonymization", "NumericPIIMatch"),
    "NumericPIIPipelineResult": ("anonmed.anonymization", "NumericPIIPipelineResult"),
    "NumericPIIRule": ("anonmed.anonymization", "NumericPIIRule"),
    "NumericPIIType": ("anonmed.anonymization", "NumericPIIType"),
    "build_default_numeric_rules": ("anonmed.anonymization", "build_default_numeric_rules"),
    "find_numeric_pii": ("anonmed.anonymization", "find_numeric_pii"),
    "mask_numeric_pii": ("anonmed.anonymization", "mask_numeric_pii"),
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
