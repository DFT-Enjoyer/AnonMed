from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from anonmed.anonymization.numeric_pii import NumericPIIMatch, NumericPIIType, find_numeric_pii
from anonmed.preprocessing import ASRNormalizationResult, run_asr_normalization


@dataclass(frozen=True, slots=True)
class NumericPIIPipelineResult:
    original_text: str
    preprocessing_result: ASRNormalizationResult
    matches: tuple[NumericPIIMatch, ...]
    masked_text: str


def run_numeric_pii_pipeline(
    text: str,
    replacement_by_type: Mapping[NumericPIIType, str] | None = None,
) -> NumericPIIPipelineResult:
    preprocessing_result: ASRNormalizationResult = run_asr_normalization(text)
    matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(preprocessing_result.normalized_text)

    replacements: Mapping[NumericPIIType, str] = replacement_by_type or {}
    parts: list[str] = []
    cursor: int = 0
    for match in matches:
        replacement: str = replacements.get(match.pii_type, f"[{match.pii_type}]")
        parts.append(preprocessing_result.normalized_text[cursor:match.start])
        parts.append(replacement)
        cursor = match.end
    parts.append(preprocessing_result.normalized_text[cursor:])
    masked_text: str = "".join(parts)

    return NumericPIIPipelineResult(
        original_text=text,
        preprocessing_result=preprocessing_result,
        matches=matches,
        masked_text=masked_text,
    )


__all__: list[str] = ["NumericPIIPipelineResult", "run_numeric_pii_pipeline"]
