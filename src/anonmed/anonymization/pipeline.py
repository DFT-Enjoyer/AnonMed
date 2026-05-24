from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from anonmed.anonymization.numeric_pii import NumericPIIMatch, NumericPIIType, find_numeric_pii
from anonmed.preprocessing.asr.alignment import SourceSpan, TextAlignment
from anonmed.preprocessing import ASRNormalizationResult, run_asr_normalization


@dataclass(frozen=True, slots=True)
class AlignedNumericPIIMatch:
    pii_type: NumericPIIType
    start: int
    end: int
    value: str
    normalized_start: int
    normalized_end: int
    normalized_value: str
    confidence: float
    rule_id: str
    context: str
    metadata: Mapping[str, object]
    normalized_match: NumericPIIMatch


@dataclass(frozen=True, slots=True)
class NumericPIIPipelineResult:
    original_text: str
    preprocessing_result: ASRNormalizationResult
    matches: tuple[AlignedNumericPIIMatch, ...]
    normalized_matches: tuple[NumericPIIMatch, ...]
    masked_normalized_text: str
    masked_original_text: str
    restored_safe_text: str

    @property
    def masked_text(self) -> str:
        return self.masked_normalized_text


def run_numeric_pii_pipeline(
    text: str,
    replacement_by_type: Mapping[NumericPIIType, str] | None = None,
    deduplicate_repetitions: bool = False,
    normalize_document_numbers: bool = False,
) -> NumericPIIPipelineResult:
    preprocessing_result: ASRNormalizationResult = run_asr_normalization(
        text,
        deduplicate_repetitions=deduplicate_repetitions,
        normalize_document_numbers=normalize_document_numbers,
    )
    normalized_matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
        preprocessing_result.normalized_text
    )
    alignment: TextAlignment | None = preprocessing_result.normalized_to_original_alignment
    if alignment is None:
        raise ValueError("ASR normalization result has no normalized-to-original alignment")

    matches: tuple[AlignedNumericPIIMatch, ...] = tuple(
        _align_match(text, alignment, normalized_match)
        for normalized_match in normalized_matches
    )

    replacements: Mapping[NumericPIIType, str] = replacement_by_type or {}
    normalized_parts: list[str] = []
    normalized_cursor: int = 0
    for match in normalized_matches:
        replacement: str = replacements.get(match.pii_type, f"[{match.pii_type}]")
        normalized_parts.append(preprocessing_result.normalized_text[normalized_cursor:match.start])
        normalized_parts.append(replacement)
        normalized_cursor = match.end
    normalized_parts.append(preprocessing_result.normalized_text[normalized_cursor:])
    masked_normalized_text: str = "".join(normalized_parts)

    original_parts: list[str] = []
    original_cursor: int = 0
    for match in matches:
        replacement = replacements.get(match.pii_type, f"[{match.pii_type}]")
        if match.start < original_cursor:
            continue
        original_parts.append(text[original_cursor:match.start])
        original_parts.append(replacement)
        original_cursor = match.end
    original_parts.append(text[original_cursor:])
    masked_original_text: str = "".join(original_parts)

    return NumericPIIPipelineResult(
        original_text=text,
        preprocessing_result=preprocessing_result,
        matches=matches,
        normalized_matches=normalized_matches,
        masked_normalized_text=masked_normalized_text,
        masked_original_text=masked_original_text,
        restored_safe_text=masked_original_text,
    )


def _align_match(
    original_text: str,
    alignment: TextAlignment,
    normalized_match: NumericPIIMatch,
) -> AlignedNumericPIIMatch:
    source_span: SourceSpan = alignment.source_span_for_target_span(
        normalized_match.start,
        normalized_match.end,
    )
    original_value: str = original_text[source_span.start:source_span.end]
    return AlignedNumericPIIMatch(
        pii_type=normalized_match.pii_type,
        start=source_span.start,
        end=source_span.end,
        value=original_value,
        normalized_start=normalized_match.start,
        normalized_end=normalized_match.end,
        normalized_value=normalized_match.normalized_value,
        confidence=normalized_match.confidence,
        rule_id=normalized_match.rule_id,
        context=normalized_match.context,
        metadata=normalized_match.metadata,
        normalized_match=normalized_match,
    )


__all__: list[str] = [
    "AlignedNumericPIIMatch",
    "NumericPIIPipelineResult",
    "run_numeric_pii_pipeline",
]
