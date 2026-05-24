from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from anonmed.anonymization.numeric_pii import (
    NumericPIIMatch,
    NumericPIIType,
    collect_numeric_pii_candidates,
)
from anonmed.anonymization.post_processing import (
    MaskingStrategy,
    PostProcessedPIIMention,
    PostProcessingMode,
    PostProcessingResult,
    run_numeric_post_processing,
)
from anonmed.preprocessing.asr.alignment import TextAlignment
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
    post_processing_result: PostProcessingResult

    @property
    def masked_text(self) -> str:
        return self.masked_normalized_text


def run_numeric_pii_pipeline(
    text: str,
    replacement_by_type: Mapping[NumericPIIType, str] | None = None,
    deduplicate_repetitions: bool = False,
    normalize_document_numbers: bool = False,
    post_processing_mode: PostProcessingMode = "production_safe",
    masking_strategy: MaskingStrategy = "type",
) -> NumericPIIPipelineResult:
    preprocessing_result: ASRNormalizationResult = run_asr_normalization(
        text,
        deduplicate_repetitions=deduplicate_repetitions,
        normalize_document_numbers=normalize_document_numbers,
    )
    normalized_candidates: tuple[NumericPIIMatch, ...] = collect_numeric_pii_candidates(
        preprocessing_result.normalized_text
    )
    alignment: TextAlignment | None = preprocessing_result.normalized_to_original_alignment
    if alignment is None:
        raise ValueError("ASR normalization result has no normalized-to-original alignment")

    post_processing_result: PostProcessingResult = run_numeric_post_processing(
        original_text=text,
        normalized_text=preprocessing_result.normalized_text,
        alignment=alignment,
        normalized_matches=normalized_candidates,
        replacement_by_type=replacement_by_type,
        mode=post_processing_mode,
        masking_strategy=masking_strategy,
    )
    normalized_matches: tuple[NumericPIIMatch, ...] = tuple(
        _normalized_match_from_mention(mention) for mention in post_processing_result.mentions
    )
    matches: tuple[AlignedNumericPIIMatch, ...] = tuple(
        _aligned_match_from_mention(mention) for mention in post_processing_result.mentions
    )

    return NumericPIIPipelineResult(
        original_text=text,
        preprocessing_result=preprocessing_result,
        matches=matches,
        normalized_matches=normalized_matches,
        masked_normalized_text=post_processing_result.masked_normalized_text,
        masked_original_text=post_processing_result.masked_original_text,
        restored_safe_text=post_processing_result.masked_original_text,
        post_processing_result=post_processing_result,
    )


def _normalized_match_from_mention(mention: PostProcessedPIIMention) -> NumericPIIMatch:
    source_match: NumericPIIMatch | None = mention.source_match
    context: str = "" if source_match is None else source_match.context
    metadata: dict[str, object] = dict(mention.metadata)
    metadata["entity_id"] = mention.entity_id
    metadata["mention_id"] = mention.mention_id
    metadata["projection_status"] = mention.projection_status
    return NumericPIIMatch(
        pii_type=mention.entity_type,
        start=mention.normalized_start,
        end=mention.normalized_end,
        value=mention.normalized_text,
        normalized_value=mention.normalized_value,
        confidence=mention.confidence,
        rule_id=mention.rule_id,
        context=context,
        metadata=metadata,
    )


def _aligned_match_from_mention(mention: PostProcessedPIIMention) -> AlignedNumericPIIMatch:
    normalized_match: NumericPIIMatch = _normalized_match_from_mention(mention)
    original_value: str = mention.original_text
    return AlignedNumericPIIMatch(
        pii_type=mention.entity_type,
        start=mention.original_start,
        end=mention.original_end,
        value=original_value,
        normalized_start=mention.normalized_start,
        normalized_end=mention.normalized_end,
        normalized_value=mention.normalized_value,
        confidence=mention.confidence,
        rule_id=mention.rule_id,
        context=normalized_match.context,
        metadata=_metadata_with_original_span(mention),
        normalized_match=normalized_match,
    )


def _metadata_with_original_span(mention: PostProcessedPIIMention) -> Mapping[str, object]:
    metadata: dict[str, object] = dict(mention.metadata)
    metadata["entity_id"] = mention.entity_id
    metadata["mention_id"] = mention.mention_id
    metadata["projection_status"] = mention.projection_status
    metadata["original_span"] = (mention.original_start, mention.original_end)
    metadata["normalized_span"] = (mention.normalized_start, mention.normalized_end)
    return metadata


__all__: list[str] = [
    "AlignedNumericPIIMatch",
    "NumericPIIPipelineResult",
    "run_numeric_pii_pipeline",
]
