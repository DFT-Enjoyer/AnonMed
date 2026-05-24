from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Literal, Mapping, Sequence

from anonmed.anonymization.numeric_pii import NumericPIIMatch, NumericPIIType
from anonmed.anonymization.restoration import (
    RestoredTextResult,
    patch_text,
    restore_safe_original_text,
)
from anonmed.preprocessing.asr.alignment import SourceSpan, TextAlignment

PostProcessingMode = Literal["balanced", "conservative", "production_safe"]
MaskingStrategy = Literal["type", "same_length"]

__all__: tuple[str, ...] = (
    "MaskingStrategy",
    "PIICandidate",
    "PostProcessedEntityGroup",
    "PostProcessedPIIMention",
    "PostProcessingMode",
    "PostProcessingResult",
    "candidate_from_numeric_match",
    "patch_text",
    "resolve_pii_candidates",
    "run_numeric_post_processing",
)

_DIRECT_IDENTIFIER_TYPES: Final[frozenset[NumericPIIType]] = frozenset(
    {
        "PHONE",
        "SNILS",
        "PASSPORT",
        "DATE_BIRTH",
        "OMS",
        "INN",
        "MSE",
        "BIRTH_CERTIFICATE",
        "DRIVER_LICENSE",
    }
)
_SENSITIVITY_RANKS: Final[Mapping[NumericPIIType, int]] = {
    "PHONE": 100,
    "SNILS": 99,
    "PASSPORT": 98,
    "OMS": 97,
    "INN": 96,
    "DRIVER_LICENSE": 94,
    "DATE_BIRTH": 90,
    "MSE": 88,
    "BIRTH_CERTIFICATE": 86,
    "AGE": 30,
}
_SOURCE_PRIORS: Final[Mapping[str, float]] = {
    "validator": 1.08,
    "regex": 1.00,
    "context_rule": 0.96,
    "ml": 0.92,
    "dictionary": 0.86,
}


@dataclass(frozen=True, slots=True)
class PIICandidate:
    entity_type: NumericPIIType
    source: str
    source_score: float
    start: int
    end: int
    value: str
    normalized_value: str
    rule_id: str
    context: str
    metadata: Mapping[str, object]
    validators: Mapping[str, object]
    negative_context_hits: tuple[str, ...]
    sensitivity_rank: int
    source_match: NumericPIIMatch | None = None

    @property
    def span_length(self) -> int:
        return self.end - self.start

    @property
    def is_direct_identifier(self) -> bool:
        return self.entity_type in _DIRECT_IDENTIFIER_TYPES

    @property
    def composite_score(self) -> float:
        return _candidate_composite_score(self)


@dataclass(frozen=True, slots=True)
class PostProcessedPIIMention:
    entity_type: NumericPIIType
    original_start: int
    original_end: int
    normalized_start: int
    normalized_end: int
    original_text: str
    normalized_text: str
    normalized_value: str
    replacement: str
    confidence: float
    rule_id: str
    source: str
    entity_id: str
    mention_id: str
    projection_status: str
    metadata: Mapping[str, object]
    source_match: NumericPIIMatch | None = None

    @property
    def original_span_length(self) -> int:
        return self.original_end - self.original_start

    @property
    def normalized_span_length(self) -> int:
        return self.normalized_end - self.normalized_start

    @property
    def is_direct_identifier(self) -> bool:
        return self.entity_type in _DIRECT_IDENTIFIER_TYPES


@dataclass(frozen=True, slots=True)
class PostProcessedEntityGroup:
    entity_id: str
    entity_type: NumericPIIType
    normalized_value: str
    mention_ids: tuple[str, ...]
    mention_count: int
    first_original_start: int
    sensitivity_rank: int
    is_direct_identifier: bool


@dataclass(frozen=True, slots=True)
class PostProcessingResult:
    mode: PostProcessingMode
    masking_strategy: MaskingStrategy
    candidates: tuple[PIICandidate, ...]
    selected_candidates: tuple[PIICandidate, ...]
    mentions: tuple[PostProcessedPIIMention, ...]
    entity_groups: tuple[PostProcessedEntityGroup, ...]
    masked_normalized_text: str
    masked_original_text: str
    restored_safe_text: str
    restoration_result: RestoredTextResult
    audit: Mapping[str, object]


def candidate_from_numeric_match(
    match: NumericPIIMatch,
    *,
    source: str = "regex",
) -> PIICandidate:
    validators: dict[str, object] = {"format_ok": True}
    checksum_value: object | None = match.metadata.get("checksum_valid")
    if checksum_value is not None:
        validators["checksum"] = checksum_value
    positive_context_value: object | None = match.metadata.get("positive_context_hit")
    if positive_context_value is not None:
        validators["context_ok"] = positive_context_value

    sensitivity_rank: int = _SENSITIVITY_RANKS.get(match.pii_type, 50)
    return PIICandidate(
        entity_type=match.pii_type,
        source=source,
        source_score=match.confidence,
        start=match.start,
        end=match.end,
        value=match.value,
        normalized_value=match.normalized_value,
        rule_id=match.rule_id,
        context=match.context,
        metadata=match.metadata,
        validators=validators,
        negative_context_hits=(),
        sensitivity_rank=sensitivity_rank,
        source_match=match,
    )


def resolve_pii_candidates(
    candidates: Sequence[PIICandidate],
    *,
    mode: PostProcessingMode = "production_safe",
) -> tuple[PIICandidate, ...]:
    sorted_candidates: list[PIICandidate] = sorted(
        candidates,
        key=lambda item: (item.start, item.end, -item.composite_score),
    )
    components: list[list[PIICandidate]] = _overlap_components(sorted_candidates)
    selected_candidates: list[PIICandidate] = []
    for component in components:
        if len(component) == 1:
            selected_candidates.append(component[0])
            continue

        if mode == "balanced":
            selected_candidates.append(_best_candidate(component))
            continue

        has_direct_identifier: bool = any(candidate.is_direct_identifier for candidate in component)
        if mode == "conservative" or has_direct_identifier:
            selected_candidates.append(_merge_candidate_component(component))
        else:
            selected_candidates.append(_best_candidate(component))

    return tuple(sorted(selected_candidates, key=lambda item: (item.start, item.end)))



def run_numeric_post_processing(
    *,
    original_text: str,
    normalized_text: str,
    alignment: TextAlignment,
    normalized_matches: Sequence[NumericPIIMatch],
    replacement_by_type: Mapping[NumericPIIType, str] | None = None,
    mode: PostProcessingMode = "production_safe",
    masking_strategy: MaskingStrategy = "type",
) -> PostProcessingResult:
    candidates: tuple[PIICandidate, ...] = tuple(
        candidate_from_numeric_match(match) for match in normalized_matches
    )
    selected_candidates: tuple[PIICandidate, ...] = resolve_pii_candidates(
        candidates,
        mode=mode,
    )
    projected_mentions: tuple[PostProcessedPIIMention, ...] = _project_candidates(
        original_text=original_text,
        normalized_text=normalized_text,
        alignment=alignment,
        candidates=selected_candidates,
        replacement_by_type=replacement_by_type or {},
        masking_strategy=masking_strategy,
    )
    projection_failures: int = sum(
        1 for mention in projected_mentions if mention.projection_status != "ok"
    )
    successful_mentions: tuple[PostProcessedPIIMention, ...] = tuple(
        mention for mention in projected_mentions if mention.projection_status == "ok"
    )
    resolved_mentions: tuple[PostProcessedPIIMention, ...] = _resolve_projected_overlaps(
        original_text=original_text,
        normalized_text=normalized_text,
        mentions=successful_mentions,
        mode=mode,
        replacement_by_type=replacement_by_type or {},
        masking_strategy=masking_strategy,
    )
    grouped_mentions, entity_groups = _assign_entity_groups(resolved_mentions)
    restoration_result: RestoredTextResult = restore_safe_original_text(
        original_text=original_text,
        normalized_text=normalized_text,
        mentions=grouped_mentions,
    )
    masked_original_text: str = restoration_result.masked_original_text
    masked_normalized_text: str = restoration_result.masked_normalized_text

    restoration_is_safe: bool = restoration_result.is_safe and projection_failures == 0
    audit: dict[str, object] = {
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected_candidates),
        "projected_mention_count": len(projected_mentions),
        "projection_failure_count": projection_failures,
        "final_mention_count": len(grouped_mentions),
        "entity_group_count": len(entity_groups),
        "overlap_resolution_mode": mode,
        "restoration": dict(restoration_result.audit),
        "restoration_is_safe": restoration_is_safe,
    }
    return PostProcessingResult(
        mode=mode,
        masking_strategy=masking_strategy,
        candidates=candidates,
        selected_candidates=selected_candidates,
        mentions=grouped_mentions,
        entity_groups=entity_groups,
        masked_normalized_text=masked_normalized_text,
        masked_original_text=masked_original_text,
        restored_safe_text=restoration_result.safe_text,
        restoration_result=restoration_result,
        audit=audit,
    )


def _candidate_composite_score(candidate: PIICandidate) -> float:
    source_prior: float = _SOURCE_PRIORS.get(candidate.source, 0.90)
    score: float = candidate.source_score * source_prior

    positive_context_value: object | None = candidate.metadata.get("positive_context_hit")
    if positive_context_value is True:
        score += 0.10

    checksum_value: object | None = candidate.validators.get("checksum")
    if checksum_value is True:
        score += 0.10
    elif checksum_value is False and candidate.entity_type in {"SNILS", "INN"}:
        score -= 0.08

    format_value: object | None = candidate.validators.get("format_ok")
    if format_value is True:
        score += 0.04

    score -= 0.12 * len(candidate.negative_context_hits)
    score += candidate.sensitivity_rank / 1000.0
    return min(0.99, max(0.01, score))


def _overlap_components(candidates: Sequence[PIICandidate]) -> list[list[PIICandidate]]:
    components: list[list[PIICandidate]] = []
    current_component: list[PIICandidate] = []
    current_end: int = -1
    for candidate in candidates:
        if not current_component:
            current_component = [candidate]
            current_end = candidate.end
            continue
        if candidate.start < current_end:
            current_component.append(candidate)
            current_end = max(current_end, candidate.end)
            continue
        components.append(current_component)
        current_component = [candidate]
        current_end = candidate.end

    if current_component:
        components.append(current_component)
    return components


def _best_candidate(candidates: Sequence[PIICandidate]) -> PIICandidate:
    return max(
        candidates,
        key=lambda item: (
            item.composite_score,
            item.sensitivity_rank,
            -item.span_length,
            -item.start,
        ),
    )


def _merge_candidate_component(candidates: Sequence[PIICandidate]) -> PIICandidate:
    best_candidate: PIICandidate = _best_candidate(candidates)
    merged_start: int = min(candidate.start for candidate in candidates)
    merged_end: int = max(candidate.end for candidate in candidates)
    metadata: dict[str, object] = dict(best_candidate.metadata)
    metadata["post_processing"] = "union_overlap"
    metadata["merged_rule_ids"] = tuple(candidate.rule_id for candidate in candidates)
    return replace(
        best_candidate,
        start=merged_start,
        end=merged_end,
        metadata=metadata,
    )


def _project_candidates(
    *,
    original_text: str,
    normalized_text: str,
    alignment: TextAlignment,
    candidates: Sequence[PIICandidate],
    replacement_by_type: Mapping[NumericPIIType, str],
    masking_strategy: MaskingStrategy,
) -> tuple[PostProcessedPIIMention, ...]:
    mentions: list[PostProcessedPIIMention] = []
    for candidate in candidates:
        source_span: SourceSpan = alignment.source_span_for_target_span(
            candidate.start,
            candidate.end,
        )
        projection_status: str = "ok" if source_span.start < source_span.end else "failed"
        original_value: str = original_text[source_span.start : source_span.end]
        normalized_value: str = normalized_text[candidate.start : candidate.end]
        replacement: str = _replacement_for_span(
            entity_type=candidate.entity_type,
            span_length=source_span.end - source_span.start,
            replacement_by_type=replacement_by_type,
            masking_strategy=masking_strategy,
        )
        mentions.append(
            PostProcessedPIIMention(
                entity_type=candidate.entity_type,
                original_start=source_span.start,
                original_end=source_span.end,
                normalized_start=candidate.start,
                normalized_end=candidate.end,
                original_text=original_value,
                normalized_text=normalized_value,
                normalized_value=candidate.normalized_value,
                replacement=replacement,
                confidence=candidate.composite_score,
                rule_id=candidate.rule_id,
                source=candidate.source,
                entity_id="",
                mention_id="",
                projection_status=projection_status,
                metadata=candidate.metadata,
                source_match=candidate.source_match,
            )
        )
    return tuple(mentions)


def _resolve_projected_overlaps(
    *,
    original_text: str,
    normalized_text: str,
    mentions: Sequence[PostProcessedPIIMention],
    mode: PostProcessingMode,
    replacement_by_type: Mapping[NumericPIIType, str],
    masking_strategy: MaskingStrategy,
) -> tuple[PostProcessedPIIMention, ...]:
    sorted_mentions: list[PostProcessedPIIMention] = sorted(
        mentions,
        key=lambda item: (item.original_start, item.original_end, -item.confidence),
    )
    components: list[list[PostProcessedPIIMention]] = []
    current_component: list[PostProcessedPIIMention] = []
    current_end: int = -1
    for mention in sorted_mentions:
        if not current_component:
            current_component = [mention]
            current_end = mention.original_end
            continue
        if mention.original_start < current_end:
            current_component.append(mention)
            current_end = max(current_end, mention.original_end)
            continue
        components.append(current_component)
        current_component = [mention]
        current_end = mention.original_end

    if current_component:
        components.append(current_component)

    resolved_mentions: list[PostProcessedPIIMention] = []
    for component in components:
        if len(component) == 1:
            resolved_mentions.append(component[0])
            continue
        if _same_entity_component(component):
            resolved_mentions.extend(component)
            continue
        if mode == "balanced":
            resolved_mentions.append(_best_mention(component))
            continue
        has_direct_identifier: bool = any(mention.is_direct_identifier for mention in component)
        if mode == "conservative" or has_direct_identifier:
            resolved_mentions.append(
                _merge_mention_component(
                    original_text=original_text,
                    normalized_text=normalized_text,
                    mentions=component,
                    replacement_by_type=replacement_by_type,
                    masking_strategy=masking_strategy,
                )
            )
        else:
            resolved_mentions.append(_best_mention(component))

    return tuple(sorted(resolved_mentions, key=lambda item: (item.original_start, item.original_end)))


def _same_entity_component(mentions: Sequence[PostProcessedPIIMention]) -> bool:
    entity_keys: set[tuple[NumericPIIType, str]] = {
        (mention.entity_type, mention.normalized_value) for mention in mentions
    }
    return len(entity_keys) == 1


def _best_mention(mentions: Sequence[PostProcessedPIIMention]) -> PostProcessedPIIMention:
    return max(
        mentions,
        key=lambda item: (
            item.confidence,
            _SENSITIVITY_RANKS.get(item.entity_type, 50),
            -item.original_span_length,
            -item.original_start,
        ),
    )


def _merge_mention_component(
    *,
    original_text: str,
    normalized_text: str,
    mentions: Sequence[PostProcessedPIIMention],
    replacement_by_type: Mapping[NumericPIIType, str],
    masking_strategy: MaskingStrategy,
) -> PostProcessedPIIMention:
    best_mention: PostProcessedPIIMention = _best_mention(mentions)
    original_start: int = min(mention.original_start for mention in mentions)
    original_end: int = max(mention.original_end for mention in mentions)
    normalized_start: int = min(mention.normalized_start for mention in mentions)
    normalized_end: int = max(mention.normalized_end for mention in mentions)
    metadata: dict[str, object] = dict(best_mention.metadata)
    metadata["post_processing"] = "union_projected_overlap"
    metadata["merged_mention_rule_ids"] = tuple(mention.rule_id for mention in mentions)
    replacement: str = _replacement_for_span(
        entity_type=best_mention.entity_type,
        span_length=original_end - original_start,
        replacement_by_type=replacement_by_type,
        masking_strategy=masking_strategy,
    )
    return replace(
        best_mention,
        original_start=original_start,
        original_end=original_end,
        normalized_start=normalized_start,
        normalized_end=normalized_end,
        original_text=original_text[original_start:original_end],
        normalized_text=normalized_text[normalized_start:normalized_end],
        replacement=replacement,
        metadata=metadata,
    )


def _assign_entity_groups(
    mentions: Sequence[PostProcessedPIIMention],
) -> tuple[tuple[PostProcessedPIIMention, ...], tuple[PostProcessedEntityGroup, ...]]:
    sorted_mentions: list[PostProcessedPIIMention] = sorted(
        mentions,
        key=lambda item: (item.original_start, item.original_end),
    )
    entity_index_by_key: dict[tuple[NumericPIIType, str], int] = {}
    mentions_by_key: dict[tuple[NumericPIIType, str], list[PostProcessedPIIMention]] = {}
    grouped_mentions: list[PostProcessedPIIMention] = []
    for mention in sorted_mentions:
        key: tuple[NumericPIIType, str] = (mention.entity_type, mention.normalized_value)
        if key not in entity_index_by_key:
            entity_index_by_key[key] = len(entity_index_by_key) + 1
        entity_index: int = entity_index_by_key[key]
        entity_id: str = f"{mention.entity_type}:{entity_index:04d}"
        mention_number: int = len(mentions_by_key.get(key, [])) + 1
        mention_id: str = f"{entity_id}:m{mention_number:02d}"
        grouped_mention: PostProcessedPIIMention = replace(
            mention,
            entity_id=entity_id,
            mention_id=mention_id,
        )
        grouped_mentions.append(grouped_mention)
        mentions_by_key.setdefault(key, []).append(grouped_mention)

    entity_groups: list[PostProcessedEntityGroup] = []
    for key, key_mentions in mentions_by_key.items():
        first_mention: PostProcessedPIIMention = min(
            key_mentions,
            key=lambda item: (item.original_start, item.original_end),
        )
        entity_groups.append(
            PostProcessedEntityGroup(
                entity_id=first_mention.entity_id,
                entity_type=key[0],
                normalized_value=key[1],
                mention_ids=tuple(mention.mention_id for mention in key_mentions),
                mention_count=len(key_mentions),
                first_original_start=first_mention.original_start,
                sensitivity_rank=_SENSITIVITY_RANKS.get(key[0], 50),
                is_direct_identifier=key[0] in _DIRECT_IDENTIFIER_TYPES,
            )
        )

    return (
        tuple(grouped_mentions),
        tuple(sorted(entity_groups, key=lambda item: item.first_original_start)),
    )


def _replacement_for_span(
    *,
    entity_type: NumericPIIType,
    span_length: int,
    replacement_by_type: Mapping[NumericPIIType, str],
    masking_strategy: MaskingStrategy,
) -> str:
    explicit_replacement: str | None = replacement_by_type.get(entity_type)
    if explicit_replacement is not None:
        return explicit_replacement
    if masking_strategy == "same_length":
        return "*" * max(1, span_length)
    return f"[{entity_type}]"


