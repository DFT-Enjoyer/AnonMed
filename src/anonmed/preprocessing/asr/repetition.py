from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Final, Iterable, Literal

from anonmed.preprocessing.asr.fuzzy_matching import (
    lexical_similarity,
    normalized_levenshtein_similarity,
)
from anonmed.preprocessing.asr.numeric_lexicon import NUMERIC_WORDS
from anonmed.preprocessing.asr.tokenization import tokenize_preserving_spans
from anonmed.preprocessing.asr.types import Token

RepeatAction = Literal["keep", "suppress", "superseded"]
RepeatReason = Literal[
    "none",
    "chunk_overlap",
    "exact_duplicate",
    "contained_in_previous",
    "superseded_by_more_complete_repeat",
    "superseded_by_repair",
]
SpeakerMatchPolicy = Literal["any", "same", "cross"]

_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+", re.UNICODE)
_NEGATION_WORDS: Final[frozenset[str]] = frozenset({"не", "нет", "ни", "без"})
_REPAIR_CUE_WORDS: Final[frozenset[str]] = frozenset(
    {
        "нет",
        "точнее",
        "вернее",
        "исправлю",
        "ошибка",
        "ошибся",
        "ошиблась",
        "сказал",
        "сказала",
        "повторяю",
        "правильно",
    }
)


@dataclass(frozen=True, slots=True)
class ASRUtterance:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ASRRepeatDeduplicationConfig:
    max_turn_distance: int = 3
    max_time_gap_seconds: float = 15.0
    speaker_policy: SpeakerMatchPolicy = "any"
    allow_unknown_speaker_matches: bool = True
    exact_similarity_threshold: float = 0.94
    exact_coverage_threshold: float = 0.90
    partial_similarity_seed: float = 0.72
    partial_coverage_threshold: float = 0.70
    repair_overlap_threshold: float = 0.50
    token_similarity_threshold: float = 0.84
    min_partial_match_tokens: int = 2
    min_confidence_for_suppression: float = 0.60
    chunk_overlap_tolerance_seconds: float = 0.20
    suppress_previous_when_current_is_more_complete: bool = True
    suppress_previous_on_repair: bool = True
    protect_numeric_delta_without_repair: bool = True
    protect_negation_delta_without_repair: bool = True


@dataclass(frozen=True, slots=True)
class RepeatCandidateScore:
    current_index: int
    previous_index: int
    lexical_similarity: float
    alignment_coverage: float
    current_coverage: float
    previous_coverage: float
    matched_tokens: int
    current_token_count: int
    previous_token_count: int
    turn_distance: int
    time_gap_seconds: float | None = None
    same_speaker: bool | None = None
    current_has_repair_cue: bool = False
    has_numeric_delta: bool = False
    has_negation_delta: bool = False


@dataclass(frozen=True, slots=True)
class DeduplicatedASRUtterance:
    index: int
    utterance: ASRUtterance
    text: str
    action: RepeatAction = "keep"
    reason: RepeatReason = "none"
    duplicate_of: int | None = None
    score: RepeatCandidateScore | None = None

    @property
    def is_visible(self) -> bool:
        visible: bool = self.action == "keep"
        return visible


@dataclass(frozen=True, slots=True)
class ASRRepeatDeduplicationResult:
    raw_utterances: tuple[ASRUtterance, ...]
    utterances: tuple[DeduplicatedASRUtterance, ...] = field(default_factory=tuple)

    @property
    def raw_transcript(self) -> str:
        text: str = _join_transcript(utterance.text for utterance in self.raw_utterances)
        return text

    @property
    def clean_transcript(self) -> str:
        text: str = _join_transcript(
            utterance.text for utterance in self.utterances if utterance.is_visible
        )
        return text

    @property
    def suppressed_indexes(self) -> tuple[int, ...]:
        indexes: tuple[int, ...] = tuple(
            utterance.index for utterance in self.utterances if not utterance.is_visible
        )
        return indexes


@dataclass(frozen=True, slots=True)
class _AlignmentResult:
    matched_tokens: int
    current_token_count: int
    previous_token_count: int

    @property
    def current_coverage(self) -> float:
        if self.current_token_count == 0:
            return 0.0
        coverage: float = float(self.matched_tokens) / float(self.current_token_count)
        return coverage

    @property
    def previous_coverage(self) -> float:
        if self.previous_token_count == 0:
            return 0.0
        coverage: float = float(self.matched_tokens) / float(self.previous_token_count)
        return coverage

    @property
    def alignment_coverage(self) -> float:
        shorter_length: int = min(self.current_token_count, self.previous_token_count)
        if shorter_length == 0:
            return 0.0
        coverage: float = float(self.matched_tokens) / float(shorter_length)
        return coverage


class ASRRepeatDeduplicator:
    def __init__(self, config: ASRRepeatDeduplicationConfig | None = None) -> None:
        self.config: ASRRepeatDeduplicationConfig = (
            config if config is not None else ASRRepeatDeduplicationConfig()
        )

    def deduplicate(self, utterances: Iterable[ASRUtterance | str]) -> ASRRepeatDeduplicationResult:
        raw_utterances: tuple[ASRUtterance, ...] = tuple(
            _coerce_utterance(item) for item in utterances
        )
        decisions: list[DeduplicatedASRUtterance] = []

        for current_index, current_utterance in enumerate(raw_utterances):
            best_score: RepeatCandidateScore | None = self._best_candidate(
                raw_utterances,
                current_index,
            )
            current_decision: DeduplicatedASRUtterance = DeduplicatedASRUtterance(
                index=current_index,
                utterance=current_utterance,
                text=current_utterance.text,
            )

            if best_score is not None:
                current_decision = self._apply_decision(
                    decisions,
                    current_decision,
                    best_score,
                )

            decisions.append(current_decision)

        result = ASRRepeatDeduplicationResult(
            raw_utterances=raw_utterances,
            utterances=tuple(decisions),
        )
        return result

    def _best_candidate(
        self,
        utterances: tuple[ASRUtterance, ...],
        current_index: int,
    ) -> RepeatCandidateScore | None:
        start_index: int = max(0, current_index - self.config.max_turn_distance)
        best_score: RepeatCandidateScore | None = None
        best_priority: float = 0.0
        current_utterance: ASRUtterance = utterances[current_index]

        for previous_index in range(start_index, current_index):
            previous_utterance: ASRUtterance = utterances[previous_index]
            if not self._is_candidate_allowed(current_utterance, previous_utterance):
                continue

            score: RepeatCandidateScore = self._score_pair(
                current_utterance,
                previous_utterance,
                current_index=current_index,
                previous_index=previous_index,
            )
            if not self._passes_candidate_gate(score):
                continue

            priority: float = max(
                score.lexical_similarity,
                score.current_coverage,
                score.previous_coverage,
            )
            if priority > best_priority:
                best_score = score
                best_priority = priority

        return best_score

    def _score_pair(
        self,
        current: ASRUtterance,
        previous: ASRUtterance,
        *,
        current_index: int,
        previous_index: int,
    ) -> RepeatCandidateScore:
        current_tokens: list[str] = _content_token_texts(current.text)
        previous_tokens: list[str] = _content_token_texts(previous.text)
        alignment: _AlignmentResult = _align_tokens(
            current_tokens,
            previous_tokens,
            token_similarity_threshold=self.config.token_similarity_threshold,
        )
        current_canonical_text: str = " ".join(current_tokens)
        previous_canonical_text: str = " ".join(previous_tokens)
        lexical_score: float = lexical_similarity(current_canonical_text, previous_canonical_text)
        time_gap_seconds: float | None = _time_gap_seconds(current, previous)
        same_speaker: bool | None = _same_speaker(current, previous)
        current_numeric_signature: tuple[str, ...] = _numeric_signature(current_tokens)
        previous_numeric_signature: tuple[str, ...] = _numeric_signature(previous_tokens)
        has_numeric_delta: bool = _has_numeric_conflict(
            current_numeric_signature,
            previous_numeric_signature,
        )
        current_negations: frozenset[str] = _token_set(current_tokens, _NEGATION_WORDS)
        previous_negations: frozenset[str] = _token_set(previous_tokens, _NEGATION_WORDS)
        has_negation_delta: bool = current_negations != previous_negations
        current_has_repair_cue: bool = bool(_token_set(current_tokens, _REPAIR_CUE_WORDS))

        score = RepeatCandidateScore(
            current_index=current_index,
            previous_index=previous_index,
            lexical_similarity=lexical_score,
            alignment_coverage=alignment.alignment_coverage,
            current_coverage=alignment.current_coverage,
            previous_coverage=alignment.previous_coverage,
            matched_tokens=alignment.matched_tokens,
            current_token_count=alignment.current_token_count,
            previous_token_count=alignment.previous_token_count,
            turn_distance=current_index - previous_index,
            time_gap_seconds=time_gap_seconds,
            same_speaker=same_speaker,
            current_has_repair_cue=current_has_repair_cue,
            has_numeric_delta=has_numeric_delta,
            has_negation_delta=has_negation_delta,
        )
        return score

    def _passes_candidate_gate(self, score: RepeatCandidateScore) -> bool:
        is_exact_like: bool = (
            score.lexical_similarity >= self.config.exact_similarity_threshold
            and score.alignment_coverage >= self.config.exact_coverage_threshold
        )
        has_partial_seed: bool = (
            score.lexical_similarity >= self.config.partial_similarity_seed
            or score.current_coverage >= self.config.partial_coverage_threshold
            or score.previous_coverage >= self.config.partial_coverage_threshold
        )
        is_partial_like: bool = (
            score.alignment_coverage >= self.config.partial_coverage_threshold
            and has_partial_seed
            and score.matched_tokens >= self.config.min_partial_match_tokens
        )
        is_repair_like: bool = (
            score.current_has_repair_cue
            and score.matched_tokens >= self.config.min_partial_match_tokens
            and score.previous_coverage >= self.config.repair_overlap_threshold
        )
        passes: bool = is_exact_like or is_partial_like or is_repair_like
        return passes

    def _apply_decision(
        self,
        decisions: list[DeduplicatedASRUtterance],
        current_decision: DeduplicatedASRUtterance,
        score: RepeatCandidateScore,
    ) -> DeduplicatedASRUtterance:
        if not self._confidence_allows_suppression(current_decision.utterance):
            return current_decision

        if self._has_protected_delta(score):
            return current_decision

        if self._is_chunk_overlap(score):
            decision: DeduplicatedASRUtterance = replace(
                current_decision,
                action="suppress",
                reason="chunk_overlap",
                duplicate_of=score.previous_index,
                score=score,
            )
            return decision

        if self._is_exact_duplicate(score):
            decision = replace(
                current_decision,
                action="suppress",
                reason="exact_duplicate",
                duplicate_of=score.previous_index,
                score=score,
            )
            return decision

        if self._current_repairs_previous(score):
            self._supersede_previous(
                decisions,
                score,
                reason="superseded_by_repair",
            )
            return current_decision

        if self._current_is_more_complete(score):
            self._supersede_previous(
                decisions,
                score,
                reason="superseded_by_more_complete_repeat",
            )
            return current_decision

        if self._current_is_contained_in_previous(score):
            decision = replace(
                current_decision,
                action="suppress",
                reason="contained_in_previous",
                duplicate_of=score.previous_index,
                score=score,
            )
            return decision

        return current_decision

    def _is_candidate_allowed(self, current: ASRUtterance, previous: ASRUtterance) -> bool:
        time_gap_seconds: float | None = _time_gap_seconds(current, previous)
        if (
            time_gap_seconds is not None
            and time_gap_seconds > self.config.max_time_gap_seconds
        ):
            return False

        same_speaker: bool | None = _same_speaker(current, previous)
        if same_speaker is None:
            return self.config.allow_unknown_speaker_matches
        if self.config.speaker_policy == "same":
            return same_speaker
        if self.config.speaker_policy == "cross":
            return not same_speaker
        return True

    def _confidence_allows_suppression(self, utterance: ASRUtterance) -> bool:
        if utterance.confidence is None:
            return True
        allowed: bool = utterance.confidence >= self.config.min_confidence_for_suppression
        return allowed

    def _has_protected_delta(self, score: RepeatCandidateScore) -> bool:
        if score.current_has_repair_cue:
            return False
        numeric_delta_is_protected: bool = (
            self.config.protect_numeric_delta_without_repair and score.has_numeric_delta
        )
        negation_delta_is_protected: bool = (
            self.config.protect_negation_delta_without_repair and score.has_negation_delta
        )
        protected: bool = numeric_delta_is_protected or negation_delta_is_protected
        return protected

    def _is_chunk_overlap(self, score: RepeatCandidateScore) -> bool:
        if score.time_gap_seconds is None:
            return False
        overlaps: bool = score.time_gap_seconds <= self.config.chunk_overlap_tolerance_seconds
        is_duplicate: bool = (
            score.lexical_similarity >= self.config.exact_similarity_threshold
            and score.alignment_coverage >= self.config.exact_coverage_threshold
        )
        return overlaps and is_duplicate

    def _is_exact_duplicate(self, score: RepeatCandidateScore) -> bool:
        is_duplicate: bool = (
            score.lexical_similarity >= self.config.exact_similarity_threshold
            and score.current_coverage >= self.config.exact_coverage_threshold
            and score.previous_coverage >= self.config.exact_coverage_threshold
        )
        return is_duplicate

    def _current_repairs_previous(self, score: RepeatCandidateScore) -> bool:
        repairs_previous: bool = (
            self.config.suppress_previous_on_repair
            and score.current_has_repair_cue
            and score.previous_coverage >= self.config.repair_overlap_threshold
            and score.matched_tokens >= self.config.min_partial_match_tokens
        )
        return repairs_previous

    def _current_is_more_complete(self, score: RepeatCandidateScore) -> bool:
        more_complete: bool = (
            self.config.suppress_previous_when_current_is_more_complete
            and score.previous_coverage >= self.config.partial_coverage_threshold
            and score.current_token_count > score.previous_token_count
            and score.matched_tokens >= self.config.min_partial_match_tokens
        )
        return more_complete

    def _current_is_contained_in_previous(self, score: RepeatCandidateScore) -> bool:
        contained: bool = (
            score.current_coverage >= self.config.partial_coverage_threshold
            and score.previous_token_count >= score.current_token_count
            and score.matched_tokens >= self.config.min_partial_match_tokens
        )
        return contained

    def _supersede_previous(
        self,
        decisions: list[DeduplicatedASRUtterance],
        score: RepeatCandidateScore,
        *,
        reason: RepeatReason,
    ) -> None:
        if score.previous_index >= len(decisions):
            return

        previous_decision: DeduplicatedASRUtterance = decisions[score.previous_index]
        if not previous_decision.is_visible:
            return

        decisions[score.previous_index] = replace(
            previous_decision,
            action="superseded",
            reason=reason,
            duplicate_of=score.current_index,
            score=score,
        )


def deduplicate_asr_utterances(
    utterances: Iterable[ASRUtterance | str],
    *,
    config: ASRRepeatDeduplicationConfig | None = None,
) -> ASRRepeatDeduplicationResult:
    deduplicator = ASRRepeatDeduplicator(config=config)
    result: ASRRepeatDeduplicationResult = deduplicator.deduplicate(utterances)
    return result


def _coerce_utterance(item: ASRUtterance | str) -> ASRUtterance:
    if isinstance(item, ASRUtterance):
        return item
    utterance = ASRUtterance(text=item)
    return utterance


def _content_token_texts(text: str) -> list[str]:
    tokens: list[Token] = tokenize_preserving_spans(text)
    content_tokens: list[str] = [
        token.normalized for token in tokens if token.kind in {"word", "digits"}
    ]
    return content_tokens


def _align_tokens(
    current_tokens: list[str],
    previous_tokens: list[str],
    *,
    token_similarity_threshold: float,
) -> _AlignmentResult:
    current_length: int = len(current_tokens)
    previous_length: int = len(previous_tokens)
    matrix: list[list[int]] = [
        [0] * (previous_length + 1) for _ in range(current_length + 1)
    ]

    for current_index, current_token in enumerate(current_tokens, start=1):
        for previous_index, previous_token in enumerate(previous_tokens, start=1):
            if _tokens_match(
                current_token,
                previous_token,
                token_similarity_threshold=token_similarity_threshold,
            ):
                diagonal_score: int = matrix[current_index - 1][previous_index - 1] + 1
            else:
                diagonal_score = matrix[current_index - 1][previous_index - 1]
            deletion_score: int = matrix[current_index - 1][previous_index]
            insertion_score: int = matrix[current_index][previous_index - 1]
            matrix[current_index][previous_index] = max(
                diagonal_score,
                deletion_score,
                insertion_score,
            )

    alignment = _AlignmentResult(
        matched_tokens=matrix[current_length][previous_length],
        current_token_count=current_length,
        previous_token_count=previous_length,
    )
    return alignment


def _tokens_match(
    current_token: str,
    previous_token: str,
    *,
    token_similarity_threshold: float,
) -> bool:
    if current_token == previous_token:
        return True
    similarity: float = normalized_levenshtein_similarity(current_token, previous_token)
    matches: bool = similarity >= token_similarity_threshold
    return matches


def _numeric_signature(tokens: list[str]) -> tuple[str, ...]:
    numeric_tokens: tuple[str, ...] = tuple(
        token for token in tokens if token.isdigit() or token in NUMERIC_WORDS
    )
    return numeric_tokens


def _has_numeric_conflict(current: tuple[str, ...], previous: tuple[str, ...]) -> bool:
    if not current or not previous:
        return False
    if current == previous:
        return False
    if _is_subsequence(previous, current) or _is_subsequence(current, previous):
        return False
    return True


def _is_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    if not needle:
        return True

    needle_index: int = 0
    for token in haystack:
        if token == needle[needle_index]:
            needle_index += 1
        if needle_index == len(needle):
            return True
    return False


def _token_set(tokens: list[str], vocabulary: frozenset[str]) -> frozenset[str]:
    matched_tokens: frozenset[str] = frozenset(token for token in tokens if token in vocabulary)
    return matched_tokens


def _same_speaker(current: ASRUtterance, previous: ASRUtterance) -> bool | None:
    if current.speaker is None or previous.speaker is None:
        return None
    same: bool = current.speaker == previous.speaker
    return same


def _time_gap_seconds(current: ASRUtterance, previous: ASRUtterance) -> float | None:
    if current.start is None or previous.end is None:
        return None
    gap: float = current.start - previous.end
    return gap


def _join_transcript(texts: Iterable[str]) -> str:
    joined: str = " ".join(text.strip() for text in texts if text.strip())
    normalized: str = _SPACE_RE.sub(" ", joined).strip()
    return normalized


__all__: list[str] = [
    "ASRRepeatDeduplicationConfig",
    "ASRRepeatDeduplicationResult",
    "ASRRepeatDeduplicator",
    "ASRUtterance",
    "DeduplicatedASRUtterance",
    "RepeatAction",
    "RepeatCandidateScore",
    "RepeatReason",
    "SpeakerMatchPolicy",
    "deduplicate_asr_utterances",
]
