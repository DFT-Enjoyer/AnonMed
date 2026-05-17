from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Final, Iterable

from anonmed.preprocessing.asr.tokenization import tokenize_preserving_spans
from anonmed.preprocessing.asr.types import Token

_WORD_TRANSLATION: Final[dict[int, str]] = str.maketrans({"ё": "е", "Ё": "е"})
_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+", re.UNICODE)
_SPACE_BEFORE_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"\s+([,.;:?!])", re.UNICODE)
_REPEATED_SOFT_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"(?:[,;:]\s*){2,}", re.UNICODE)
_PUNCT_BEFORE_FINAL_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[,;:]\s*([.?!])", re.UNICODE)
_SOFT_PUNCT_AFTER_FINAL_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"([.?!])\s*[,;:]+", re.UNICODE)
_LEADING_STRANDED_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"^[\s,;:.?!]+", re.UNICODE)
_TRAILING_STRANDED_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[\s,;:]+$", re.UNICODE)

_HESITATION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "э",
        "ээ",
        "эээ",
        "ээээ",
        "эм",
        "эмм",
        "эммм",
        "мм",
        "ммм",
        "м",
        "хм",
    }
)

_INTERJECTION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "ах",
        "ой",
        "ох",
        "ух",
        "эх",
    }
)

_DISCOURSE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "а",
        "вот",
        "значит",
        "ну",
        "типа",
    }
)

_PHRASE_FILLERS: Final[tuple[tuple[str, ...], ...]] = (
    ("как", "бы"),
    ("это", "самое"),
    ("так", "сказать"),
    ("в", "общем"),
)

_HYPHEN_TOKENS: Final[frozenset[str]] = frozenset({"-", "–", "—", "−"})
_SOFT_BOUNDARY_PUNCT: Final[frozenset[str]] = frozenset({",", ";", ":", "-", "–", "—", "−"})
_HARD_BOUNDARY_PUNCT: Final[frozenset[str]] = frozenset({".", "?", "!"})
_BOUNDARY_PUNCT: Final[frozenset[str]] = _SOFT_BOUNDARY_PUNCT | _HARD_BOUNDARY_PUNCT


@dataclass(frozen=True, slots=True)
class RemovedSpan:
    start: int
    end: int
    raw: str
    normalized: str
    reason: str


@dataclass(frozen=True, slots=True)
class CleanedText:
    original_text: str
    text: str
    removed_spans: tuple[RemovedSpan, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DisfluencyFilterConfig:
    remove_hesitations: bool = True
    remove_interjections: bool = True
    remove_phrase_fillers: bool = True
    remove_discourse_markers: bool = True
    aggressive_discourse_markers: bool = False


class DisfluencyFilter:
    def __init__(self, config: DisfluencyFilterConfig | None = None) -> None:
        self.config: DisfluencyFilterConfig = (
            config if config is not None else DisfluencyFilterConfig()
        )

    def clean(self, text: str) -> CleanedText:
        tokens: list[Token] = tokenize_preserving_spans(text)
        removed_token_indexes: set[int] = set()
        removed_spans: list[RemovedSpan] = []

        if self.config.remove_phrase_fillers:
            self._collect_phrase_fillers(text, tokens, removed_token_indexes, removed_spans)

        if self.config.remove_hesitations:
            self._collect_hyphenated_hesitations(text, tokens, removed_token_indexes, removed_spans)
            self._collect_single_word_class(
                text,
                tokens,
                removed_token_indexes,
                removed_spans,
                vocabulary=_HESITATION_WORDS,
                reason="hesitation",
                context_required=False,
            )

        if self.config.remove_interjections:
            self._collect_single_word_class(
                text,
                tokens,
                removed_token_indexes,
                removed_spans,
                vocabulary=_INTERJECTION_WORDS,
                reason="interjection",
                context_required=True,
            )

        if self.config.remove_discourse_markers:
            self._collect_discourse_markers(text, tokens, removed_token_indexes, removed_spans)

        normalized_spans: list[RemovedSpan] = _merge_removed_spans(removed_spans)
        cleaned_text: str = _remove_spans_from_text(text, normalized_spans)
        result = CleanedText(
            original_text=text,
            text=cleaned_text,
            removed_spans=tuple(normalized_spans),
        )
        return result

    def _collect_phrase_fillers(
        self,
        text: str,
        tokens: list[Token],
        removed_token_indexes: set[int],
        removed_spans: list[RemovedSpan],
    ) -> None:
        for phrase in sorted(_PHRASE_FILLERS, key=len, reverse=True):
            phrase_length: int = len(phrase)
            if phrase_length == 0:
                continue
            last_start_index: int = len(tokens) - phrase_length
            for start_index in range(last_start_index + 1):
                token_indexes: list[int] = list(range(start_index, start_index + phrase_length))
                if any(index in removed_token_indexes for index in token_indexes):
                    continue
                token_words: list[str] = [
                    _canonical_token_text(tokens[index]) for index in token_indexes
                ]
                token_kinds_are_words: bool = all(
                    tokens[index].kind == "word" for index in token_indexes
                )
                if not token_kinds_are_words or tuple(token_words) != phrase:
                    continue
                self._add_token_span(
                    text,
                    tokens,
                    token_indexes,
                    removed_token_indexes,
                    removed_spans,
                    normalized=" ".join(phrase),
                    reason="phrase_filler",
                )

    def _collect_hyphenated_hesitations(
        self,
        text: str,
        tokens: list[Token],
        removed_token_indexes: set[int],
        removed_spans: list[RemovedSpan],
    ) -> None:
        for index in range(max(0, len(tokens) - 2)):
            token_indexes: list[int] = [index, index + 1, index + 2]
            if any(token_index in removed_token_indexes for token_index in token_indexes):
                continue
            left: Token = tokens[index]
            middle: Token = tokens[index + 1]
            right: Token = tokens[index + 2]
            left_text: str = _canonical_token_text(left)
            right_text: str = _canonical_token_text(right)
            is_hyphenated_hesitation: bool = (
                left.kind == "word"
                and middle.text in _HYPHEN_TOKENS
                and right.kind == "word"
                and left_text in {"э", "м"}
                and right_text in {"э", "м"}
            )
            if not is_hyphenated_hesitation:
                continue
            self._add_token_span(
                text,
                tokens,
                token_indexes,
                removed_token_indexes,
                removed_spans,
                normalized=f"{left_text}-{right_text}",
                reason="hyphenated_hesitation",
            )

    def _collect_single_word_class(
        self,
        text: str,
        tokens: list[Token],
        removed_token_indexes: set[int],
        removed_spans: list[RemovedSpan],
        *,
        vocabulary: frozenset[str],
        reason: str,
        context_required: bool,
    ) -> None:
        for index, token in enumerate(tokens):
            if index in removed_token_indexes or token.kind != "word":
                continue
            normalized: str = _canonical_token_text(token)
            if normalized not in vocabulary:
                continue
            if context_required and not _is_boundary_context(tokens, index, removed_token_indexes):
                continue
            self._add_token_span(
                text,
                tokens,
                [index],
                removed_token_indexes,
                removed_spans,
                normalized=normalized,
                reason=reason,
            )

    def _collect_discourse_markers(
        self,
        text: str,
        tokens: list[Token],
        removed_token_indexes: set[int],
        removed_spans: list[RemovedSpan],
    ) -> None:
        for index, token in enumerate(tokens):
            if index in removed_token_indexes or token.kind != "word":
                continue
            normalized: str = _canonical_token_text(token)
            if normalized not in _DISCOURSE_MARKERS:
                continue
            can_remove: bool = (
                self.config.aggressive_discourse_markers
                or _is_discourse_marker_context(tokens, index, removed_token_indexes)
            )
            if not can_remove:
                continue
            self._add_token_span(
                text,
                tokens,
                [index],
                removed_token_indexes,
                removed_spans,
                normalized=normalized,
                reason="discourse_marker",
            )

    def _add_token_span(
        self,
        text: str,
        tokens: list[Token],
        token_indexes: Iterable[int],
        removed_token_indexes: set[int],
        removed_spans: list[RemovedSpan],
        *,
        normalized: str,
        reason: str,
    ) -> None:
        indexes: list[int] = sorted(token_indexes)
        if not indexes:
            return
        if any(index in removed_token_indexes for index in indexes):
            return
        start: int = tokens[indexes[0]].start
        end: int = tokens[indexes[-1]].end
        raw: str = text[start:end]
        removed_spans.append(
            RemovedSpan(
                start=start,
                end=end,
                raw=raw,
                normalized=normalized,
                reason=reason,
            )
        )
        removed_token_indexes.update(indexes)


def _canonical_token_text(token: Token) -> str:
    result: str = token.text.lower().translate(_WORD_TRANSLATION)
    return result


def _previous_available_index(
    tokens: list[Token],
    index: int,
    removed_token_indexes: set[int],
) -> int | None:
    previous_index: int = index - 1
    while previous_index >= 0:
        if previous_index not in removed_token_indexes:
            return previous_index
        previous_index -= 1
    return None


def _next_available_index(
    tokens: list[Token],
    index: int,
    removed_token_indexes: set[int],
) -> int | None:
    next_index: int = index + 1
    while next_index < len(tokens):
        if next_index not in removed_token_indexes:
            return next_index
        next_index += 1
    return None


def _is_boundary_context(tokens: list[Token], index: int, removed_token_indexes: set[int]) -> bool:
    previous_index: int | None = _previous_available_index(tokens, index, removed_token_indexes)
    next_index: int | None = _next_available_index(tokens, index, removed_token_indexes)
    left_boundary: bool = previous_index is None or tokens[previous_index].text in _BOUNDARY_PUNCT
    right_boundary: bool = next_index is None or tokens[next_index].text in _BOUNDARY_PUNCT
    result: bool = left_boundary or right_boundary
    return result


def _is_discourse_marker_context(
    tokens: list[Token],
    index: int,
    removed_token_indexes: set[int],
) -> bool:
    previous_index: int | None = _previous_available_index(tokens, index, removed_token_indexes)
    next_index: int | None = _next_available_index(tokens, index, removed_token_indexes)
    left_boundary: bool = previous_index is None or tokens[previous_index].text in _BOUNDARY_PUNCT
    right_boundary: bool = next_index is None or tokens[next_index].text in _SOFT_BOUNDARY_PUNCT
    result: bool = left_boundary or right_boundary
    return result


def _merge_removed_spans(spans: list[RemovedSpan]) -> list[RemovedSpan]:
    sorted_spans: list[RemovedSpan] = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[RemovedSpan] = []
    for span in sorted_spans:
        if not merged:
            merged.append(span)
            continue
        previous: RemovedSpan = merged[-1]
        if span.start < previous.end:
            merged[-1] = RemovedSpan(
                start=previous.start,
                end=max(previous.end, span.end),
                raw=previous.raw,
                normalized=previous.normalized,
                reason=previous.reason,
            )
            continue
        merged.append(span)
    return merged


def _remove_spans_from_text(text: str, spans: list[RemovedSpan]) -> str:
    pieces: list[str] = []
    cursor: int = 0
    for span in spans:
        if span.start < cursor:
            continue
        pieces.append(text[cursor : span.start])
        cursor = span.end
    pieces.append(text[cursor:])
    raw_result: str = "".join(pieces)
    result: str = normalize_cleaned_text(raw_result)
    return result


def normalize_cleaned_text(text: str) -> str:
    normalized: str = _SPACE_RE.sub(" ", text).strip()
    normalized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
    normalized = _REPEATED_SOFT_PUNCT_RE.sub(" ", normalized)
    normalized = _PUNCT_BEFORE_FINAL_PUNCT_RE.sub(r"\1", normalized)
    normalized = _SOFT_PUNCT_AFTER_FINAL_PUNCT_RE.sub(r"\1", normalized)
    normalized = _LEADING_STRANDED_PUNCT_RE.sub("", normalized)
    normalized = _TRAILING_STRANDED_PUNCT_RE.sub("", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def remove_disfluencies(
    text: str,
    config: DisfluencyFilterConfig | None = None,
) -> str:
    disfluency_filter = DisfluencyFilter(config=config)
    cleaned: CleanedText = disfluency_filter.clean(text)
    return cleaned.text


__all__: list[str] = [
    "CleanedText",
    "DisfluencyFilter",
    "DisfluencyFilterConfig",
    "RemovedSpan",
    "normalize_cleaned_text",
    "remove_disfluencies",
]
