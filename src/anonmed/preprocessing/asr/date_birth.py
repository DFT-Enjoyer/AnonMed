from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Final

from anonmed.preprocessing.asr.tokenization import tokenize_preserving_spans
from anonmed.preprocessing.asr.types import Token


_CONTENT_TOKEN_KINDS: Final[frozenset[str]] = frozenset({"word", "digits"})
_BIRTH_CONTEXT_PREFIXES: Final[tuple[str, ...]] = (
    "рожд",
    "родил",
    "родив",
)
_MONTH_VALUES: Final[dict[str, int]] = {
    "январь": 1,
    "января": 1,
    "январе": 1,
    "февраль": 2,
    "февраля": 2,
    "феврале": 2,
    "март": 3,
    "марта": 3,
    "марте": 3,
    "апрель": 4,
    "апреля": 4,
    "апреле": 4,
    "май": 5,
    "мая": 5,
    "мае": 5,
    "июнь": 6,
    "июня": 6,
    "июне": 6,
    "июль": 7,
    "июля": 7,
    "июле": 7,
    "август": 8,
    "августа": 8,
    "августе": 8,
    "сентябрь": 9,
    "сентября": 9,
    "сентябре": 9,
    "октябрь": 10,
    "октября": 10,
    "октябре": 10,
    "ноябрь": 11,
    "ноября": 11,
    "ноябре": 11,
    "декабрь": 12,
    "декабря": 12,
    "декабре": 12,
}


@dataclass(frozen=True, slots=True)
class DateBirthSpan:
    start: int
    end: int
    raw: str
    normalized: str
    day: int
    month: int
    year: int
    reason: str = "spoken_date_birth"


@dataclass(frozen=True, slots=True)
class DateBirthNormalizedText:
    original_text: str
    text: str
    spans: tuple[DateBirthSpan, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DateBirthNormalizerConfig:
    require_birth_context: bool = True
    context_window_tokens: int = 8
    min_year: int = 1900
    max_year: int | None = None


class DateBirthNormalizer:
    def __init__(self, config: DateBirthNormalizerConfig | None = None) -> None:
        self.config: DateBirthNormalizerConfig = (
            config if config is not None else DateBirthNormalizerConfig()
        )

    def normalize(self, text: str) -> DateBirthNormalizedText:
        tokens: list[Token] = tokenize_preserving_spans(text)
        content_tokens: list[Token] = [
            token for token in tokens if token.kind in _CONTENT_TOKEN_KINDS
        ]
        spans: list[DateBirthSpan] = []
        last_span_end: int = -1

        for content_index in range(max(0, len(content_tokens) - 2)):
            day_token: Token = content_tokens[content_index]
            month_token: Token = content_tokens[content_index + 1]
            year_token: Token = content_tokens[content_index + 2]

            if day_token.start < last_span_end:
                continue

            day_value: int | None = _parse_digits_token(day_token)
            month_value: int | None = _month_value(month_token)
            year_value: int | None = self._parse_year_token(year_token)
            if day_value is None or month_value is None or year_value is None:
                continue

            if not self._is_allowed_year(year_value):
                continue

            if not _is_valid_date(day_value, month_value, year_value):
                continue

            if self.config.require_birth_context and not _has_birth_context(
                content_tokens,
                content_index,
                content_index + 3,
                window_tokens=self.config.context_window_tokens,
            ):
                continue

            normalized: str = f"{day_value:02d}.{month_value:02d}.{year_value:04d}"
            span = DateBirthSpan(
                start=day_token.start,
                end=year_token.end,
                raw=text[day_token.start : year_token.end],
                normalized=normalized,
                day=day_value,
                month=month_value,
                year=year_value,
            )
            spans.append(span)
            last_span_end = span.end

        normalized_text: str = _replace_date_birth_spans(text, spans)
        result = DateBirthNormalizedText(
            original_text=text,
            text=normalized_text,
            spans=tuple(spans),
        )
        return result

    def _is_allowed_year(self, year_value: int) -> bool:
        max_year: int = date.today().year if self.config.max_year is None else self.config.max_year
        allowed: bool = self.config.min_year <= year_value <= max_year
        return allowed

    def _parse_year_token(self, token: Token) -> int | None:
        raw_year_value: int | None = _parse_digits_token(token)
        if raw_year_value is None:
            return None

        normalized_year_value: int | None = _normalize_echoed_year(
            token.normalized,
            min_year=self.config.min_year,
            max_year=date.today().year if self.config.max_year is None else self.config.max_year,
        )
        if normalized_year_value is not None:
            return normalized_year_value

        return raw_year_value


def _parse_digits_token(token: Token) -> int | None:
    if token.kind != "digits":
        return None
    try:
        value: int = int(token.normalized)
    except ValueError:
        return None
    return value


def _normalize_echoed_year(
    digits: str,
    *,
    min_year: int,
    max_year: int,
) -> int | None:
    if not digits.isdigit() or len(digits) < 6:
        return None

    first_four_digits: str = digits[:4]
    if not (first_four_digits.startswith("19") or first_four_digits.startswith("20")):
        return None

    echoed_tail: str = digits[4:]
    expected_tail: str = first_four_digits[-len(echoed_tail) :]
    if echoed_tail != expected_tail:
        return None

    year_value: int = int(first_four_digits)
    if min_year <= year_value <= max_year:
        return year_value
    return None


def _month_value(token: Token) -> int | None:
    if token.kind != "word":
        return None
    value: int | None = _MONTH_VALUES.get(token.normalized)
    return value


def _is_valid_date(day_value: int, month_value: int, year_value: int) -> bool:
    try:
        date(year_value, month_value, day_value)
    except ValueError:
        return False
    return True


def _is_birth_context_token(token: Token) -> bool:
    if token.kind != "word":
        return False
    token_text: str = token.normalized
    result: bool = token_text.startswith(_BIRTH_CONTEXT_PREFIXES)
    return result


def _has_birth_context(
    tokens: list[Token],
    start_index: int,
    end_index: int,
    *,
    window_tokens: int,
) -> bool:
    context_start: int = max(0, start_index - window_tokens)
    context_end: int = min(len(tokens), end_index + window_tokens)
    result: bool = any(_is_birth_context_token(token) for token in tokens[context_start:context_end])
    return result


def _replace_date_birth_spans(text: str, spans: list[DateBirthSpan]) -> str:
    pieces: list[str] = []
    cursor: int = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor:
            continue
        pieces.append(text[cursor:span.start])
        pieces.append(span.normalized)
        cursor = span.end
    pieces.append(text[cursor:])
    result: str = "".join(pieces)
    return result


def normalize_spoken_date_birth(
    text: str,
    config: DateBirthNormalizerConfig | None = None,
) -> str:
    normalizer = DateBirthNormalizer(config=config)
    result: DateBirthNormalizedText = normalizer.normalize(text)
    return result.text


__all__: list[str] = [
    "DateBirthNormalizedText",
    "DateBirthNormalizer",
    "DateBirthNormalizerConfig",
    "DateBirthSpan",
    "normalize_spoken_date_birth",
]
