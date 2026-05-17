from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Final

_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+", re.UNICODE)
_SPACE_BEFORE_FINAL_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"\s+([.?!])", re.UNICODE)
_EXTRA_PUNCTUATION_CHARS: Final[frozenset[str]] = frozenset({"№"})
_PROTECTED_NUMERIC_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)\d+(?:[.,:/\-]\d+)+(?!\d)",
    re.UNICODE,
)
_PROTECTED_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:https?|ftp)://[^\s]+",
    re.IGNORECASE | re.UNICODE,
)
_PROTECTED_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.+\-])"
    r"[A-Za-zА-Яа-яЁё0-9._%+\-]+@"
    r"(?:[A-Za-zА-Яа-яЁё0-9\-]+\.)+"
    r"[A-Za-zА-Яа-яЁё]{2,}"
    r"(?![\w.\-])",
    re.IGNORECASE | re.UNICODE,
)
_PROTECTED_DOMAIN_TLDS: Final[tuple[str, ...]] = (
    "com",
    "org",
    "net",
    "ru",
    "рф",
    "ком",
    "su",
    "io",
    "ai",
    "app",
    "dev",
    "gov",
    "edu",
    "info",
    "biz",
    "me",
    "co",
    "uk",
    "de",
    "fr",
    "cn",
    "jp",
    "kz",
    "by",
    "ua",
    "бел",
    "рус",
    "москва",
    "онлайн",
    "сайт",
)
_PROTECTED_DOMAIN_TLD_RE: Final[str] = "|".join(
    re.escape(tld) for tld in sorted(_PROTECTED_DOMAIN_TLDS, key=len, reverse=True)
)
_PROTECTED_DOMAIN_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?<![\w@])"
    rf"(?:[A-Za-zА-Яа-яЁё0-9](?:[A-Za-zА-Яа-яЁё0-9\-]{{0,61}}[A-Za-zА-Яа-яЁё0-9])?\.)+"
    rf"(?:{_PROTECTED_DOMAIN_TLD_RE})"
    rf"(?![A-Za-zА-Яа-яЁё0-9\-])",
    re.IGNORECASE | re.UNICODE,
)
_URL_TRAILING_UNPROTECTED_PUNCTUATION: Final[frozenset[str]] = frozenset(
    {".", ",", ";", ":", "!", "?", "…"}
)


@dataclass(frozen=True, slots=True)
class RemovedPunctuationSpan:
    start: int
    end: int
    raw: str
    reason: str


@dataclass(frozen=True, slots=True)
class ProtectedPunctuationSpan:
    start: int
    end: int
    raw: str
    reason: str


@dataclass(frozen=True, slots=True)
class PunctuationCleanedText:
    original_text: str
    text: str
    removed_spans: tuple[RemovedPunctuationSpan, ...] = field(default_factory=tuple)
    protected_spans: tuple[ProtectedPunctuationSpan, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class PunctuationRemovalConfig:
    enabled: bool = True
    preserve_numeric_separators: bool = True
    preserve_domains: bool = True
    preserve_urls: bool = True
    preserve_emails: bool = True
    preserve_sentence_final_punctuation: bool = False
    replacement: str = " "


class PunctuationRemover:
    def __init__(self, config: PunctuationRemovalConfig | None = None) -> None:
        self.config: PunctuationRemovalConfig = (
            config if config is not None else PunctuationRemovalConfig()
        )

    def clean(self, text: str) -> PunctuationCleanedText:
        if not self.config.enabled:
            disabled_result: PunctuationCleanedText = PunctuationCleanedText(
                original_text=text,
                text=text,
            )
            return disabled_result

        protected_spans: tuple[ProtectedPunctuationSpan, ...] = self._collect_protected_spans(text)
        protected_indexes: frozenset[int] = _span_indexes(protected_spans)
        output_characters: list[str] = []
        removed_spans: list[RemovedPunctuationSpan] = []
        current_span_start: int | None = None
        current_span_characters: list[str] = []

        for index, character in enumerate(text):
            should_remove: bool = self._should_remove_character(text, index, protected_indexes)
            if should_remove:
                if current_span_start is None:
                    current_span_start = index
                    current_span_characters = []
                current_span_characters.append(character)
                self._append_replacement(output_characters)
                continue

            if current_span_start is not None:
                removed_spans.append(
                    RemovedPunctuationSpan(
                        start=current_span_start,
                        end=index,
                        raw="".join(current_span_characters),
                        reason="punctuation",
                    )
                )
                current_span_start = None
                current_span_characters = []

            output_characters.append(character)

        if current_span_start is not None:
            removed_spans.append(
                RemovedPunctuationSpan(
                    start=current_span_start,
                    end=len(text),
                    raw="".join(current_span_characters),
                    reason="punctuation",
                )
            )

        raw_cleaned_text: str = "".join(output_characters)
        cleaned_text: str = normalize_punctuation_cleaned_text(raw_cleaned_text)
        result = PunctuationCleanedText(
            original_text=text,
            text=cleaned_text,
            removed_spans=tuple(removed_spans),
            protected_spans=protected_spans,
        )
        return result

    def _collect_protected_spans(self, text: str) -> tuple[ProtectedPunctuationSpan, ...]:
        spans: list[ProtectedPunctuationSpan] = []
        if self.config.preserve_numeric_separators:
            spans.extend(
                _find_regex_spans(text, _PROTECTED_NUMERIC_SEPARATOR_RE, "numeric_separator")
            )
        if self.config.preserve_urls:
            spans.extend(_find_url_spans(text))
        if self.config.preserve_emails:
            spans.extend(_find_regex_spans(text, _PROTECTED_EMAIL_RE, "email"))
        if self.config.preserve_domains:
            spans.extend(_find_regex_spans(text, _PROTECTED_DOMAIN_RE, "domain"))
        merged_spans: tuple[ProtectedPunctuationSpan, ...] = _merge_protected_spans(spans)
        return merged_spans

    def _should_remove_character(
        self,
        text: str,
        index: int,
        protected_indexes: frozenset[int],
    ) -> bool:
        if index in protected_indexes:
            return False
        character: str = text[index]
        if self.config.preserve_sentence_final_punctuation and character in {".", "?", "!"}:
            return False
        result: bool = is_punctuation_character(character)
        return result

    def _append_replacement(self, output_characters: list[str]) -> None:
        replacement: str = self.config.replacement
        if not replacement:
            return
        if output_characters and output_characters[-1].isspace():
            return
        output_characters.append(replacement)


def _find_regex_spans(
    text: str,
    pattern: re.Pattern[str],
    reason: str,
) -> list[ProtectedPunctuationSpan]:
    spans: list[ProtectedPunctuationSpan] = []
    for match in pattern.finditer(text):
        start: int = match.start()
        end: int = match.end()
        if start >= end:
            continue
        spans.append(
            ProtectedPunctuationSpan(
                start=start,
                end=end,
                raw=text[start:end],
                reason=reason,
            )
        )
    return spans


def _find_url_spans(text: str) -> list[ProtectedPunctuationSpan]:
    spans: list[ProtectedPunctuationSpan] = []
    for match in _PROTECTED_URL_RE.finditer(text):
        start: int = match.start()
        end: int = _trim_url_span_end(text, match.start(), match.end())
        if start >= end:
            continue
        spans.append(
            ProtectedPunctuationSpan(
                start=start,
                end=end,
                raw=text[start:end],
                reason="url",
            )
        )
    return spans


def _trim_url_span_end(text: str, start: int, end: int) -> int:
    trimmed_end: int = end
    while trimmed_end > start and text[trimmed_end - 1] in _URL_TRAILING_UNPROTECTED_PUNCTUATION:
        trimmed_end -= 1
    return trimmed_end


def _merge_protected_spans(
    spans: list[ProtectedPunctuationSpan],
) -> tuple[ProtectedPunctuationSpan, ...]:
    sorted_spans: list[ProtectedPunctuationSpan] = sorted(
        spans, key=lambda span: (span.start, span.end)
    )
    merged: list[ProtectedPunctuationSpan] = []
    for span in sorted_spans:
        if not merged:
            merged.append(span)
            continue
        previous: ProtectedPunctuationSpan = merged[-1]
        if span.start <= previous.end:
            end: int = max(previous.end, span.end)
            reason: str = previous.reason if previous.reason == span.reason else "mixed_protected"
            merged[-1] = ProtectedPunctuationSpan(
                start=previous.start,
                end=end,
                raw=(
                    span.raw
                    if previous.start == span.start and span.end > previous.end
                    else previous.raw
                ),
                reason=reason,
            )
            continue
        merged.append(span)
    normalized: tuple[ProtectedPunctuationSpan, ...] = tuple(
        ProtectedPunctuationSpan(
            start=span.start,
            end=span.end,
            raw=span.raw,
            reason=span.reason,
        )
        for span in merged
    )
    return normalized


def _span_indexes(spans: tuple[ProtectedPunctuationSpan, ...]) -> frozenset[int]:
    indexes: set[int] = set()
    for span in spans:
        indexes.update(range(span.start, span.end))
    result: frozenset[int] = frozenset(indexes)
    return result


def is_punctuation_character(character: str) -> bool:
    category: str = unicodedata.category(character)
    result: bool = category.startswith("P") or character in _EXTRA_PUNCTUATION_CHARS
    return result


def normalize_punctuation_cleaned_text(text: str) -> str:
    normalized: str = _SPACE_RE.sub(" ", text).strip()
    normalized = _SPACE_BEFORE_FINAL_PUNCT_RE.sub(r"\1", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized


def remove_punctuation(
    text: str,
    config: PunctuationRemovalConfig | None = None,
) -> str:
    remover = PunctuationRemover(config=config)
    cleaned: PunctuationCleanedText = remover.clean(text)
    return cleaned.text


PunctuationFilterConfig = PunctuationRemovalConfig


__all__: list[str] = [
    "ProtectedPunctuationSpan",
    "PunctuationFilterConfig",
    "PunctuationCleanedText",
    "PunctuationRemovalConfig",
    "PunctuationRemover",
    "RemovedPunctuationSpan",
    "is_punctuation_character",
    "normalize_punctuation_cleaned_text",
    "remove_punctuation",
]
