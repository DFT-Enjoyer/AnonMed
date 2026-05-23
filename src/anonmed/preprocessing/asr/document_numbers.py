from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Final, Literal, Pattern


DocumentNumberKind = Literal[
    "PHONE",
    "SNILS",
    "PASSPORT",
    "OMS",
    "INN",
    "BIRTH_CERTIFICATE",
    "DRIVER_LICENSE",
    "MSE",
]

DocumentNumberReason = Literal[
    "split_repeated_document_number",
    "echo_tail_document_number",
]

_DIGIT_RUN_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\d)\d{8,}(?!\d)", re.UNICODE)
_REGEX_FLAGS: Final[int] = re.IGNORECASE | re.UNICODE


@dataclass(frozen=True, slots=True)
class DocumentNumberRule:
    pii_type: DocumentNumberKind
    expected_lengths: tuple[int, ...]
    context_patterns: tuple[Pattern[str], ...]
    context_window: int = 96
    priority: int = 50


@dataclass(frozen=True, slots=True)
class DocumentNumberSpan:
    start: int
    end: int
    raw: str
    normalized: str
    pii_type: DocumentNumberKind
    reason: DocumentNumberReason


@dataclass(frozen=True, slots=True)
class DocumentNumberNormalizedText:
    original_text: str
    text: str
    spans: tuple[DocumentNumberSpan, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DocumentNumberNormalizerConfig:
    rules: tuple[DocumentNumberRule, ...] | None = None
    max_echo_tail_digits: int = 4
    trim_echo_tail: bool = False


class DocumentNumberNormalizer:
    def __init__(self, config: DocumentNumberNormalizerConfig | None = None) -> None:
        self.config: DocumentNumberNormalizerConfig = (
            config if config is not None else DocumentNumberNormalizerConfig()
        )
        self.rules: tuple[DocumentNumberRule, ...] = (
            build_default_document_number_rules()
            if self.config.rules is None
            else self.config.rules
        )

    def normalize(self, text: str) -> DocumentNumberNormalizedText:
        spans: list[DocumentNumberSpan] = []
        for regex_match in _DIGIT_RUN_RE.finditer(text):
            raw_digits: str = regex_match.group(0)
            rule: DocumentNumberRule | None = self._matching_rule(
                text,
                regex_match.start(),
                regex_match.end(),
            )
            if rule is None:
                continue

            replacement_text: str | None
            reason: DocumentNumberReason
            replacement_text, reason = _normalize_digit_run(
                raw_digits,
                rule,
                max_echo_tail_digits=self.config.max_echo_tail_digits,
                trim_echo_tail=self.config.trim_echo_tail,
            )
            if replacement_text is None:
                continue

            span = DocumentNumberSpan(
                start=regex_match.start(),
                end=regex_match.end(),
                raw=raw_digits,
                normalized=replacement_text,
                pii_type=rule.pii_type,
                reason=reason,
            )
            spans.append(span)

        normalized_text: str = _replace_document_number_spans(text, spans)
        result = DocumentNumberNormalizedText(
            original_text=text,
            text=normalized_text,
            spans=tuple(spans),
        )
        return result

    def _matching_rule(
        self,
        text: str,
        start: int,
        end: int,
    ) -> DocumentNumberRule | None:
        matching_rules: list[DocumentNumberRule] = []
        for rule in self.rules:
            scope_start: int = max(0, start - rule.context_window)
            scope_end: int = min(len(text), end + rule.context_window)
            scope: str = text[scope_start:scope_end]
            has_context: bool = any(pattern.search(scope) is not None for pattern in rule.context_patterns)
            if has_context:
                matching_rules.append(rule)

        if not matching_rules:
            return None

        selected_rule: DocumentNumberRule = max(matching_rules, key=lambda item: item.priority)
        return selected_rule


def _compile(pattern: str) -> Pattern[str]:
    compiled_pattern: Pattern[str] = re.compile(pattern, _REGEX_FLAGS)
    return compiled_pattern


def _compile_many(patterns: tuple[str, ...]) -> tuple[Pattern[str], ...]:
    compiled_patterns: tuple[Pattern[str], ...] = tuple(_compile(pattern) for pattern in patterns)
    return compiled_patterns


def build_default_document_number_rules() -> tuple[DocumentNumberRule, ...]:
    rules: tuple[DocumentNumberRule, ...] = (
        DocumentNumberRule(
            pii_type="PASSPORT",
            expected_lengths=(10, 6, 4),
            context_patterns=_compile_many(
                (
                    r"\bпаспорт\w*\b",
                    r"\bпаспорт\w*.*\b(?:серия|номер)\w*\b",
                )
            ),
            priority=100,
        ),
        DocumentNumberRule(
            pii_type="DRIVER_LICENSE",
            expected_lengths=(10,),
            context_patterns=_compile_many(
                (
                    r"\bводительск\w*\s+(?:удостоверени\w*|прав\w*)\b",
                    r"\b(?:права|удостоверени\w*)\s+номер\w*\b",
                )
            ),
            priority=98,
        ),
        DocumentNumberRule(
            pii_type="OMS",
            expected_lengths=(16, 10),
            context_patterns=_compile_many(
                (
                    r"\bомс\b",
                    r"\bполис\w*\b",
                    r"\bмедицинск\w*\s+страхован\w*\b",
                )
            ),
            priority=96,
        ),
        DocumentNumberRule(
            pii_type="SNILS",
            expected_lengths=(11,),
            context_patterns=_compile_many((r"\bснилс\b", r"\bстрахов\w*\s+номер\w*\b")),
            priority=94,
        ),
        DocumentNumberRule(
            pii_type="INN",
            expected_lengths=(12,),
            context_patterns=_compile_many(
                (r"\bинн\b", r"\bидентификацион\w*\s+номер\w*\b")
            ),
            priority=94,
        ),
        DocumentNumberRule(
            pii_type="PHONE",
            expected_lengths=(11, 10),
            context_patterns=_compile_many(
                (
                    r"\bтелефон\w*\b",
                    r"\b(?:мобильн\w*|сотов\w*|домашн\w*)\b",
                    r"\bдля\s+связи\b",
                    r"\b(?:позвон\w*|перезвон\w*)\b",
                )
            ),
            priority=70,
        ),
        DocumentNumberRule(
            pii_type="BIRTH_CERTIFICATE",
            expected_lengths=(12, 11, 10, 9, 8, 7, 6, 5, 4),
            context_patterns=_compile_many(
                (
                    r"\bсвидетельств\w*\s+о\s+рождени\w*\b",
                    r"\bномер\w*\s+свидетельств\w*\b",
                )
            ),
            priority=66,
        ),
        DocumentNumberRule(
            pii_type="MSE",
            expected_lengths=(12, 11, 10, 9, 8, 7, 6, 5, 4),
            context_patterns=_compile_many(
                (
                    r"\bмсэ\b",
                    r"\b(?:справк\w*|акт\w*)\s+мсэ\b",
                    r"\bинвалидност\w*\b",
                )
            ),
            priority=64,
        ),
    )
    return rules


def _normalize_digit_run(
    digits: str,
    rule: DocumentNumberRule,
    *,
    max_echo_tail_digits: int,
    trim_echo_tail: bool,
) -> tuple[str | None, DocumentNumberReason]:
    repeated_digits: str | None = _split_exact_repetition(digits, rule.expected_lengths)
    if repeated_digits is not None:
        return repeated_digits, "split_repeated_document_number"

    if trim_echo_tail:
        trimmed_digits: str | None = _trim_echo_tail(
            digits,
            rule.expected_lengths,
            max_echo_tail_digits=max_echo_tail_digits,
        )
        if trimmed_digits is not None:
            return trimmed_digits, "echo_tail_document_number"

    return None, "split_repeated_document_number"


def _split_exact_repetition(digits: str, expected_lengths: tuple[int, ...]) -> str | None:
    for expected_length in sorted(expected_lengths, reverse=True):
        if expected_length <= 0 or len(digits) <= expected_length:
            continue
        if len(digits) % expected_length != 0:
            continue
        chunk: str = digits[:expected_length]
        repetitions: int = len(digits) // expected_length
        if repetitions < 2:
            continue
        if chunk * repetitions == digits:
            chunks: list[str] = [chunk] * repetitions
            return ", ".join(chunks)
    return None


def _trim_echo_tail(
    digits: str,
    expected_lengths: tuple[int, ...],
    *,
    max_echo_tail_digits: int,
) -> str | None:
    for expected_length in sorted(expected_lengths, reverse=True):
        if len(digits) <= expected_length:
            continue
        tail: str = digits[expected_length:]
        if len(tail) > max_echo_tail_digits:
            continue
        candidate: str = digits[:expected_length]
        if candidate.endswith(tail):
            return candidate
    return None


def _replace_document_number_spans(text: str, spans: list[DocumentNumberSpan]) -> str:
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


def normalize_document_number_runs(
    text: str,
    config: DocumentNumberNormalizerConfig | None = None,
) -> str:
    normalizer = DocumentNumberNormalizer(config=config)
    result: DocumentNumberNormalizedText = normalizer.normalize(text)
    return result.text


__all__: list[str] = [
    "DocumentNumberKind",
    "DocumentNumberNormalizedText",
    "DocumentNumberNormalizer",
    "DocumentNumberNormalizerConfig",
    "DocumentNumberReason",
    "DocumentNumberRule",
    "DocumentNumberSpan",
    "build_default_document_number_rules",
    "normalize_document_number_runs",
]
