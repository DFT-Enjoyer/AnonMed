from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TokenKind = Literal["word", "digits", "punct"]
SpanKind = Literal["digits", "digit_sequence", "cardinal", "mixed"]
SpanStatus = Literal["ok", "fuzzy_ok", "ambiguous", "failed"]


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    normalized: str
    start: int
    end: int
    kind: TokenKind


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    source: str
    normalized: str
    canonical: str
    score: float
    is_fuzzy: bool


@dataclass(frozen=True, slots=True)
class NumericToken:
    token: Token
    canonical: str
    score: float
    is_fuzzy: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    value: str
    kind: SpanKind
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class IntegerSpan:
    value: str
    start: int
    end: int
    raw: str
    normalized: str
    kind: SpanKind
    status: SpanStatus
    confidence: float
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ExtractorConfig:
    prefer_digit_sequences: bool = True
    digit_sequence_min_tokens: int = 2
    fuzzy_enabled: bool = True
    fuzzy_threshold: float = 0.82
    min_token_length_for_fuzzy: int = 4
    include_fraction_tail_in_span: bool = True
    max_span_tokens: int = 24
    allow_negative: bool = True


__all__: list[str] = [
    "Candidate",
    "ExtractorConfig",
    "IntegerSpan",
    "LexicalMatch",
    "NumericToken",
    "SpanKind",
    "SpanStatus",
    "Token",
    "TokenKind",
]
