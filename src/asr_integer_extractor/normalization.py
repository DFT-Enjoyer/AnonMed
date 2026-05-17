from __future__ import annotations

import re
from typing import Final

from asr_integer_extractor.lexicon import ALIASES
from asr_integer_extractor.models import Token, TokenKind

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\d+|[A-Za-zА-Яа-яЁё]+|[^\w\s]", re.UNICODE)
_DASH_TRANSLATION: Final[dict[int, str]] = str.maketrans({"ё": "е", "Ё": "е", "—": "-", "–": "-", "−": "-"})


def normalize_token_text(text: str) -> str:
    lowered: str = text.lower().translate(_DASH_TRANSLATION)
    normalized: str = ALIASES.get(lowered, lowered)
    return normalized


def token_kind(text: str) -> TokenKind:
    kind: TokenKind = "punct"
    if text.isdigit():
        kind = "digits"
    elif text.isalpha():
        kind = "word"
    return kind


def tokenize_preserving_spans(text: str) -> list[Token]:
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(text):
        raw_text: str = match.group(0)
        normalized: str = normalize_token_text(raw_text)
        kind: TokenKind = token_kind(raw_text)
        tokens.append(
            Token(
                text=raw_text,
                normalized=normalized,
                start=match.start(),
                end=match.end(),
                kind=kind,
            )
        )
    return tokens


__all__: list[str] = ["normalize_token_text", "token_kind", "tokenize_preserving_spans"]
