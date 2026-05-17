from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable

from asr_integer_extractor.fuzzy import best_fuzzy_match
from asr_integer_extractor.lexicon import (
    ASR_ALIASES,
    CANONICAL_WORD_VALUES,
    FILLER_WORDS,
    FRACTION_MARKERS,
    FRACTION_PREFIXES,
    NEGATIVE_WORDS,
)
from asr_integer_extractor.models import ExtractorConfig, IntegerSpan, LexicalMatch, NumericToken, Token
from asr_integer_extractor.normalization import tokenize_preserving_spans
from asr_integer_extractor.parser import parse_numeric_tokens


class IntegerExtractor:
    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.config: ExtractorConfig = config if config is not None else ExtractorConfig()

    def extract(self, text: str) -> list[IntegerSpan]:
        tokens: list[Token] = tokenize_preserving_spans(text)
        results: list[IntegerSpan] = []
        index: int = 0
        while index < len(tokens):
            token: Token = tokens[index]
            if not self._can_start_numeric_span(token):
                index += 1
                continue

            span_tokens: list[NumericToken] = self._collect_numeric_span(tokens, index)
            if not span_tokens:
                index += 1
                continue

            parsed = parse_numeric_tokens(span_tokens, self.config)
            if parsed is None:
                index += 1
                continue

            used_tokens: list[NumericToken] = span_tokens[: parsed.consumed_token_count]
            if parsed.has_fraction_tail and self.config.include_fraction_tail_in_span:
                used_tokens = self._include_fraction_tail(span_tokens, parsed.consumed_token_count)

            start: int = used_tokens[0].token.start
            end: int = used_tokens[-1].token.end
            raw: str = text[start:end]
            item = IntegerSpan(
                value=parsed.value,
                start=start,
                end=end,
                raw=raw,
                normalized=parsed.normalized,
                kind=parsed.kind,
                status=parsed.status,  # type: ignore[arg-type]
                confidence=parsed.confidence,
                candidates=parsed.candidates,
            )
            results.append(item)
            index = self._token_index_after(tokens, end)
        return results

    def replace(self, text: str) -> str:
        spans: list[IntegerSpan] = self.extract(text)
        replaced: str = replace_spans(text, spans)
        return replaced

    def to_json(self, text: str, *, ensure_ascii: bool = False) -> str:
        spans: list[IntegerSpan] = self.extract(text)
        payload: list[dict[str, object]] = [asdict(span) for span in spans]
        serialized: str = json.dumps(payload, ensure_ascii=ensure_ascii, indent=2)
        return serialized

    def _can_start_numeric_span(self, token: Token) -> bool:
        result: bool = False
        if token.kind == "digits":
            result = True
        elif token.normalized in NEGATIVE_WORDS:
            result = True
        elif self._lexical_match(token) is not None:
            result = True
        return result

    def _lexical_match(self, token: Token) -> LexicalMatch | None:
        if token.kind == "digits":
            return LexicalMatch(
                source=token.text,
                normalized=token.normalized,
                canonical=token.normalized,
                score=1.0,
                is_fuzzy=False,
            )

        normalized: str = token.normalized
        if normalized in CANONICAL_WORD_VALUES or normalized in NEGATIVE_WORDS or normalized in FRACTION_MARKERS:
            return LexicalMatch(
                source=token.text,
                normalized=normalized,
                canonical=normalized,
                score=1.0,
                is_fuzzy=token.text.lower().replace("ё", "е") in ASR_ALIASES,
            )

        if not self.config.fuzzy_enabled or token.kind != "word":
            return None

        fuzzy_match: LexicalMatch | None = best_fuzzy_match(
            normalized,
            threshold=self.config.fuzzy_threshold,
            min_length=self.config.min_token_length_for_fuzzy,
        )
        return fuzzy_match

    def _collect_numeric_span(self, tokens: list[Token], start_index: int) -> list[NumericToken]:
        numeric_tokens: list[NumericToken] = []
        index: int = start_index
        while index < len(tokens) and len(numeric_tokens) < self.config.max_span_tokens:
            token: Token = tokens[index]
            lexical_match: LexicalMatch | None = self._lexical_match(token)
            if lexical_match is not None:
                numeric_tokens.append(
                    NumericToken(
                        token=token,
                        canonical=lexical_match.canonical,
                        score=lexical_match.score,
                        is_fuzzy=lexical_match.is_fuzzy,
                    )
                )
                index += 1
                continue

            if token.kind == "word" and token.normalized in FRACTION_PREFIXES:
                can_include_prefix: bool = self._is_fraction_prefix_context(tokens, index, numeric_tokens)
                if can_include_prefix:
                    numeric_tokens.append(
                        NumericToken(
                            token=token,
                            canonical=token.normalized,
                            score=1.0,
                            is_fuzzy=False,
                        )
                    )
                    index += 1
                    continue

            if token.kind == "word" and token.normalized in FILLER_WORDS and not numeric_tokens:
                index += 1
                continue

            break

        return numeric_tokens

    def _is_fraction_prefix_context(
        self,
        tokens: list[Token],
        index: int,
        numeric_tokens: list[NumericToken],
    ) -> bool:
        if not numeric_tokens:
            return False
        next_index: int = index + 1
        if next_index >= len(tokens):
            return False
        next_token: Token = tokens[next_index]
        next_match: LexicalMatch | None = self._lexical_match(next_token)
        if next_match is None:
            return False
        result: bool = next_match.canonical in FRACTION_MARKERS
        return result

    def _include_fraction_tail(
        self,
        span_tokens: list[NumericToken],
        consumed_count: int,
    ) -> list[NumericToken]:
        used_tokens: list[NumericToken] = list(span_tokens[:consumed_count])
        for numeric_token in span_tokens[consumed_count:]:
            used_tokens.append(numeric_token)
            if numeric_token.canonical in FRACTION_MARKERS:
                break
        return used_tokens

    def _token_index_after(self, tokens: list[Token], end: int) -> int:
        next_index: int = len(tokens)
        for index, token in enumerate(tokens):
            if token.start >= end:
                next_index = index
                break
        return next_index


def replace_spans(text: str, spans: Iterable[IntegerSpan]) -> str:
    pieces: list[str] = []
    cursor: int = 0
    sorted_spans: list[IntegerSpan] = sorted(spans, key=lambda span: (span.start, span.end))
    for span in sorted_spans:
        if span.start < cursor:
            continue
        pieces.append(text[cursor : span.start])
        pieces.append(span.value)
        cursor = span.end
    pieces.append(text[cursor:])
    result: str = "".join(pieces)
    return result


def extract_integers(text: str, config: ExtractorConfig | None = None) -> list[IntegerSpan]:
    extractor = IntegerExtractor(config=config)
    spans: list[IntegerSpan] = extractor.extract(text)
    return spans


def replace_integer_spans(text: str, config: ExtractorConfig | None = None) -> str:
    extractor = IntegerExtractor(config=config)
    replaced: str = extractor.replace(text)
    return replaced


__all__: list[str] = [
    "IntegerExtractor",
    "extract_integers",
    "replace_integer_spans",
    "replace_spans",
]
