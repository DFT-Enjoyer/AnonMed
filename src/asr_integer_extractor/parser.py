from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from asr_integer_extractor.lexicon import DIGIT_WORDS, FRACTION_MARKERS, HUNDREDS, NEGATIVE_WORDS, SCALES, TEENS, TENS
from asr_integer_extractor.models import Candidate, ExtractorConfig, NumericToken, SpanKind
from asr_integer_extractor.scoring import build_feature_vector, score_confidence


@dataclass(frozen=True, slots=True)
class ParsedSpan:
    value: str
    kind: SpanKind
    status: str
    confidence: float
    normalized: str
    candidates: tuple[Candidate, ...]
    consumed_token_count: int
    has_fraction_tail: bool


_NUMERIC_VALUE_WORDS: Final[frozenset[str]] = frozenset(
    set(DIGIT_WORDS) | set(TEENS) | set(TENS) | set(HUNDREDS) | set(SCALES)
)


def is_fraction_marker(canonical: str) -> bool:
    return canonical in FRACTION_MARKERS


def is_negative_marker(canonical: str) -> bool:
    return canonical in NEGATIVE_WORDS


def is_digit_atom(token: NumericToken) -> bool:
    canonical: str = token.canonical
    result: bool = False
    if token.token.kind == "digits":
        result = True
    elif canonical in DIGIT_WORDS:
        result = True
    return result


def digit_atom_to_string(token: NumericToken) -> str:
    if token.token.kind == "digits":
        return token.token.normalized
    value: int = DIGIT_WORDS[token.canonical]
    result: str = str(value)
    return result


def has_cardinal_structure(tokens: list[NumericToken]) -> bool:
    result: bool = False
    for numeric_token in tokens:
        canonical: str = numeric_token.canonical
        if canonical in TEENS or canonical in TENS or canonical in HUNDREDS or canonical in SCALES:
            result = True
            break
    return result


def parse_digit_sequence(tokens: list[NumericToken]) -> Candidate | None:
    sign: str = ""
    body_tokens: list[NumericToken] = tokens
    if tokens and is_negative_marker(tokens[0].canonical):
        sign = "-"
        body_tokens = tokens[1:]

    if not body_tokens:
        return None

    can_parse: bool = all(is_digit_atom(numeric_token) for numeric_token in body_tokens)
    if not can_parse:
        return None

    digits: str = "".join(digit_atom_to_string(numeric_token) for numeric_token in body_tokens)
    if not digits:
        return None

    candidate: Candidate = Candidate(
        value=f"{sign}{digits}",
        kind="digit_sequence",
        confidence=0.98,
        reason="all tokens are digit atoms",
    )
    return candidate


def parse_cardinal(tokens: list[NumericToken]) -> Candidate | None:
    body_tokens: list[NumericToken] = tokens[1:] if tokens and is_negative_marker(tokens[0].canonical) else tokens
    if len(body_tokens) > 1 and not has_cardinal_structure(body_tokens) and all(is_digit_atom(token) for token in body_tokens):
        return None

    sign: int = 1
    total: int = 0
    group: int = 0
    seen_any: bool = False
    last_small_rank: int = 10_000

    for position, numeric_token in enumerate(tokens):
        canonical: str = numeric_token.canonical
        token_text: str = numeric_token.token.normalized

        if position == 0 and is_negative_marker(canonical):
            sign = -1
            continue

        if numeric_token.token.kind == "digits":
            digit_value: int = int(token_text)
            if digit_value >= 1000:
                total += digit_value
                seen_any = True
                last_small_rank = 10_000
                continue
            group += digit_value
            seen_any = True
            last_small_rank = min(last_small_rank, 1)
            continue

        if canonical in HUNDREDS:
            if last_small_rank <= 100:
                return None
            group += HUNDREDS[canonical]
            seen_any = True
            last_small_rank = 100
            continue

        if canonical in TENS:
            if last_small_rank <= 10:
                return None
            group += TENS[canonical]
            seen_any = True
            last_small_rank = 10
            continue

        if canonical in TEENS:
            if last_small_rank <= 10:
                return None
            group += TEENS[canonical]
            seen_any = True
            last_small_rank = 10
            continue

        if canonical in DIGIT_WORDS:
            if last_small_rank <= 1 and has_cardinal_structure(tokens):
                return None
            group += DIGIT_WORDS[canonical]
            seen_any = True
            last_small_rank = 1
            continue

        if canonical in SCALES:
            multiplier: int = SCALES[canonical]
            safe_group: int = group if group > 0 else 1
            total += safe_group * multiplier
            group = 0
            seen_any = True
            last_small_rank = 10_000
            continue

        return None

    if not seen_any:
        return None

    value: int = sign * (total + group)
    candidate: Candidate = Candidate(
        value=str(value),
        kind="cardinal",
        confidence=0.96,
        reason="cardinal grammar parse",
    )
    return candidate


def parse_digit_or_mixed(tokens: list[NumericToken]) -> Candidate | None:
    if not tokens:
        return None

    sign: str = ""
    body_tokens: list[NumericToken] = tokens
    if is_negative_marker(tokens[0].canonical):
        sign = "-"
        body_tokens = tokens[1:]

    if len(body_tokens) != 1:
        return None

    numeric_token: NumericToken = body_tokens[0]
    if numeric_token.token.kind != "digits":
        return None

    candidate: Candidate = Candidate(
        value=f"{sign}{numeric_token.token.normalized}",
        kind="digits",
        confidence=1.0,
        reason="digit span already in written form",
    )
    return candidate


def trim_integer_tokens(tokens: list[NumericToken]) -> tuple[list[NumericToken], bool]:
    trimmed: list[NumericToken] = []
    has_fraction_tail: bool = False
    index: int = 0
    while index < len(tokens):
        numeric_token: NumericToken = tokens[index]
        canonical: str = numeric_token.canonical
        if is_fraction_marker(canonical):
            has_fraction_tail = True
            break
        if canonical in {"с", "и"}:
            next_index: int = index + 1
            if next_index < len(tokens) and is_fraction_marker(tokens[next_index].canonical):
                has_fraction_tail = True
            break
        if canonical not in _NUMERIC_VALUE_WORDS and not is_negative_marker(canonical) and numeric_token.token.kind != "digits":
            break
        trimmed.append(numeric_token)
        index += 1
    return trimmed, has_fraction_tail


def parse_numeric_tokens(tokens: list[NumericToken], config: ExtractorConfig) -> ParsedSpan | None:
    trimmed_tokens: list[NumericToken]
    has_fraction_tail: bool
    trimmed_tokens, has_fraction_tail = trim_integer_tokens(tokens)
    if not trimmed_tokens:
        return None

    candidates: list[Candidate] = []
    digit_candidate: Candidate | None = parse_digit_or_mixed(trimmed_tokens)
    if digit_candidate is not None:
        candidates.append(digit_candidate)

    sequence_candidate: Candidate | None = parse_digit_sequence(trimmed_tokens)
    if sequence_candidate is not None and len(trimmed_tokens) >= config.digit_sequence_min_tokens:
        candidates.append(sequence_candidate)

    cardinal_candidate: Candidate | None = parse_cardinal(trimmed_tokens)
    if cardinal_candidate is not None:
        candidates.append(cardinal_candidate)

    if not candidates:
        return None

    selected: Candidate = candidates[0]
    if config.prefer_digit_sequences:
        sequence_candidates: list[Candidate] = [candidate for candidate in candidates if candidate.kind == "digit_sequence"]
        if sequence_candidates:
            selected = sequence_candidates[0]
        else:
            selected = max(candidates, key=lambda candidate: candidate.confidence)
    else:
        selected = max(candidates, key=lambda candidate: candidate.confidence)

    fuzzy_count: int = sum(1 for numeric_token in trimmed_tokens if numeric_token.is_fuzzy)
    score_sum: float = sum(numeric_token.score for numeric_token in trimmed_tokens)
    mean_score: float = score_sum / float(len(trimmed_tokens))
    features = build_feature_vector(
        token_count=len(trimmed_tokens),
        fuzzy_count=fuzzy_count,
        mean_lexical_score=mean_score,
        has_fraction_tail=has_fraction_tail,
        kind=selected.kind,
    )
    confidence: float = min(selected.confidence, score_confidence(features))
    status: str = "fuzzy_ok" if fuzzy_count > 0 else "ok"
    normalized: str = " ".join(numeric_token.canonical for numeric_token in trimmed_tokens)
    parsed = ParsedSpan(
        value=selected.value,
        kind=selected.kind,
        status=status,
        confidence=confidence,
        normalized=normalized,
        candidates=tuple(candidates),
        consumed_token_count=len(trimmed_tokens),
        has_fraction_tail=has_fraction_tail,
    )
    return parsed


__all__: list[str] = [
    "ParsedSpan",
    "digit_atom_to_string",
    "has_cardinal_structure",
    "is_digit_atom",
    "is_fraction_marker",
    "is_negative_marker",
    "parse_cardinal",
    "parse_digit_or_mixed",
    "parse_digit_sequence",
    "parse_numeric_tokens",
    "trim_integer_tokens",
]
