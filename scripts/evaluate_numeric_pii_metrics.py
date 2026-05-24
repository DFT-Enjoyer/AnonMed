#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, cast

from anonmed.anonymization import NumericPIIType, normalize_numeric_pii_value, run_numeric_pii_pipeline
from anonmed.preprocessing.asr.number_extractor import IntegerExtractor
from anonmed.preprocessing.asr.number_parser import parse_numeric_tokens
from anonmed.preprocessing.asr.tokenization import tokenize_preserving_spans
from anonmed.preprocessing.asr.types import ExtractorConfig, NumericToken, Token

NUMERIC_TYPE_MAP: dict[str, str] = {
    "ТЕЛЕФОН": "PHONE",
    "phone": "PHONE",
    "СНИЛС": "SNILS",
    "snils": "SNILS",
    "ПАСПОРТ": "PASSPORT",
    "passport": "PASSPORT",
    "ДАТА_РОЖДЕНИЯ": "DATE_BIRTH",
    "birthdate": "DATE_BIRTH",
    "date_birth": "DATE_BIRTH",
    "ОМС": "OMS",
    "oms": "OMS",
    "ИНН": "INN",
    "inn": "INN",
    "ВОЗРАСТ": "AGE",
    "age": "AGE",
    "МСЭ": "MSE",
    "mse": "MSE",
    "СВИДЕТЕЛЬСТВО": "BIRTH_CERTIFICATE",
    "birth_certificate": "BIRTH_CERTIFICATE",
    "ВУ": "DRIVER_LICENSE",
    "driver_license": "DRIVER_LICENSE",
}

DIRECT_IDENTIFIER_TYPES: frozenset[str] = frozenset(
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
QUASI_IDENTIFIER_TYPES: frozenset[str] = frozenset({"AGE"})

MONTH_MAP: dict[str, int] = {
    "января": 1,
    "январь": 1,
    "февраля": 2,
    "февраль": 2,
    "марта": 3,
    "март": 3,
    "апреля": 4,
    "апрель": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июля": 7,
    "июль": 7,
    "августа": 8,
    "август": 8,
    "сентября": 9,
    "сентябрь": 9,
    "октября": 10,
    "октябрь": 10,
    "ноября": 11,
    "ноябрь": 11,
    "декабря": 12,
    "декабрь": 12,
}

ORDINAL_PREFIXES: tuple[tuple[str, int], ...] = (
    ("тридцать перв", 31),
    ("тридцать", 30),
    ("двадцать девят", 29),
    ("двадцать восьм", 28),
    ("двадцать седьм", 27),
    ("двадцать шест", 26),
    ("двадцать пят", 25),
    ("двадцать четверт", 24),
    ("двадцать трет", 23),
    ("двадцать втор", 22),
    ("двадцать перв", 21),
    ("двадцать", 20),
    ("девятнадцат", 19),
    ("восемнадцат", 18),
    ("семнадцат", 17),
    ("шестнадцат", 16),
    ("пятнадцат", 15),
    ("четырнадцат", 14),
    ("тринадцат", 13),
    ("двенадцат", 12),
    ("одиннадцат", 11),
    ("десят", 10),
    ("девят", 9),
    ("восьм", 8),
    ("седьм", 7),
    ("шест", 6),
    ("пят", 5),
    ("четверт", 4),
    ("трет", 3),
    ("втор", 2),
    ("перв", 1),
)

MAX_NUMERIC_SEGMENT_TOKENS: int = 8
NUMERIC_EXTRACTOR = IntegerExtractor()
NUMERIC_CONFIG = ExtractorConfig()

__all__: tuple[str, ...] = ("main",)


@dataclass(frozen=True, slots=True)
class NumericAnnotation:
    record_id: int
    pii_type: str
    start: int
    end: int
    raw_text: str
    normalized_text: str
    digit_segments: tuple[str, ...]
    digit_sequence: str
    canonical_value: str | None


@dataclass(frozen=True, slots=True)
class NumericPrediction:
    record_id: int
    pii_type: str
    start: int
    end: int
    normalized_start: int
    normalized_end: int
    raw_value: str
    canonical_value: str
    confidence: float
    rule_id: str
    entity_id: str
    mention_id: str


@dataclass(frozen=True, slots=True)
class EvaluatedRecord:
    source: dict[str, object]
    annotations: tuple[NumericAnnotation, ...]
    predictions: tuple[NumericPrediction, ...]
    preprocessed_text: str
    masked_text: str
    masked_original_text: str
    repetition_suppressed_indexes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SpanPair:
    annotation: NumericAnnotation
    prediction: NumericPrediction
    overlap: int


@dataclass(frozen=True, slots=True)
class CharacterCounts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denominator: int = self.tp + self.fp
        return 0.0 if denominator == 0 else self.tp / denominator

    @property
    def recall(self) -> float:
        denominator: int = self.tp + self.fn
        return 0.0 if denominator == 0 else self.tp / denominator

    @property
    def f1(self) -> float:
        precision_value: float = self.precision
        recall_value: float = self.recall
        denominator: float = precision_value + recall_value
        return 0.0 if denominator == 0.0 else 2.0 * precision_value * recall_value / denominator


@dataclass(frozen=True, slots=True)
class MetricCounts:
    tp: int
    fp: int
    fn: int
    tn: int
    negative_fp: int

    @property
    def precision(self) -> float:
        denominator: int = self.tp + self.fp
        return 0.0 if denominator == 0 else self.tp / denominator

    @property
    def recall(self) -> float:
        denominator: int = self.tp + self.fn
        return 0.0 if denominator == 0 else self.tp / denominator

    @property
    def f1(self) -> float:
        precision_value: float = self.precision
        recall_value: float = self.recall
        denominator: float = precision_value + recall_value
        return 0.0 if denominator == 0.0 else 2.0 * precision_value * recall_value / denominator

    @property
    def accuracy(self) -> float:
        denominator: int = self.tp + self.fp + self.fn + self.tn
        return 0.0 if denominator == 0 else (self.tp + self.tn) / denominator

    @property
    def specificity(self) -> float:
        denominator: int = self.tn + self.negative_fp
        return 0.0 if denominator == 0 else self.tn / denominator


def digits_only(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def tokenize_words(text: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize_preserving_spans(text):
        if token.kind != "word":
            continue
        tokens.append(token.normalized)
    return tokens


@lru_cache(maxsize=None)
def try_parse_date_birth(text: str) -> str | None:
    words: list[str] = tokenize_words(text)
    month_index: int | None = None
    month_value: int | None = None
    for index, word in enumerate(words):
        if word in MONTH_MAP:
            month_index = index
            month_value = MONTH_MAP[word]
            break
    if month_index is None or month_value is None:
        return None

    day_value: int | None = parse_day_words(words[:month_index])
    if day_value is None:
        return None

    year_words: list[str] = words[month_index + 1 :]
    year_value: int | None = parse_year_words(year_words)
    if year_value is None:
        return None

    current_year: int = date.today().year
    if year_value < 1900 or year_value > current_year:
        return None

    try:
        date(year_value, month_value, day_value)
    except ValueError:
        return None

    return f"{day_value:02d}.{month_value:02d}.{year_value:04d}"


def parse_day_words(words: list[str]) -> int | None:
    if not words:
        return None
    phrase: str = normalize_whitespace(" ".join(words))
    for prefix, value in ORDINAL_PREFIXES:
        if phrase.startswith(prefix):
            return value
    return None


def parse_year_words(words: list[str]) -> int | None:
    if not words:
        return None
    segments: tuple[str, ...] = extract_digit_segments(" ".join(words))
    if len(segments) != 1:
        return None
    try:
        return int(segments[0])
    except ValueError:
        return None


def to_numeric_token(token: Token) -> NumericToken | None:
    lexical_match = NUMERIC_EXTRACTOR._lexical_match(token)
    if lexical_match is None:
        return None
    return NumericToken(
        token=token,
        canonical=lexical_match.canonical,
        score=lexical_match.score,
        is_fuzzy=lexical_match.is_fuzzy,
    )


@lru_cache(maxsize=None)
def extract_digit_segments(text: str) -> tuple[str, ...]:
    tokens: list[Token] = tokenize_preserving_spans(text)
    segments: list[str] = []
    index: int = 0
    while index < len(tokens):
        token: Token = tokens[index]
        if token.kind == "word" and token.normalized in {"плюс", "плюсом"}:
            index += 1
            continue

        if token.kind == "punct":
            index += 1
            continue

        best_value: str | None = None
        best_consumed: int = 0
        upper_bound: int = min(len(tokens), index + MAX_NUMERIC_SEGMENT_TOKENS)
        for end_index in range(upper_bound, index, -1):
            prefix_tokens: list[NumericToken] = []
            valid_prefix: bool = True
            for prefix_token in tokens[index:end_index]:
                numeric_token = to_numeric_token(prefix_token)
                if numeric_token is None:
                    valid_prefix = False
                    break
                prefix_tokens.append(numeric_token)
            if not valid_prefix or not prefix_tokens:
                continue

            parsed = parse_numeric_tokens(prefix_tokens, NUMERIC_CONFIG)
            if parsed is None or parsed.consumed_token_count != len(prefix_tokens):
                continue

            best_value = parsed.value
            best_consumed = len(prefix_tokens)
            break

        if best_value is None:
            numeric_token = to_numeric_token(token)
            if numeric_token is not None and numeric_token.token.kind == "digits":
                segments.append(numeric_token.token.normalized)
            index += 1
            continue

        segments.append(best_value)
        index += best_consumed

    normalized_segments: list[str] = []
    for segment in segments:
        cleaned_segment: str = digits_only(segment)
        if cleaned_segment:
            normalized_segments.append(cleaned_segment)
    return tuple(normalized_segments)


@lru_cache(maxsize=None)
def canonicalize_annotation(pii_type: str, raw_text: str) -> tuple[str, tuple[str, ...], str, str | None]:
    normalized_text: str = normalize_whitespace(raw_text)
    if pii_type == "DATE_BIRTH":
        parsed_date: str | None = try_parse_date_birth(raw_text)
        return normalized_text, (), "", parsed_date

    digit_segments: tuple[str, ...] = extract_digit_segments(raw_text)
    digit_sequence: str = "".join(digit_segments)

    candidates: list[str] = []
    if normalized_text:
        candidates.append(normalized_text)
    if digit_sequence and digit_sequence not in candidates:
        candidates.append(digit_sequence)
    raw_digits: str = digits_only(raw_text)
    if raw_digits and raw_digits not in candidates:
        candidates.append(raw_digits)

    canonical_value: str | None = None
    for candidate in candidates:
        canonical_value = normalize_numeric_pii_value(cast(NumericPIIType, pii_type), candidate)
        if canonical_value is not None:
            break

    return normalized_text, digit_segments, digit_sequence, canonical_value


def annotation_key(annotation: NumericAnnotation) -> tuple[str, str]:
    if annotation.canonical_value is not None:
        return annotation.pii_type, annotation.canonical_value
    if annotation.digit_sequence:
        return annotation.pii_type, f"digits:{annotation.digit_sequence}"
    return annotation.pii_type, f"text:{annotation.normalized_text.lower()}"


def prediction_key(prediction: NumericPrediction) -> tuple[str, str]:
    return prediction.pii_type, prediction.canonical_value


def load_annotations(record: dict[str, object]) -> list[NumericAnnotation]:
    annotations: list[NumericAnnotation] = []
    record_id: int = int(str(record["id"]))
    raw_annotations: object = record.get("annotations", [])
    if not isinstance(raw_annotations, list):
        return annotations
    for annotation in raw_annotations:
        annotation_dict: dict[str, object] = dict(annotation)
        raw_type: str = str(annotation_dict["type"])
        pii_type: str | None = NUMERIC_TYPE_MAP.get(raw_type)
        if pii_type is None:
            continue
        raw_text: str = str(annotation_dict["text"])
        normalized_text, digit_segments, digit_sequence, canonical_value = canonicalize_annotation(
            pii_type,
            raw_text,
        )
        annotations.append(
            NumericAnnotation(
                record_id=record_id,
                pii_type=pii_type,
                start=int(str(annotation_dict["start"])),
                end=int(str(annotation_dict["end"])),
                raw_text=raw_text,
                normalized_text=normalized_text,
                digit_segments=digit_segments,
                digit_sequence=digit_sequence,
                canonical_value=canonical_value,
            )
        )
    return annotations


def load_predictions(
    record: dict[str, object],
    *,
    deduplicate_repetitions: bool = False,
    normalize_document_numbers: bool = True,
) -> tuple[list[NumericPrediction], str, str, str, tuple[int, ...]]:
    record_id: int = int(str(record["id"]))
    text: str = str(record["value"])
    predictions: list[NumericPrediction] = []
    pipeline_result = run_numeric_pii_pipeline(
        text,
        deduplicate_repetitions=deduplicate_repetitions,
        normalize_document_numbers=normalize_document_numbers,
    )
    for match in pipeline_result.matches:
        predictions.append(
            NumericPrediction(
                record_id=record_id,
                pii_type=match.pii_type,
                start=match.start,
                end=match.end,
                normalized_start=match.normalized_start,
                normalized_end=match.normalized_end,
                raw_value=match.value,
                canonical_value=match.normalized_value,
                confidence=match.confidence,
                rule_id=match.rule_id,
                entity_id=str(match.metadata.get("entity_id", "")),
                mention_id=str(match.metadata.get("mention_id", "")),
            )
        )
    return (
        predictions,
        pipeline_result.preprocessing_result.normalized_text,
        pipeline_result.masked_normalized_text,
        pipeline_result.masked_original_text,
        pipeline_result.preprocessing_result.repetition_suppressed_indexes,
    )


def cluster_annotations(
    annotations: Sequence[NumericAnnotation],
    max_gap: int,
) -> list[list[NumericAnnotation]]:
    if not annotations:
        return []
    sorted_annotations: list[NumericAnnotation] = sorted(annotations, key=lambda item: (item.start, item.end))
    clusters: list[list[NumericAnnotation]] = [[sorted_annotations[0]]]
    for annotation in sorted_annotations[1:]:
        current_cluster: list[NumericAnnotation] = clusters[-1]
        previous: NumericAnnotation = current_cluster[-1]
        same_type: bool = previous.pii_type == annotation.pii_type
        gap: int = annotation.start - previous.end
        if same_type and gap <= max_gap:
            current_cluster.append(annotation)
            continue
        clusters.append([annotation])
    return clusters


def generate_soft_keys(
    annotations: Sequence[NumericAnnotation],
    max_gap: int,
) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    annotations_by_type: dict[str, list[NumericAnnotation]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_type[annotation.pii_type].append(annotation)

    for pii_type, typed_annotations in annotations_by_type.items():
        for cluster in cluster_annotations(typed_annotations, max_gap=max_gap):
            for annotation in cluster:
                if annotation.canonical_value is not None:
                    keys.add((pii_type, annotation.canonical_value))

            if pii_type in {"DATE_BIRTH", "AGE"}:
                continue

            for start_index in range(len(cluster)):
                concatenated: str = ""
                for end_index in range(start_index, len(cluster)):
                    digit_sequence: str = cluster[end_index].digit_sequence
                    if not digit_sequence:
                        break
                    concatenated += digit_sequence
                    if not concatenated:
                        continue
                    normalized_value: str | None = normalize_numeric_pii_value(
                        cast(NumericPIIType, pii_type),
                        concatenated,
                    )
                    if normalized_value is not None:
                        keys.add((pii_type, normalized_value))
    return keys


def evaluate_hard_negatives(
    records: list[dict[str, object]],
    *,
    normalize_document_numbers: bool,
) -> tuple[int, int, dict[str, int]]:
    total: int = 0
    false_positives: int = 0
    per_category_fp: dict[str, int] = defaultdict(int)
    for record in records:
        raw_hard_negatives: object = record.get("hard_negatives", [])
        if not isinstance(raw_hard_negatives, list):
            continue
        for hard_negative in raw_hard_negatives:
            hard_negative_dict: dict[str, object] = dict(hard_negative)
            raw_text: str = str(hard_negative_dict["text"])
            category: str = str(hard_negative_dict["category"])
            total += 1
            if hard_negative_has_match(raw_text, normalize_document_numbers):
                false_positives += 1
                per_category_fp[category] += 1
    return total, false_positives, dict(sorted(per_category_fp.items()))


@lru_cache(maxsize=None)
def hard_negative_has_match(raw_text: str, normalize_document_numbers: bool) -> bool:
    return bool(
        run_numeric_pii_pipeline(
            raw_text,
            normalize_document_numbers=normalize_document_numbers,
        ).matches
    )


def multiset_match_counts(
    gt_keys: Iterable[tuple[str, str]],
    pred_keys: Iterable[tuple[str, str]],
) -> tuple[int, int, int, dict[str, tuple[int, int, int]]]:
    gt_counter: Counter[tuple[str, str]] = Counter(gt_keys)
    pred_counter: Counter[tuple[str, str]] = Counter(pred_keys)

    tp: int = 0
    per_type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for key in set(gt_counter) | set(pred_counter):
        gt_count: int = gt_counter.get(key, 0)
        pred_count: int = pred_counter.get(key, 0)
        matched: int = min(gt_count, pred_count)
        tp += matched
        pii_type: str = key[0]
        per_type_counts[pii_type][0] += matched
        per_type_counts[pii_type][1] += max(pred_count - matched, 0)
        per_type_counts[pii_type][2] += max(gt_count - matched, 0)

    fp: int = sum(pred_counter.values()) - tp
    fn: int = sum(gt_counter.values()) - tp
    serialized_per_type: dict[str, tuple[int, int, int]] = {
        pii_type: (values[0], values[1], values[2]) for pii_type, values in per_type_counts.items()
    }
    return tp, fp, fn, serialized_per_type


def build_metric_counts(
    tp: int,
    fp: int,
    fn: int,
    hard_negative_total: int,
    hard_negative_false_positives: int,
) -> MetricCounts:
    tn: int = max(hard_negative_total - hard_negative_false_positives, 0)
    return MetricCounts(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        negative_fp=hard_negative_false_positives,
    )


def format_ratio(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def print_metric_block(name: str, counts: MetricCounts) -> None:
    print(name)
    print(f"  precision: {format_ratio(counts.precision)}")
    print(f"  recall:    {format_ratio(counts.recall)}")
    print(f"  f1:        {format_ratio(counts.f1)}")
    print(f"  accuracy:  {format_ratio(counts.accuracy)}")
    print(f"  specificity:{format_ratio(counts.specificity)}")


def print_per_type_table(
    title: str,
    per_type_counts: dict[str, tuple[int, int, int]],
    hard_negative_total: int,
    hard_negative_false_positives: int,
) -> None:
    print(title)
    header: str = (
        f"{'type':<18} {'tp':>5} {'fp':>5} {'fn':>5} "
        f"{'precision':>10} {'recall':>10} {'f1':>10}"
    )
    print(header)
    for pii_type in sorted(set(NUMERIC_TYPE_MAP.values())):
        tp, fp, fn = per_type_counts.get(pii_type, (0, 0, 0))
        counts = build_metric_counts(
            tp=tp,
            fp=fp,
            fn=fn,
            hard_negative_total=hard_negative_total,
            hard_negative_false_positives=hard_negative_false_positives,
        )
        print(
            f"{pii_type:<18} {tp:>5} {fp:>5} {fn:>5} "
            f"{format_ratio(counts.precision):>10} "
            f"{format_ratio(counts.recall):>10} "
            f"{format_ratio(counts.f1):>10}"
        )


def span_overlap_length(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> int:
    return max(0, min(first_end, second_end) - max(first_start, second_start))


def annotation_identity(annotation: NumericAnnotation) -> tuple[int, str, int, int, str]:
    return (
        annotation.record_id,
        annotation.pii_type,
        annotation.start,
        annotation.end,
        annotation.raw_text,
    )


def prediction_identity(prediction: NumericPrediction) -> tuple[int, str, int, int, str]:
    return (
        prediction.record_id,
        prediction.pii_type,
        prediction.start,
        prediction.end,
        prediction.canonical_value,
    )


def metric_counts_payload(counts: MetricCounts) -> dict[str, object]:
    return {
        "counts": {
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "tn": counts.tn,
            "negative_fp": counts.negative_fp,
        },
        "metrics": {
            "precision": counts.precision,
            "recall": counts.recall,
            "f1": counts.f1,
            "accuracy": counts.accuracy,
            "specificity": counts.specificity,
        },
    }


def character_counts_payload(counts: CharacterCounts) -> dict[str, object]:
    return {
        "counts": {
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
        },
        "metrics": {
            "precision": counts.precision,
            "recall": counts.recall,
            "f1": counts.f1,
        },
    }


def pii_type_bucket(pii_type: str) -> Literal["direct", "quasi", "other"]:
    if pii_type in DIRECT_IDENTIFIER_TYPES:
        return "direct"
    if pii_type in QUASI_IDENTIFIER_TYPES:
        return "quasi"
    return "other"


def percentile(values: list[int], percentile_value: float) -> float:
    if not values:
        return 0.0
    sorted_values: list[int] = sorted(values)
    bounded_percentile: float = min(100.0, max(0.0, percentile_value))
    index: int = math.ceil((bounded_percentile / 100.0) * len(sorted_values)) - 1
    bounded_index: int = min(len(sorted_values) - 1, max(0, index))
    return float(sorted_values[bounded_index])


def match_span_pairs(
    annotations: Sequence[NumericAnnotation],
    predictions: Sequence[NumericPrediction],
    *,
    require_type: bool,
) -> tuple[SpanPair, ...]:
    candidate_pairs: list[tuple[int, int, int, int]] = []
    for annotation_index, annotation in enumerate(annotations):
        for prediction_index, prediction in enumerate(predictions):
            if annotation.record_id != prediction.record_id:
                continue
            if require_type and annotation.pii_type != prediction.pii_type:
                continue
            overlap: int = span_overlap_length(
                annotation.start,
                annotation.end,
                prediction.start,
                prediction.end,
            )
            if overlap <= 0:
                continue
            annotation_length: int = max(1, annotation.end - annotation.start)
            prediction_length: int = max(1, prediction.end - prediction.start)
            length_penalty: int = abs(annotation_length - prediction_length)
            candidate_pairs.append((overlap, -length_penalty, annotation_index, prediction_index))

    candidate_pairs.sort(reverse=True)
    used_annotations: set[int] = set()
    used_predictions: set[int] = set()
    pairs: list[SpanPair] = []
    for overlap, _negative_penalty, annotation_index, prediction_index in candidate_pairs:
        if annotation_index in used_annotations or prediction_index in used_predictions:
            continue
        annotation = annotations[annotation_index]
        prediction = predictions[prediction_index]
        pairs.append(SpanPair(annotation=annotation, prediction=prediction, overlap=overlap))
        used_annotations.add(annotation_index)
        used_predictions.add(prediction_index)

    return tuple(sorted(pairs, key=lambda item: (item.annotation.record_id, item.annotation.start)))


def span_counts_from_pairs(
    annotations: Sequence[NumericAnnotation],
    predictions: Sequence[NumericPrediction],
    pairs: Sequence[SpanPair],
    hard_negative_total: int,
    hard_negative_false_positives: int,
) -> tuple[MetricCounts, dict[str, tuple[int, int, int]]]:
    matched_annotations: set[tuple[int, str, int, int, str]] = {
        annotation_identity(pair.annotation) for pair in pairs
    }
    matched_predictions: set[tuple[int, str, int, int, str]] = {
        prediction_identity(pair.prediction) for pair in pairs
    }
    per_type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for pair in pairs:
        per_type_counts[pair.annotation.pii_type][0] += 1

    for prediction in predictions:
        if prediction_identity(prediction) not in matched_predictions:
            per_type_counts[prediction.pii_type][1] += 1

    for annotation in annotations:
        if annotation_identity(annotation) not in matched_annotations:
            per_type_counts[annotation.pii_type][2] += 1

    tp: int = len(pairs)
    fp: int = len(predictions) - tp
    fn: int = len(annotations) - tp
    counts: MetricCounts = build_metric_counts(
        tp=tp,
        fp=fp,
        fn=fn,
        hard_negative_total=hard_negative_total,
        hard_negative_false_positives=hard_negative_false_positives,
    )
    serialized_per_type: dict[str, tuple[int, int, int]] = {
        pii_type: (values[0], values[1], values[2])
        for pii_type, values in sorted(per_type_counts.items())
    }
    return counts, serialized_per_type


def build_exact_span_report(
    annotations: Sequence[NumericAnnotation],
    predictions: Sequence[NumericPrediction],
    *,
    hard_negative_total: int,
    hard_negative_false_positives: int,
) -> dict[str, object]:
    exact_gt_keys: list[tuple[str, str]] = [
        (
            annotation.pii_type,
            f"{annotation.record_id}:{annotation.start}:{annotation.end}",
        )
        for annotation in annotations
    ]
    exact_pred_keys: list[tuple[str, str]] = [
        (
            prediction.pii_type,
            f"{prediction.record_id}:{prediction.start}:{prediction.end}",
        )
        for prediction in predictions
    ]
    tp, fp, fn, per_type = multiset_match_counts(exact_gt_keys, exact_pred_keys)
    counts: MetricCounts = build_metric_counts(
        tp=tp,
        fp=fp,
        fn=fn,
        hard_negative_total=hard_negative_total,
        hard_negative_false_positives=hard_negative_false_positives,
    )
    payload: dict[str, object] = metric_counts_payload(counts)
    payload["per_type"] = {
        pii_type: {"tp": values[0], "fp": values[1], "fn": values[2]}
        for pii_type, values in sorted(per_type.items())
    }
    return payload


def build_relaxed_span_report(
    annotations: Sequence[NumericAnnotation],
    predictions: Sequence[NumericPrediction],
    *,
    hard_negative_total: int,
    hard_negative_false_positives: int,
) -> tuple[dict[str, object], tuple[SpanPair, ...]]:
    pairs: tuple[SpanPair, ...] = match_span_pairs(
        annotations,
        predictions,
        require_type=True,
    )
    counts, per_type = span_counts_from_pairs(
        annotations,
        predictions,
        pairs,
        hard_negative_total,
        hard_negative_false_positives,
    )
    payload: dict[str, object] = metric_counts_payload(counts)
    payload["per_type"] = {
        pii_type: {"tp": values[0], "fp": values[1], "fn": values[2]}
        for pii_type, values in sorted(per_type.items())
    }
    payload["boundary_errors"] = build_boundary_error_report(pairs)
    return payload, pairs


def build_boundary_error_report(pairs: Sequence[SpanPair]) -> dict[str, object]:
    start_errors: list[int] = []
    end_errors: list[int] = []
    boundary_errors: list[int] = []
    exact_boundary_matches: int = 0
    for pair in pairs:
        start_error: int = abs(pair.annotation.start - pair.prediction.start)
        end_error: int = abs(pair.annotation.end - pair.prediction.end)
        total_error: int = start_error + end_error
        start_errors.append(start_error)
        end_errors.append(end_error)
        boundary_errors.append(total_error)
        if total_error == 0:
            exact_boundary_matches += 1

    pair_count: int = len(pairs)
    mean_boundary_error: float = 0.0 if pair_count == 0 else sum(boundary_errors) / pair_count
    mean_start_error: float = 0.0 if pair_count == 0 else sum(start_errors) / pair_count
    mean_end_error: float = 0.0 if pair_count == 0 else sum(end_errors) / pair_count
    return {
        "matched_pairs": pair_count,
        "exact_boundary_match_rate": 0.0 if pair_count == 0 else exact_boundary_matches / pair_count,
        "mean_boundary_error_chars": mean_boundary_error,
        "p50_boundary_error_chars": percentile(boundary_errors, 50.0),
        "p95_boundary_error_chars": percentile(boundary_errors, 95.0),
        "mean_start_error_chars": mean_start_error,
        "mean_end_error_chars": mean_end_error,
    }


def char_positions_for_annotations(
    annotations: Sequence[NumericAnnotation],
    *,
    bucket: Literal["all", "direct", "quasi"] = "all",
) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for annotation in annotations:
        if bucket != "all" and pii_type_bucket(annotation.pii_type) != bucket:
            continue
        for offset in range(annotation.start, annotation.end):
            positions.add((annotation.record_id, offset))
    return positions


def char_positions_for_predictions(
    predictions: Sequence[NumericPrediction],
    *,
    bucket: Literal["all", "direct", "quasi"] = "all",
) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for prediction in predictions:
        if bucket != "all" and pii_type_bucket(prediction.pii_type) != bucket:
            continue
        for offset in range(prediction.start, prediction.end):
            positions.add((prediction.record_id, offset))
    return positions


def build_character_counts(
    annotations: Sequence[NumericAnnotation],
    predictions: Sequence[NumericPrediction],
    *,
    bucket: Literal["all", "direct", "quasi"] = "all",
) -> CharacterCounts:
    gold_positions: set[tuple[int, int]] = char_positions_for_annotations(
        annotations,
        bucket=bucket,
    )
    predicted_positions: set[tuple[int, int]] = char_positions_for_predictions(
        predictions,
        bucket=bucket,
    )
    return CharacterCounts(
        tp=len(gold_positions & predicted_positions),
        fp=len(predicted_positions - gold_positions),
        fn=len(gold_positions - predicted_positions),
    )


def build_character_report(
    annotations: Sequence[NumericAnnotation],
    predictions: Sequence[NumericPrediction],
) -> dict[str, object]:
    all_counts: CharacterCounts = build_character_counts(annotations, predictions)
    direct_counts: CharacterCounts = build_character_counts(
        annotations,
        predictions,
        bucket="direct",
    )
    quasi_counts: CharacterCounts = build_character_counts(
        annotations,
        predictions,
        bucket="quasi",
    )
    return {
        "all": character_counts_payload(all_counts),
        "direct": character_counts_payload(direct_counts),
        "quasi": character_counts_payload(quasi_counts),
    }


def build_type_confusion_matrix(
    annotations: Sequence[NumericAnnotation],
    predictions: Sequence[NumericPrediction],
) -> dict[str, dict[str, int]]:
    pairs: tuple[SpanPair, ...] = match_span_pairs(
        annotations,
        predictions,
        require_type=False,
    )
    matched_annotations: set[tuple[int, str, int, int, str]] = {
        annotation_identity(pair.annotation) for pair in pairs
    }
    matched_predictions: set[tuple[int, str, int, int, str]] = {
        prediction_identity(pair.prediction) for pair in pairs
    }
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pair in pairs:
        matrix[pair.annotation.pii_type][pair.prediction.pii_type] += 1
    for annotation in annotations:
        if annotation_identity(annotation) not in matched_annotations:
            matrix[annotation.pii_type]["<MISS>"] += 1
    for prediction in predictions:
        if prediction_identity(prediction) not in matched_predictions:
            matrix["<SPURIOUS>"][prediction.pii_type] += 1
    return {
        gold_type: dict(sorted(predicted_types.items()))
        for gold_type, predicted_types in sorted(matrix.items())
    }


def annotation_has_residual_text_hit(annotation: NumericAnnotation, sanitized_text: str) -> bool:
    normalized_output: str = normalize_whitespace(sanitized_text).lower()
    raw_probe: str = normalize_whitespace(annotation.raw_text).lower()
    if len(raw_probe) >= 4 and raw_probe in normalized_output:
        return True

    digit_probe: str = annotation.canonical_value or annotation.digit_sequence
    if len(digit_probe) < 4:
        return False
    return digit_probe in digits_only(sanitized_text)


def build_privacy_output_report(
    evaluations: Sequence[EvaluatedRecord],
    relaxed_pairs: Sequence[SpanPair],
) -> dict[str, object]:
    all_annotations: list[NumericAnnotation] = [
        annotation for evaluation in evaluations for annotation in evaluation.annotations
    ]
    all_predictions: list[NumericPrediction] = [
        prediction for evaluation in evaluations for prediction in evaluation.predictions
    ]
    covered_annotations: set[tuple[int, str, int, int, str]] = {
        annotation_identity(pair.annotation) for pair in relaxed_pairs
    }

    mention_totals_by_bucket: dict[str, int] = {"direct": 0, "quasi": 0, "other": 0}
    missed_mentions_by_bucket: dict[str, int] = {"direct": 0, "quasi": 0, "other": 0}
    for annotation in all_annotations:
        bucket: Literal["direct", "quasi", "other"] = pii_type_bucket(annotation.pii_type)
        mention_totals_by_bucket[bucket] += 1
        if annotation_identity(annotation) not in covered_annotations:
            missed_mentions_by_bucket[bucket] += 1

    entity_totals_by_bucket: dict[str, int] = {"direct": 0, "quasi": 0, "other": 0}
    complete_entities_by_bucket: dict[str, int] = {"direct": 0, "quasi": 0, "other": 0}
    entity_annotations: dict[tuple[int, tuple[str, str]], list[NumericAnnotation]] = defaultdict(list)
    for annotation in all_annotations:
        entity_annotations[(annotation.record_id, annotation_key(annotation))].append(annotation)

    for (_record_id, key), grouped_annotations in entity_annotations.items():
        pii_type: str = key[0]
        bucket = pii_type_bucket(pii_type)
        entity_totals_by_bucket[bucket] += 1
        entity_is_complete: bool = all(
            annotation_identity(annotation) in covered_annotations
            for annotation in grouped_annotations
        )
        if entity_is_complete:
            complete_entities_by_bucket[bucket] += 1

    record_ids: set[int] = {int(str(evaluation.source["id"])) for evaluation in evaluations}
    direct_missed_record_ids: set[int] = {
        annotation.record_id
        for annotation in all_annotations
        if pii_type_bucket(annotation.pii_type) == "direct"
        and annotation_identity(annotation) not in covered_annotations
    }
    document_total: int = len(record_ids)
    document_pass_count: int = document_total - len(direct_missed_record_ids)

    residual_hits_by_type: dict[str, int] = defaultdict(int)
    residual_hit_count: int = 0
    for evaluation in evaluations:
        for annotation in evaluation.annotations:
            if annotation_has_residual_text_hit(annotation, evaluation.masked_original_text):
                residual_hit_count += 1
                residual_hits_by_type[annotation.pii_type] += 1

    total_original_chars: int = sum(len(str(evaluation.source.get("value", ""))) for evaluation in evaluations)
    predicted_char_positions: set[tuple[int, int]] = char_positions_for_predictions(all_predictions)
    char_counts: CharacterCounts = build_character_counts(all_annotations, all_predictions)
    masked_character_rate: float = (
        0.0 if total_original_chars == 0 else len(predicted_char_positions) / total_original_chars
    )
    overmasking_character_rate: float = (
        0.0 if total_original_chars == 0 else char_counts.fp / total_original_chars
    )

    return {
        "direct_identifier_leakage_rate": _safe_rate(
            missed_mentions_by_bucket["direct"],
            mention_totals_by_bucket["direct"],
        ),
        "quasi_identifier_leakage_rate": _safe_rate(
            missed_mentions_by_bucket["quasi"],
            mention_totals_by_bucket["quasi"],
        ),
        "missed_mentions": missed_mentions_by_bucket,
        "mention_totals": mention_totals_by_bucket,
        "entity_complete_masking": {
            bucket: {
                "complete": complete_entities_by_bucket[bucket],
                "total": entity_totals_by_bucket[bucket],
                "rate": _safe_rate(
                    complete_entities_by_bucket[bucket],
                    entity_totals_by_bucket[bucket],
                ),
            }
            for bucket in ("direct", "quasi", "other")
        },
        "document_level_privacy_pass_rate": _safe_rate(document_pass_count, document_total),
        "document_pass_count": document_pass_count,
        "document_total": document_total,
        "residual_raw_text_hit_count": residual_hit_count,
        "residual_raw_text_hits_by_type": dict(sorted(residual_hits_by_type.items())),
        "masked_character_rate": masked_character_rate,
        "overmasking_character_rate": overmasking_character_rate,
        "utility_preservation_score": max(0.0, 1.0 - overmasking_character_rate),
    }


def build_alignment_projection_report(
    predictions: Sequence[NumericPrediction],
    relaxed_pairs: Sequence[SpanPair],
) -> dict[str, object]:
    projection_failure_count: int = sum(1 for prediction in predictions if prediction.start >= prediction.end)
    expansion_ratios: list[float] = []
    original_lengths: list[int] = []
    normalized_lengths: list[int] = []
    for prediction in predictions:
        original_length: int = max(0, prediction.end - prediction.start)
        normalized_length: int = max(0, prediction.normalized_end - prediction.normalized_start)
        original_lengths.append(original_length)
        normalized_lengths.append(normalized_length)
        if normalized_length > 0:
            expansion_ratios.append(original_length / normalized_length)

    mean_expansion_ratio: float = (
        0.0 if not expansion_ratios else sum(expansion_ratios) / len(expansion_ratios)
    )
    mean_original_length: float = (
        0.0 if not original_lengths else sum(original_lengths) / len(original_lengths)
    )
    mean_normalized_length: float = (
        0.0 if not normalized_lengths else sum(normalized_lengths) / len(normalized_lengths)
    )
    return {
        "projection_failure_count": projection_failure_count,
        "projection_failure_rate": _safe_rate(projection_failure_count, len(predictions)),
        "mean_original_span_length": mean_original_length,
        "mean_normalized_span_length": mean_normalized_length,
        "mean_original_to_normalized_span_ratio": mean_expansion_ratio,
        "boundary_errors": build_boundary_error_report(relaxed_pairs),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def as_mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value)


def as_float(value: object) -> float:
    return float(str(value))


def as_int(value: object) -> int:
    return int(str(value))


def metric_counts_from_report(
    payload: Mapping[str, object],
    *,
    fallback_negative_fp: int = 0,
) -> MetricCounts:
    counts: Mapping[str, object] = as_mapping(payload["counts"])
    negative_fp_value: object = counts.get("negative_fp", fallback_negative_fp)
    return MetricCounts(
        tp=as_int(counts["tp"]),
        fp=as_int(counts["fp"]),
        fn=as_int(counts["fn"]),
        tn=as_int(counts["tn"]),
        negative_fp=as_int(negative_fp_value),
    )


def per_type_counts_from_report(payload: Mapping[str, object]) -> dict[str, tuple[int, int, int]]:
    per_type: Mapping[str, object] = as_mapping(payload["per_type"])
    counts_by_type: dict[str, tuple[int, int, int]] = {}
    for pii_type, raw_counts in per_type.items():
        counts: Mapping[str, object] = as_mapping(raw_counts)
        counts_by_type[pii_type] = (
            as_int(counts["tp"]),
            as_int(counts["fp"]),
            as_int(counts["fn"]),
        )
    return counts_by_type


def load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line: str = line.strip()
            if not stripped_line:
                continue
            record: dict[str, object] = json.loads(stripped_line)
            record.setdefault("id", len(records) + 1)
            if "value" not in record and "source_text" in record:
                record["value"] = record["source_text"]
            if "annotations" not in record and "entities" in record:
                record["annotations"] = record["entities"]
            records.append(record)
    return records


def evaluate_records(
    records: list[dict[str, object]],
    *,
    deduplicate_repetitions: bool = False,
    normalize_document_numbers: bool = True,
) -> list[EvaluatedRecord]:
    evaluations: list[EvaluatedRecord] = []
    for record in records:
        annotations: list[NumericAnnotation] = load_annotations(record)
        predictions: list[NumericPrediction]
        preprocessed_text: str
        masked_text: str
        masked_original_text: str
        repetition_suppressed_indexes: tuple[int, ...]
        (
            predictions,
            preprocessed_text,
            masked_text,
            masked_original_text,
            repetition_suppressed_indexes,
        ) = load_predictions(
            record,
            deduplicate_repetitions=deduplicate_repetitions,
            normalize_document_numbers=normalize_document_numbers,
        )
        evaluations.append(
            EvaluatedRecord(
                source=record,
                annotations=tuple(annotations),
                predictions=tuple(predictions),
                preprocessed_text=preprocessed_text,
                masked_text=masked_text,
                masked_original_text=masked_original_text,
                repetition_suppressed_indexes=repetition_suppressed_indexes,
            )
        )
    return evaluations


def build_report(
    evaluations: list[EvaluatedRecord],
    soft_gap: int,
    *,
    normalize_document_numbers: bool = True,
) -> dict[str, object]:
    all_annotations: list[NumericAnnotation] = []
    all_predictions: list[NumericPrediction] = []
    soft_gt_keys: list[tuple[str, str]] = []
    soft_pred_keys: list[tuple[str, str]] = []

    for evaluation in evaluations:
        annotations: tuple[NumericAnnotation, ...] = evaluation.annotations
        predictions: tuple[NumericPrediction, ...] = evaluation.predictions
        all_annotations.extend(annotations)
        all_predictions.extend(predictions)
        soft_gt_keys.extend(generate_soft_keys(annotations, max_gap=soft_gap))
        soft_pred_keys.extend({prediction_key(prediction) for prediction in predictions})

    records: list[dict[str, object]] = [evaluation.source for evaluation in evaluations]
    hard_negative_total, hard_negative_false_positives, hard_negative_by_category = (
        evaluate_hard_negatives(
            records,
            normalize_document_numbers=normalize_document_numbers,
        )
    )

    hard_tp, hard_fp, hard_fn, hard_per_type = multiset_match_counts(
        gt_keys=(annotation_key(annotation) for annotation in all_annotations),
        pred_keys=(prediction_key(prediction) for prediction in all_predictions),
    )
    soft_tp, soft_fp, soft_fn, soft_per_type = multiset_match_counts(
        gt_keys=soft_gt_keys,
        pred_keys=soft_pred_keys,
    )

    hard_counts = build_metric_counts(
        tp=hard_tp,
        fp=hard_fp,
        fn=hard_fn,
        hard_negative_total=hard_negative_total,
        hard_negative_false_positives=hard_negative_false_positives,
    )
    soft_counts = build_metric_counts(
        tp=soft_tp,
        fp=soft_fp,
        fn=soft_fn,
        hard_negative_total=hard_negative_total,
        hard_negative_false_positives=hard_negative_false_positives,
    )
    exact_span_report: dict[str, object] = build_exact_span_report(
        all_annotations,
        all_predictions,
        hard_negative_total=hard_negative_total,
        hard_negative_false_positives=hard_negative_false_positives,
    )
    relaxed_span_report: dict[str, object]
    relaxed_span_pairs: tuple[SpanPair, ...]
    relaxed_span_report, relaxed_span_pairs = build_relaxed_span_report(
        all_annotations,
        all_predictions,
        hard_negative_total=hard_negative_total,
        hard_negative_false_positives=hard_negative_false_positives,
    )
    character_report: dict[str, object] = build_character_report(all_annotations, all_predictions)
    privacy_output_report: dict[str, object] = build_privacy_output_report(
        evaluations,
        relaxed_span_pairs,
    )
    alignment_projection_report: dict[str, object] = build_alignment_projection_report(
        all_predictions,
        relaxed_span_pairs,
    )
    type_confusion_matrix: dict[str, dict[str, int]] = build_type_confusion_matrix(
        all_annotations,
        all_predictions,
    )

    return {
        "records": len(evaluations),
        "hard_negatives": {
            "total": hard_negative_total,
            "false_positives": hard_negative_false_positives,
            "false_positive_rate": _safe_rate(
                hard_negative_false_positives,
                hard_negative_total,
            ),
            "specificity": 0.0
            if hard_negative_total == 0
            else (hard_negative_total - hard_negative_false_positives) / hard_negative_total,
            "false_positives_by_category": hard_negative_by_category,
        },
        "hard": {
            "counts": {
                "tp": hard_counts.tp,
                "fp": hard_counts.fp,
                "fn": hard_counts.fn,
                "tn": hard_counts.tn,
            },
            "metrics": {
                "precision": hard_counts.precision,
                "recall": hard_counts.recall,
                "f1": hard_counts.f1,
                "accuracy": hard_counts.accuracy,
                "specificity": hard_counts.specificity,
            },
            "references": len(all_annotations),
            "predictions": len(all_predictions),
            "per_type": {
                pii_type: {
                    "tp": counts[0],
                    "fp": counts[1],
                    "fn": counts[2],
                }
                for pii_type, counts in sorted(hard_per_type.items())
            },
        },
        "soft": {
            "counts": {
                "tp": soft_counts.tp,
                "fp": soft_counts.fp,
                "fn": soft_counts.fn,
                "tn": soft_counts.tn,
            },
            "metrics": {
                "precision": soft_counts.precision,
                "recall": soft_counts.recall,
                "f1": soft_counts.f1,
                "accuracy": soft_counts.accuracy,
                "specificity": soft_counts.specificity,
            },
            "references": len(soft_gt_keys),
            "predictions": len(soft_pred_keys),
            "per_type": {
                pii_type: {
                    "tp": counts[0],
                    "fp": counts[1],
                    "fn": counts[2],
                }
                for pii_type, counts in sorted(soft_per_type.items())
            },
        },
        "span": {
            "strict_exact": exact_span_report,
            "relaxed_overlap": relaxed_span_report,
        },
        "character": character_report,
        "privacy_output": privacy_output_report,
        "alignment_projection": alignment_projection_report,
        "type_confusion_matrix": type_confusion_matrix,
    }


def build_artifact_dir(base_dir: Path) -> Path:
    timestamp: datetime = datetime.now()
    date_dir: Path = base_dir / timestamp.strftime("%Y-%m-%d")
    run_dir: Path = date_dir / timestamp.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_artifacts(
    artifact_dir: Path,
    dataset_path: Path,
    evaluations: list[EvaluatedRecord],
    report: dict[str, object],
    *,
    deduplicate_repetitions: bool,
    normalize_document_numbers: bool,
) -> None:
    preprocessed_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for evaluation in evaluations:
        source: dict[str, object] = evaluation.source
        record_id: int = as_int(source["id"])
        preprocessed_rows.append(
            {
                "id": record_id,
                "domain": source.get("domain"),
                "split": source.get("split"),
                "original_text": source.get("value"),
                "preprocessed_text": evaluation.preprocessed_text,
                "repetition_suppressed_indexes": list(
                    evaluation.repetition_suppressed_indexes
                ),
            }
        )
        model_rows.append(
            {
                "id": record_id,
                "domain": source.get("domain"),
                "split": source.get("split"),
                "preprocessed_text": evaluation.preprocessed_text,
                "masked_text": evaluation.masked_text,
                "masked_original_text": evaluation.masked_original_text,
                "repetition_suppressed_indexes": list(
                    evaluation.repetition_suppressed_indexes
                ),
                "matches": [
                    {
                        "type": prediction.pii_type,
                        "start": prediction.start,
                        "end": prediction.end,
                        "normalized_start": prediction.normalized_start,
                        "normalized_end": prediction.normalized_end,
                        "raw_value": prediction.raw_value,
                        "canonical_value": prediction.canonical_value,
                        "confidence": prediction.confidence,
                        "rule_id": prediction.rule_id,
                        "entity_id": prediction.entity_id,
                        "mention_id": prediction.mention_id,
                    }
                    for prediction in evaluation.predictions
                ],
            }
        )

    write_jsonl(artifact_dir / "dataset_after_preprocessing.jsonl", preprocessed_rows)
    write_jsonl(artifact_dir / "dataset_after_model.jsonl", model_rows)

    metadata: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "records": len(evaluations),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deduplicate_repetitions": deduplicate_repetitions,
        "normalize_document_numbers": normalize_document_numbers,
        "files": {
            "preprocessed_dataset": "dataset_after_preprocessing.jsonl",
            "model_output_dataset": "dataset_after_model.jsonl",
            "metrics_report": "metrics.json",
        },
    }
    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate numeric PII extraction on a JSONL dataset with hard and soft metrics. "
            "Hard metrics are mention-level multiset matches. Soft metrics deduplicate per record "
            "and additionally merge nearby numeric fragments into canonical entities."
        )
    )
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default="gt_asr.jsonl",
        help="Path to the source JSONL dataset.",
    )
    parser.add_argument(
        "--soft-gap",
        type=int,
        default=12,
        help="Maximum gap in characters between same-type annotations when building soft entity groups.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Base directory for run artifacts. A run directory is created as artifacts/YYYY-MM-DD/HH-MM-SS.",
    )
    parser.add_argument(
        "--deduplicate-repetitions",
        action="store_true",
        help=(
            "Enable the ASR repetition deduplication stage before numeric PII extraction. "
            "This is off by default because it changes the evaluated text layer."
        ),
    )
    parser.add_argument(
        "--no-document-number-normalization",
        action="store_true",
        help=(
            "Disable the document-number post-normalization stage. It is enabled by default "
            "for evaluation because repeated spoken IDs often need splitting before detection."
        ),
    )
    args = parser.parse_args()

    normalize_document_numbers: bool = not args.no_document_number_normalization
    dataset_path = Path(args.jsonl_path)
    records = load_records(dataset_path)
    evaluations = evaluate_records(
        records,
        deduplicate_repetitions=args.deduplicate_repetitions,
        normalize_document_numbers=normalize_document_numbers,
    )
    report = build_report(
        evaluations,
        soft_gap=args.soft_gap,
        normalize_document_numbers=normalize_document_numbers,
    )
    artifact_dir = build_artifact_dir(Path(args.artifacts_root))
    write_artifacts(
        artifact_dir,
        dataset_path,
        evaluations,
        report,
        deduplicate_repetitions=args.deduplicate_repetitions,
        normalize_document_numbers=normalize_document_numbers,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    hard_negatives_report: Mapping[str, object] = as_mapping(report["hard_negatives"])
    hard_report: Mapping[str, object] = as_mapping(report["hard"])
    soft_report: Mapping[str, object] = as_mapping(report["soft"])
    span_report: Mapping[str, object] = as_mapping(report["span"])
    strict_span_report: Mapping[str, object] = as_mapping(span_report["strict_exact"])
    relaxed_span_report: Mapping[str, object] = as_mapping(span_report["relaxed_overlap"])
    character_report: Mapping[str, object] = as_mapping(report["character"])
    privacy_output: Mapping[str, object] = as_mapping(report["privacy_output"])
    alignment_projection: Mapping[str, object] = as_mapping(report["alignment_projection"])
    hard_negative_false_positives: int = as_int(hard_negatives_report["false_positives"])

    print(f"dataset: {dataset_path}")
    print(f"artifacts: {artifact_dir}")
    print(f"records: {report['records']}")
    print(f"deduplicate repetitions: {args.deduplicate_repetitions}")
    print(f"normalize document numbers: {normalize_document_numbers}")
    print(
        "hard negatives: "
        f"{hard_negatives_report['total']} total, "
        f"{hard_negative_false_positives} false positives, "
        f"specificity={format_ratio(as_float(hard_negatives_report['specificity']))}"
    )
    print()

    hard_counts: MetricCounts = metric_counts_from_report(
        hard_report,
        fallback_negative_fp=hard_negative_false_positives,
    )
    soft_counts: MetricCounts = metric_counts_from_report(
        soft_report,
        fallback_negative_fp=hard_negative_false_positives,
    )
    print_metric_block("hard", hard_counts)
    print()
    print_metric_block("soft", soft_counts)
    print()

    strict_span_counts: MetricCounts = metric_counts_from_report(strict_span_report)
    relaxed_span_counts: MetricCounts = metric_counts_from_report(relaxed_span_report)
    print_metric_block("strict span", strict_span_counts)
    print()
    print_metric_block("relaxed span", relaxed_span_counts)
    print()

    print("character metrics")
    for bucket in ("all", "direct", "quasi"):
        bucket_report: Mapping[str, object] = as_mapping(character_report[bucket])
        bucket_metrics: Mapping[str, object] = as_mapping(bucket_report["metrics"])
        print(
            f"  {bucket:<6} precision={format_ratio(as_float(bucket_metrics['precision']))} "
            f"recall={format_ratio(as_float(bucket_metrics['recall']))} "
            f"f1={format_ratio(as_float(bucket_metrics['f1']))}"
        )
    print()

    entity_complete: Mapping[str, object] = as_mapping(privacy_output["entity_complete_masking"])
    direct_entity_complete: Mapping[str, object] = as_mapping(entity_complete["direct"])
    print("privacy output")
    print(
        "  direct leakage rate: "
        f"{format_ratio(as_float(privacy_output['direct_identifier_leakage_rate']))}"
    )
    print(
        "  quasi leakage rate:  "
        f"{format_ratio(as_float(privacy_output['quasi_identifier_leakage_rate']))}"
    )
    print(
        "  entity-complete direct: "
        f"{format_ratio(as_float(direct_entity_complete['rate']))}"
    )
    print(
        "  document pass rate: "
        f"{format_ratio(as_float(privacy_output['document_level_privacy_pass_rate']))}"
    )
    print(
        "  overmasking char rate: "
        f"{format_ratio(as_float(privacy_output['overmasking_character_rate']))}"
    )
    print(
        "  residual raw text hits: "
        f"{as_int(privacy_output['residual_raw_text_hit_count'])}"
    )
    print()

    boundary_errors: Mapping[str, object] = as_mapping(alignment_projection["boundary_errors"])
    print("alignment projection")
    print(
        "  projection failures: "
        f"{alignment_projection['projection_failure_count']} "
        f"({format_ratio(as_float(alignment_projection['projection_failure_rate']))})"
    )
    print(
        "  mean original/normalized span ratio: "
        f"{format_ratio(as_float(alignment_projection['mean_original_to_normalized_span_ratio']))}"
    )
    print(
        "  mean boundary error chars: "
        f"{format_ratio(as_float(boundary_errors['mean_boundary_error_chars']))}"
    )
    print()

    print_per_type_table(
        "hard per type",
        per_type_counts_from_report(hard_report),
        hard_negative_total=as_int(hard_negatives_report["total"]),
        hard_negative_false_positives=hard_negative_false_positives,
    )
    print()
    print_per_type_table(
        "soft per type",
        per_type_counts_from_report(soft_report),
        hard_negative_total=as_int(hard_negatives_report["total"]),
        hard_negative_false_positives=hard_negative_false_positives,
    )
    print()
    print("hard negative false positives by category")
    hard_negative_by_category: Mapping[str, object] = as_mapping(
        hard_negatives_report["false_positives_by_category"]
    )
    for category, count in hard_negative_by_category.items():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
