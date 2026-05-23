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
from typing import Iterable

from anonmed.anonymization import normalize_numeric_pii_value, run_numeric_pii_pipeline
from anonmed.preprocessing.asr.number_extractor import IntegerExtractor
from anonmed.preprocessing.asr.number_parser import parse_numeric_tokens
from anonmed.preprocessing.asr.tokenization import tokenize_preserving_spans
from anonmed.preprocessing.asr.types import ExtractorConfig, NumericToken, Token

NUMERIC_TYPE_MAP: dict[str, str] = {
    "ТЕЛЕФОН": "PHONE",
    "СНИЛС": "SNILS",
    "ПАСПОРТ": "PASSPORT",
    "ДАТА_РОЖДЕНИЯ": "DATE_BIRTH",
    "ОМС": "OMS",
    "ИНН": "INN",
    "ВОЗРАСТ": "AGE",
    "МСЭ": "MSE",
    "СВИДЕТЕЛЬСТВО": "BIRTH_CERTIFICATE",
    "ВУ": "DRIVER_LICENSE",
}

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
    raw_value: str
    canonical_value: str


@dataclass(frozen=True, slots=True)
class EvaluatedRecord:
    source: dict[str, object]
    annotations: tuple[NumericAnnotation, ...]
    predictions: tuple[NumericPrediction, ...]
    preprocessed_text: str
    masked_text: str
    repetition_suppressed_indexes: tuple[int, ...] = ()


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
        canonical_value = normalize_numeric_pii_value(pii_type, candidate)
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
    record_id: int = int(record["id"])
    for annotation in record.get("annotations", []):
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
                start=int(annotation_dict["start"]),
                end=int(annotation_dict["end"]),
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
) -> tuple[list[NumericPrediction], str, str, tuple[int, ...]]:
    record_id: int = int(record["id"])
    text: str = str(record["value"])
    predictions: list[NumericPrediction] = []
    pipeline_result = run_numeric_pii_pipeline(
        text,
        deduplicate_repetitions=deduplicate_repetitions,
    )
    for match in pipeline_result.matches:
        predictions.append(
            NumericPrediction(
                record_id=record_id,
                pii_type=match.pii_type,
                start=match.start,
                end=match.end,
                raw_value=match.value,
                canonical_value=match.normalized_value,
            )
        )
    return (
        predictions,
        pipeline_result.preprocessing_result.normalized_text,
        pipeline_result.masked_text,
        pipeline_result.preprocessing_result.repetition_suppressed_indexes,
    )


def cluster_annotations(
    annotations: list[NumericAnnotation],
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
    annotations: list[NumericAnnotation],
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
                    normalized_value: str | None = normalize_numeric_pii_value(pii_type, concatenated)
                    if normalized_value is not None:
                        keys.add((pii_type, normalized_value))
    return keys


def evaluate_hard_negatives(records: list[dict[str, object]]) -> tuple[int, int, dict[str, int]]:
    total: int = 0
    false_positives: int = 0
    per_category_fp: dict[str, int] = defaultdict(int)
    for record in records:
        for hard_negative in record.get("hard_negatives", []):
            hard_negative_dict: dict[str, object] = dict(hard_negative)
            raw_text: str = str(hard_negative_dict["text"])
            category: str = str(hard_negative_dict["category"])
            total += 1
            if hard_negative_has_match(raw_text):
                false_positives += 1
                per_category_fp[category] += 1
    return total, false_positives, dict(sorted(per_category_fp.items()))


@lru_cache(maxsize=None)
def hard_negative_has_match(raw_text: str) -> bool:
    return bool(run_numeric_pii_pipeline(raw_text).matches)


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
    for pii_type in sorted(NUMERIC_TYPE_MAP.values()):
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


def load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line: str = line.strip()
            if not stripped_line:
                continue
            records.append(json.loads(stripped_line))
    return records


def evaluate_records(
    records: list[dict[str, object]],
    *,
    deduplicate_repetitions: bool = False,
) -> list[EvaluatedRecord]:
    evaluations: list[EvaluatedRecord] = []
    for record in records:
        annotations: list[NumericAnnotation] = load_annotations(record)
        predictions: list[NumericPrediction]
        preprocessed_text: str
        masked_text: str
        repetition_suppressed_indexes: tuple[int, ...]
        (
            predictions,
            preprocessed_text,
            masked_text,
            repetition_suppressed_indexes,
        ) = load_predictions(
            record,
            deduplicate_repetitions=deduplicate_repetitions,
        )
        evaluations.append(
            EvaluatedRecord(
                source=record,
                annotations=tuple(annotations),
                predictions=tuple(predictions),
                preprocessed_text=preprocessed_text,
                masked_text=masked_text,
                repetition_suppressed_indexes=repetition_suppressed_indexes,
            )
        )
    return evaluations


def build_report(evaluations: list[EvaluatedRecord], soft_gap: int) -> dict[str, object]:
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
    hard_negative_total, hard_negative_false_positives, hard_negative_by_category = evaluate_hard_negatives(records)

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

    return {
        "records": len(evaluations),
        "hard_negatives": {
            "total": hard_negative_total,
            "false_positives": hard_negative_false_positives,
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
) -> None:
    preprocessed_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for evaluation in evaluations:
        source: dict[str, object] = evaluation.source
        record_id: int = int(source["id"])
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
                "repetition_suppressed_indexes": list(
                    evaluation.repetition_suppressed_indexes
                ),
                "matches": [
                    {
                        "type": prediction.pii_type,
                        "start": prediction.start,
                        "end": prediction.end,
                        "raw_value": prediction.raw_value,
                        "canonical_value": prediction.canonical_value,
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
    args = parser.parse_args()

    dataset_path = Path(args.jsonl_path)
    records = load_records(dataset_path)
    evaluations = evaluate_records(
        records,
        deduplicate_repetitions=args.deduplicate_repetitions,
    )
    report = build_report(evaluations, soft_gap=args.soft_gap)
    artifact_dir = build_artifact_dir(Path(args.artifacts_root))
    write_artifacts(
        artifact_dir,
        dataset_path,
        evaluations,
        report,
        deduplicate_repetitions=args.deduplicate_repetitions,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"dataset: {dataset_path}")
    print(f"artifacts: {artifact_dir}")
    print(f"records: {report['records']}")
    print(f"deduplicate repetitions: {args.deduplicate_repetitions}")
    print(
        "hard negatives: "
        f"{report['hard_negatives']['total']} total, "
        f"{report['hard_negatives']['false_positives']} false positives, "
        f"specificity={format_ratio(float(report['hard_negatives']['specificity']))}"
    )
    print()

    hard_counts = MetricCounts(
        **report["hard"]["counts"],
        negative_fp=int(report["hard_negatives"]["false_positives"]),
    )
    soft_counts = MetricCounts(
        **report["soft"]["counts"],
        negative_fp=int(report["hard_negatives"]["false_positives"]),
    )
    print_metric_block("hard", hard_counts)
    print()
    print_metric_block("soft", soft_counts)
    print()
    print_per_type_table(
        "hard per type",
        {
            pii_type: (counts["tp"], counts["fp"], counts["fn"])
            for pii_type, counts in report["hard"]["per_type"].items()
        },
        hard_negative_total=int(report["hard_negatives"]["total"]),
        hard_negative_false_positives=int(report["hard_negatives"]["false_positives"]),
    )
    print()
    print_per_type_table(
        "soft per type",
        {
            pii_type: (counts["tp"], counts["fp"], counts["fn"])
            for pii_type, counts in report["soft"]["per_type"].items()
        },
        hard_negative_total=int(report["hard_negatives"]["total"]),
        hard_negative_false_positives=int(report["hard_negatives"]["false_positives"]),
    )
    print()
    print("hard negative false positives by category")
    for category, count in report["hard_negatives"]["false_positives_by_category"].items():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
