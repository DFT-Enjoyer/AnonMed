#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from anonmed import PIIAnonymizer  # noqa: E402
from anonmed.ml.core.types import (  # noqa: E402
    AnnotationSet,
    AnnotationSetLine,
    Case,
    Role,
    Span,
)
from anonmed.ml.data.in_the_wild_datasets import InTheWildComprehensivePIIDataset  # noqa: E402
from anonmed.ml.data.in_the_wild_datasets.base import (  # noqa: E402
    DEFAULT_IN_THE_WILD_DATASET_ROOT,
    Representation,
)
from anonmed.ml.metrics.utils import (  # noqa: E402
    Counts,
    aggregate_counts,
    f1,
    precision,
    recall,
    soft_char_counts,
)

DEFAULT_LIMIT: int = 1000
DEFAULT_SOFT_IOU_THRESHOLD: float = 0.5

LabelMapper = Callable[[str], str]

_DATASET_LABEL_MAP: Mapping[str, str] = {
    "fio": "PER",
    "first_name": "PER",
    "full_name": "PER",
    "last_name": "PER",
    "middle_name": "PER",
    "name": "PER",
    "patronymic": "PER",
    "per": "PER",
    "person": "PER",
    "person_name": "PER",
    "address": "ADDRESS",
    "city": "ADDRESS",
    "country": "ADDRESS",
    "flat": "ADDRESS",
    "full_address": "ADDRESS",
    "house": "ADDRESS",
    "locality": "ADDRESS",
    "region": "ADDRESS",
    "settlement": "ADDRESS",
    "street": "ADDRESS",
    "phone": "PHONE",
    "telephone": "PHONE",
    "email": "EMAIL",
    "mail": "EMAIL",
    "telegram": "TELEGRAM",
    "username": "TELEGRAM",
    "nickname": "NICKNAME",
    "snils": "SNILS",
    "passport": "PASSPORT",
    "date_birth": "DATE_BIRTH",
    "birthdate": "DATE_BIRTH",
    "birth_date": "DATE_BIRTH",
    "oms": "OMS",
    "inn": "INN",
    "age": "AGE",
    "mse": "MSE",
    "birth_certificate": "BIRTH_CERTIFICATE",
    "driver_license": "DRIVER_LICENSE",
    "workplace": "WORKPLACE",
    "employer": "WORKPLACE",
    "company": "WORKPLACE",
    "organization": "WORKPLACE",
    "account_number": "ACCOUNT_NUMBER",
    "bank_account": "ACCOUNT_NUMBER",
}


@dataclass(frozen=True, slots=True)
class RunStats:
    samples: int
    failures: int
    gt_entities: int
    predicted_entities: int
    preprocessed_changed: int
    masked_original_changed: int
    warnings: int
    gt_by_type: Mapping[str, int]
    pred_by_type: Mapping[str, int]
    pred_by_source: Mapping[str, int]
    pred_by_rule: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class EvaluationOutput:
    dataset: str
    representation: str
    limit: int
    ml_model: str
    soft_iou_threshold: float
    metrics: Mapping[str, Mapping[str, float | int]]
    per_type: Mapping[str, Mapping[str, Mapping[str, float | int]]]
    stats: RunStats


def normalize_dataset_label(label: str) -> str:
    key: str = label.strip().lower()
    normalized: str = _DATASET_LABEL_MAP.get(key, key.upper())
    return normalized


def normalize_case_labels(case: Case, label_mapper: LabelMapper = normalize_dataset_label) -> Case:
    normalized_lines: list[AnnotationSetLine] = []
    for line in case.target.lines:
        normalized_spans: list[Span] = [
            Span(
                line_idx=span.line_idx,
                begin=span.begin,
                end=span.end,
                label=label_mapper(span.label),
                data=span.data,
            )
            for span in line.spans
        ]
        normalized_lines.append(
            AnnotationSetLine(idx=line.idx, role=line.role, spans=normalized_spans)
        )
    target = AnnotationSet(lines=tuple(normalized_lines), idx=case.target.idx)
    normalized_case = Case(document=case.document, target=target)
    return normalized_case


def load_dataset_cases(
    *,
    root: Path,
    representation: Representation | None,
    limit: int,
    strict_spans: bool,
) -> tuple[Case, ...]:
    dataset = InTheWildComprehensivePIIDataset(
        root=root,
        representation=representation,
        sample_size=limit,
        strict_spans=strict_spans,
    )
    cases: tuple[Case, ...] = tuple(
        normalize_case_labels(case) for case in dataset.cases
    )
    return cases


def prediction_from_result(case: Case, result: Any) -> AnnotationSet:
    role: Role = case.document.lines[0].role
    text_length: int = len(case.document.lines[0].text)
    spans: list[Span] = []
    for mention in result.postprocessed_mentions:
        begin: int = int(mention.original_start)
        end: int = int(mention.original_end)
        label: str = str(mention.entity_type)
        if begin < 0 or end <= begin or end > text_length:
            continue
        spans.append(
            Span(
                line_idx=0,
                begin=begin,
                end=end,
                label=label,
                data=case.document.lines[0].text[begin:end],
            )
        )
    line = AnnotationSetLine(idx=0, role=role, spans=spans)
    prediction = AnnotationSet(lines=(line,), idx=case.document.sample_id)
    return prediction


def counts_to_report(counts: Counts) -> dict[str, float | int]:
    return {
        "precision": precision(counts),
        "recall": recall(counts),
        "f1": f1(counts),
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
    }


def compute_metrics(
    cases: tuple[Case, ...],
    predictions: tuple[AnnotationSet, ...],
    *,
    soft_iou_threshold: float,
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for mode in ("entity_hard", "entity_soft", "char_hard", "char_soft"):
        counts: Counts = aggregate_counts(
            cases,
            predictions,
            mode=mode,
            entity_iou_threshold=soft_iou_threshold,
        )
        metrics[mode] = counts_to_report(counts)
    return metrics


def filter_annotation_by_label(annotation: AnnotationSet, label: str) -> AnnotationSet:
    lines: list[AnnotationSetLine] = []
    for line in annotation.lines:
        spans: list[Span] = [span for span in line.spans if span.label == label]
        lines.append(AnnotationSetLine(idx=line.idx, role=line.role, spans=spans))
    filtered = AnnotationSet(lines=tuple(lines), idx=annotation.idx)
    return filtered


def all_labels(cases: Sequence[Case], predictions: Sequence[AnnotationSet]) -> tuple[str, ...]:
    labels: set[str] = set()
    for case in cases:
        for line in case.target.lines:
            labels.update(span.label for span in line.spans)
    for prediction in predictions:
        for line in prediction.lines:
            labels.update(span.label for span in line.spans)
    return tuple(sorted(labels))


def compute_per_type_metrics(
    cases: tuple[Case, ...],
    predictions: tuple[AnnotationSet, ...],
    *,
    soft_iou_threshold: float,
) -> dict[str, dict[str, dict[str, float | int]]]:
    per_type: dict[str, dict[str, dict[str, float | int]]] = {}
    for label in all_labels(cases, predictions):
        label_cases: list[Case] = []
        label_predictions: list[AnnotationSet] = []
        for case, prediction in zip(cases, predictions, strict=True):
            target: AnnotationSet = filter_annotation_by_label(case.target, label)
            document_case = Case(document=case.document, target=target)
            label_cases.append(document_case)
            label_predictions.append(filter_annotation_by_label(prediction, label))

        typed_cases: tuple[Case, ...] = tuple(label_cases)
        typed_predictions: tuple[AnnotationSet, ...] = tuple(label_predictions)
        hard_counts: Counts = aggregate_counts(
            typed_cases,
            typed_predictions,
            mode="entity_hard",
        )
        soft_counts: Counts = aggregate_counts(
            typed_cases,
            typed_predictions,
            mode="entity_soft",
            entity_iou_threshold=soft_iou_threshold,
        )
        char_hard_counts: Counts = aggregate_counts(
            typed_cases,
            typed_predictions,
            mode="char_hard",
        )
        char_soft_total: Counts = _aggregate_typed_soft_char_counts(
            typed_cases,
            typed_predictions,
        )
        per_type[label] = {
            "entity_hard": counts_to_report(hard_counts),
            "entity_soft": counts_to_report(soft_counts),
            "char_hard": counts_to_report(char_hard_counts),
            "char_soft": counts_to_report(char_soft_total),
        }
    return per_type


def _aggregate_typed_soft_char_counts(
    cases: tuple[Case, ...],
    predictions: tuple[AnnotationSet, ...],
) -> Counts:
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    for case, prediction in zip(cases, predictions, strict=True):
        counts: Counts = soft_char_counts(prediction=prediction, target=case.target)
        total_tp += counts.tp
        total_fp += counts.fp
        total_fn += counts.fn
    return Counts(tp=total_tp, fp=total_fp, fn=total_fn)


def build_stats(
    cases: tuple[Case, ...],
    predictions: tuple[AnnotationSet, ...],
    results: Sequence[Any],
    failures: int,
) -> RunStats:
    gt_by_type: Counter[str] = Counter()
    pred_by_type: Counter[str] = Counter()
    pred_by_source: Counter[str] = Counter()
    pred_by_rule: Counter[str] = Counter()
    gt_entities: int = 0
    predicted_entities: int = 0
    preprocessed_changed: int = 0
    masked_original_changed: int = 0
    warnings_count: int = 0

    for case in cases:
        for line in case.target.lines:
            for span in line.spans:
                gt_by_type[span.label] += 1
                gt_entities += 1

    for prediction in predictions:
        for line in prediction.lines:
            for span in line.spans:
                pred_by_type[span.label] += 1
                predicted_entities += 1

    for result in results:
        if result.preprocessed_text != result.original_text:
            preprocessed_changed += 1
        if result.masked_original_text != result.original_text:
            masked_original_changed += 1
        warnings_count += len(result.warnings)
        for mention in result.postprocessed_mentions:
            pred_by_source[str(mention.source)] += 1
            pred_by_rule[str(mention.rule_id)] += 1

    stats = RunStats(
        samples=len(cases),
        failures=failures,
        gt_entities=gt_entities,
        predicted_entities=predicted_entities,
        preprocessed_changed=preprocessed_changed,
        masked_original_changed=masked_original_changed,
        warnings=warnings_count,
        gt_by_type=dict(sorted(gt_by_type.items())),
        pred_by_type=dict(sorted(pred_by_type.items())),
        pred_by_source=dict(sorted(pred_by_source.items())),
        pred_by_rule=dict(sorted(pred_by_rule.items())),
    )
    return stats


def evaluate_cases(
    cases: tuple[Case, ...],
    *,
    anonymizer: PIIAnonymizer,
    soft_iou_threshold: float,
    show_progress: bool,
) -> tuple[tuple[AnnotationSet, ...], EvaluationOutput]:
    predictions: list[AnnotationSet] = []
    results: list[Any] = []
    failures: int = 0
    iterator: Sequence[Case] | Any = cases
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(cases, total=len(cases), desc="Anonymizing", unit="sample")
        except ImportError:
            iterator = cases

    for case in iterator:
        text: str = case.document.lines[0].text
        try:
            result: Any = anonymizer(text, use_ml=True)
            prediction: AnnotationSet = prediction_from_result(case, result)
            results.append(result)
        except Exception as error:
            failures += 1
            print(
                f"warning: failed sample {case.document.sample_id}: {error}",
                file=sys.stderr,
            )
            prediction = AnnotationSet(
                lines=(AnnotationSetLine(idx=0, role=case.document.lines[0].role, spans=[]),),
                idx=case.document.sample_id,
            )
        predictions.append(prediction)

    prediction_tuple: tuple[AnnotationSet, ...] = tuple(predictions)
    metrics: dict[str, dict[str, float | int]] = compute_metrics(
        cases,
        prediction_tuple,
        soft_iou_threshold=soft_iou_threshold,
    )
    per_type: dict[str, dict[str, dict[str, float | int]]] = compute_per_type_metrics(
        cases,
        prediction_tuple,
        soft_iou_threshold=soft_iou_threshold,
    )
    stats: RunStats = build_stats(cases, prediction_tuple, results, failures)
    output = EvaluationOutput(
        dataset="in_the_wild_russian_pii_speech",
        representation="all",
        limit=len(cases),
        ml_model="PIDR_finetuned",
        soft_iou_threshold=soft_iou_threshold,
        metrics=metrics,
        per_type=per_type,
        stats=stats,
    )
    return prediction_tuple, output


def print_report(output: EvaluationOutput) -> None:
    print("In-the-wild dataset 1 PIIAnonymizer pipeline evaluation")
    print(f"samples: {output.stats.samples}")
    print(f"ml_model: {output.ml_model}")
    print(f"soft_iou_threshold: {output.soft_iou_threshold:.2f}")
    print(f"failures: {output.stats.failures}")
    print(f"gt_entities: {output.stats.gt_entities}")
    print(f"predicted_entities: {output.stats.predicted_entities}")
    print(f"preprocessed_changed: {output.stats.preprocessed_changed}")
    print(f"masked_original_changed: {output.stats.masked_original_changed}")
    print()
    print("overall metrics")
    _print_metric_table(output.metrics)
    print()
    print("per-type entity_soft metrics")
    _print_per_type_table(output.per_type, metric_name="entity_soft")
    print()
    print("gt by type")
    _print_counter(output.stats.gt_by_type)
    print()
    print("predictions by type")
    _print_counter(output.stats.pred_by_type)
    print()
    print("predictions by source")
    _print_counter(output.stats.pred_by_source)
    print()
    print("top prediction rules")
    _print_counter(output.stats.pred_by_rule, limit=20)


def _print_metric_table(metrics: Mapping[str, Mapping[str, float | int]]) -> None:
    print(f"{'metric':<14} {'precision':>10} {'recall':>10} {'f1':>10} {'tp':>7} {'fp':>7} {'fn':>7}")
    for name, values in metrics.items():
        print(
            f"{name:<14} "
            f"{float(values['precision']):>10.4f} "
            f"{float(values['recall']):>10.4f} "
            f"{float(values['f1']):>10.4f} "
            f"{int(values['tp']):>7} "
            f"{int(values['fp']):>7} "
            f"{int(values['fn']):>7}"
        )


def _print_per_type_table(
    per_type: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    *,
    metric_name: str,
) -> None:
    print(f"{'type':<20} {'precision':>10} {'recall':>10} {'f1':>10} {'tp':>7} {'fp':>7} {'fn':>7}")
    for label, metric_bucket in per_type.items():
        values: Mapping[str, float | int] = metric_bucket[metric_name]
        print(
            f"{label:<20} "
            f"{float(values['precision']):>10.4f} "
            f"{float(values['recall']):>10.4f} "
            f"{float(values['f1']):>10.4f} "
            f"{int(values['tp']):>7} "
            f"{int(values['fp']):>7} "
            f"{int(values['fn']):>7}"
        )


def _print_counter(counter: Mapping[str, int], *, limit: int | None = None) -> None:
    items: list[tuple[str, int]] = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        items = items[:limit]
    for key, value in items:
        print(f"  {key:<32} {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PIIAnonymizer(ml_model='PIDR_finetuned') on the first N samples of "
            "data/in_the_wild_datasets/final_version/2 and print metrics."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_IN_THE_WILD_DATASET_ROOT,
        help="Root containing final_version/* dataset folders.",
    )
    parser.add_argument(
        "--representation",
        choices=("digits", "letters", "all"),
        default="all",
        help="Dataset representation to evaluate.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--ml-model", default="natasha_per")
    parser.add_argument("--device", default=None)
    parser.add_argument("--soft-iou-threshold", type=float, default=DEFAULT_SOFT_IOU_THRESHOLD)
    parser.add_argument("--strict-spans", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace = parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    representation: Representation | None = (
        None if args.representation == "all" else args.representation
    )
    cases: tuple[Case, ...] = load_dataset_cases(
        root=args.root,
        representation=representation,
        limit=args.limit,
        strict_spans=args.strict_spans,
    )
    anonymizer = PIIAnonymizer(ml_model=args.ml_model, device=args.device)
    _predictions, output = evaluate_cases(
        cases,
        anonymizer=anonymizer,
        soft_iou_threshold=args.soft_iou_threshold,
        show_progress=not args.no_progress,
    )
    output = EvaluationOutput(
        dataset=output.dataset,
        representation=args.representation,
        limit=args.limit,
        ml_model=args.ml_model,
        soft_iou_threshold=output.soft_iou_threshold,
        metrics=output.metrics,
        per_type=output.per_type,
        stats=output.stats,
    )
    print_report(output)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(asdict(output), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\njson_report: {args.json_out}")
    return 0


__all__: tuple[str, ...] = (
    "DEFAULT_LIMIT",
    "EvaluationOutput",
    "RunStats",
    "build_stats",
    "compute_metrics",
    "compute_per_type_metrics",
    "evaluate_cases",
    "load_dataset_cases",
    "main",
    "normalize_case_labels",
    "normalize_dataset_label",
    "prediction_from_result",
)


if __name__ == "__main__":
    raise SystemExit(main())
