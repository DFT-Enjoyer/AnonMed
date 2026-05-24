from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from anonmed.ml.core.types import Case, Span
from anonmed.ml.data.in_the_wild_datasets import InTheWildComprehensivePIIDataset
from anonmed.ml.models.PIDR import DEFAULT_FINE_TUNED_MODEL_PATH, DEFAULT_MODEL_PATH


PathLike = str | Path
Representation = Literal["digits", "letters"]
_TokenFeatureValue = int | list[int]

DEFAULT_ENTITY_LABELS: tuple[str, ...] = (
    "account_number",
    "bank_card_number",
    "birthdate",
    "city",
    "driver_license",
    "email",
    "first_name",
    "full_address",
    "full_name",
    "house",
    "inn",
    "last_name",
    "nickname",
    "passport",
    "password",
    "phone",
    "snils",
    "street",
    "zipcode",
)
DEFAULT_LABEL_SCHEMA: tuple[str, ...] = (
    "O",
    *(
        label
        for entity_label in DEFAULT_ENTITY_LABELS
        for label in (f"B-{entity_label}", f"I-{entity_label}")
    ),
)
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = ("Wqkv", "Wo")
DEFAULT_LORA_MODULES_TO_SAVE: tuple[str, ...] = ("classifier",)
DEFAULT_REPORT_TO: tuple[str, ...] = ("none",)


@dataclass(frozen=True, slots=True)
class PIDRTrainingSettings:
    base_model_path: PathLike = DEFAULT_MODEL_PATH
    output_dir: PathLike = DEFAULT_FINE_TUNED_MODEL_PATH
    dataset_root: PathLike | None = None
    representation: Representation | None = "letters"
    sample_size: int | None = None
    eval_fraction: float = 0.1
    random_seed: int = 42
    max_length: int = 512
    num_train_epochs: float = 3.0
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 25
    save_total_limit: int = 2
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    lora_modules_to_save: tuple[str, ...] = DEFAULT_LORA_MODULES_TO_SAVE
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    resume_from_checkpoint: str | None = None
    local_files_only: bool = True
    report_to: tuple[str, ...] = DEFAULT_REPORT_TO

    def __post_init__(self) -> None:
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive or None, got {self.sample_size}")
        if not 0.0 < self.eval_fraction < 1.0:
            raise ValueError(f"eval_fraction must be in (0, 1), got {self.eval_fraction}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.per_device_eval_batch_size <= 0:
            raise ValueError("per_device_eval_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive or None")
        if self.max_eval_samples is not None and self.max_eval_samples <= 0:
            raise ValueError("max_eval_samples must be positive or None")


@dataclass(frozen=True, slots=True)
class PIDRTrainingResult:
    output_dir: str
    train_samples: int
    eval_samples: int
    labels: tuple[str, ...]
    train_metrics: Mapping[str, float]
    eval_metrics: Mapping[str, float]


def select_training_spans(spans: Sequence[Span]) -> tuple[Span, ...]:
    sorted_spans: list[Span] = sorted(
        spans,
        key=lambda span: (
            -(span.end - span.begin),
            span.begin,
            span.end,
            span.label,
        ),
    )
    selected: list[Span] = []
    occupied: list[tuple[int, int]] = []
    for span in sorted_spans:
        overlaps: bool = any(span.begin < end and begin < span.end for begin, end in occupied)
        if overlaps:
            continue
        selected.append(span)
        occupied.append((span.begin, span.end))
    return tuple(sorted(selected, key=lambda span: (span.begin, span.end, span.label)))


def case_to_token_features(
    case: Case,
    tokenizer: Any,
    label2id: Mapping[str, int],
    *,
    max_length: int,
) -> dict[str, _TokenFeatureValue]:
    line = case.document.lines[0]
    spans: tuple[Span, ...] = select_training_spans(case.target.lines[0].spans)
    encoded: Mapping[str, Any] = tokenizer(
        line.text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offset_mapping: Sequence[Sequence[int]] = _offset_mapping_from_encoded(encoded)
    labels: list[int] = _token_labels(
        offsets=offset_mapping,
        spans=spans,
        label2id=label2id,
    )
    features: dict[str, _TokenFeatureValue] = {}
    for key, value in encoded.items():
        if key == "offset_mapping":
            continue
        features[key] = _list_or_int(value)
    features["labels"] = labels
    return features


def compute_token_metrics(
    predictions: npt.NDArray[np.float64] | npt.NDArray[np.float32] | npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
    label_schema: Sequence[str] = DEFAULT_LABEL_SCHEMA,
) -> dict[str, float]:
    predicted_ids: npt.NDArray[np.int64]
    if predictions.ndim == labels.ndim + 1:
        predicted_ids = np.asarray(np.argmax(predictions, axis=-1), dtype=np.int64)
    else:
        predicted_ids = np.asarray(predictions, dtype=np.int64)

    label_ids: npt.NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
    outside_id: int = int(_label2id(label_schema)["O"])
    active_mask: npt.NDArray[np.bool_] = label_ids != -100
    entity_label_mask: npt.NDArray[np.bool_] = np.logical_and(active_mask, label_ids != outside_id)
    entity_pred_mask: npt.NDArray[np.bool_] = np.logical_and(active_mask, predicted_ids != outside_id)
    correct_entity_mask: npt.NDArray[np.bool_] = np.logical_and(
        entity_label_mask,
        predicted_ids == label_ids,
    )

    tp: int = int(np.sum(correct_entity_mask))
    fp: int = int(np.sum(np.logical_and(entity_pred_mask, predicted_ids != label_ids)))
    fn: int = int(np.sum(np.logical_and(entity_label_mask, predicted_ids != label_ids)))
    active_total: int = int(np.sum(active_mask))
    correct_total: int = int(np.sum(np.logical_and(active_mask, predicted_ids == label_ids)))
    precision_value: float = _safe_div(tp, tp + fp)
    recall_value: float = _safe_div(tp, tp + fn)
    f1_value: float = _safe_div(2.0 * precision_value * recall_value, precision_value + recall_value)
    return {
        "token_precision": precision_value,
        "token_recall": recall_value,
        "token_f1": f1_value,
        "token_accuracy": _safe_div(correct_total, active_total),
        "token_tp": float(tp),
        "token_fp": float(fp),
        "token_fn": float(fn),
    }


def train_pidr(settings: PIDRTrainingSettings | None = None) -> PIDRTrainingResult:
    resolved_settings: PIDRTrainingSettings = settings or PIDRTrainingSettings()
    transformers: Any = _import_transformers()

    tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
        str(resolved_settings.base_model_path),
        local_files_only=resolved_settings.local_files_only,
    )
    label_schema: tuple[str, ...] = DEFAULT_LABEL_SCHEMA
    label2id: dict[str, int] = _label2id(label_schema)
    model: Any = _load_model_for_training(
        transformers=transformers,
        settings=resolved_settings,
        label_schema=label_schema,
        label2id=label2id,
    )
    if resolved_settings.use_lora:
        model = _apply_lora(model, resolved_settings)

    cases: tuple[Case, ...] = _load_training_cases(resolved_settings)
    train_cases, eval_cases = _split_cases(
        cases,
        eval_fraction=resolved_settings.eval_fraction,
        random_seed=resolved_settings.random_seed,
    )
    train_cases = _limit_cases(train_cases, resolved_settings.max_train_samples)
    eval_cases = _limit_cases(eval_cases, resolved_settings.max_eval_samples)
    train_dataset: Any = _features_dataset(
        train_cases,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=resolved_settings.max_length,
    )
    eval_dataset: Any = _features_dataset(
        eval_cases,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=resolved_settings.max_length,
    )

    data_collator: Any = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args: Any = transformers.TrainingArguments(
        **_training_arguments_kwargs(transformers, resolved_settings)
    )

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        return compute_token_metrics(
            np.asarray(eval_prediction.predictions),
            np.asarray(eval_prediction.label_ids),
            label_schema,
        )

    trainer: Any = transformers.Trainer(
        **_trainer_kwargs(
            transformers=transformers,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    )
    train_output: Any = trainer.train(
        resume_from_checkpoint=resolved_settings.resume_from_checkpoint
    )
    eval_metrics: Mapping[str, float] = trainer.evaluate()
    output_dir: Path = Path(resolved_settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _write_training_metadata(
        output_dir=output_dir,
        settings=resolved_settings,
        label_schema=label_schema,
        train_samples=len(train_cases),
        eval_samples=len(eval_cases),
        train_metrics=dict(getattr(train_output, "metrics", {})),
        eval_metrics=dict(eval_metrics),
    )
    return PIDRTrainingResult(
        output_dir=str(output_dir),
        train_samples=len(train_cases),
        eval_samples=len(eval_cases),
        labels=label_schema,
        train_metrics=dict(getattr(train_output, "metrics", {})),
        eval_metrics=dict(eval_metrics),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the local PIDR ModernBERT token-classifier on "
            "in_the_wild_russian_pii_speech."
        )
    )
    parser.add_argument("--base-model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_FINE_TUNED_MODEL_PATH))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--representation", choices=("digits", "letters"), default="letters")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="Wqkv,Wo")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print result as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    settings: PIDRTrainingSettings = PIDRTrainingSettings(
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        representation=args.representation,
        sample_size=args.sample_size,
        eval_fraction=args.eval_fraction,
        random_seed=args.seed,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        use_lora=not bool(args.no_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_comma_tuple(args.lora_target_modules),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
        local_files_only=not bool(args.allow_remote_files),
    )
    result: PIDRTrainingResult = train_pidr(settings)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"saved model: {result.output_dir}")
    return 0


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:
        message: str = (
            "PIDR training requires transformers, torch, accelerate and safetensors. "
            "Install the ML dependencies before running this module."
        )
        raise ImportError(message) from error
    return transformers


def _load_model_for_training(
    *,
    transformers: Any,
    settings: PIDRTrainingSettings,
    label_schema: Sequence[str],
    label2id: Mapping[str, int],
) -> Any:
    config: Any = transformers.AutoConfig.from_pretrained(
        str(settings.base_model_path),
        local_files_only=settings.local_files_only,
    )
    config.id2label = {index: label for index, label in enumerate(label_schema)}
    config.label2id = dict(label2id)
    config.num_labels = len(label_schema)
    model: Any = transformers.AutoModelForTokenClassification.from_pretrained(
        str(settings.base_model_path),
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=settings.local_files_only,
    )
    return model


def _apply_lora(model: Any, settings: PIDRTrainingSettings) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        message: str = "LoRA training requires the 'peft' package. Use --no-lora or install peft."
        raise ImportError(message) from error

    lora_config = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=list(settings.lora_target_modules),
        modules_to_save=list(settings.lora_modules_to_save),
        task_type=TaskType.TOKEN_CLS,
    )
    return get_peft_model(model, lora_config)


def _load_training_cases(settings: PIDRTrainingSettings) -> tuple[Case, ...]:
    dataset = InTheWildComprehensivePIIDataset(
        root=settings.dataset_root,
        representation=settings.representation,
        sample_size=settings.sample_size,
    )
    return dataset.cases


def _split_cases(
    cases: Sequence[Case],
    *,
    eval_fraction: float,
    random_seed: int,
) -> tuple[tuple[Case, ...], tuple[Case, ...]]:
    if len(cases) < 2:
        raise ValueError("At least two cases are required to create train/eval splits.")
    shuffled_cases: list[Case] = list(cases)
    random.Random(random_seed).shuffle(shuffled_cases)
    eval_size: int = max(1, round(len(shuffled_cases) * eval_fraction))
    if eval_size >= len(shuffled_cases):
        eval_size = len(shuffled_cases) - 1
    eval_cases: tuple[Case, ...] = tuple(shuffled_cases[:eval_size])
    train_cases: tuple[Case, ...] = tuple(shuffled_cases[eval_size:])
    return train_cases, eval_cases


def _limit_cases(cases: tuple[Case, ...], limit: int | None) -> tuple[Case, ...]:
    if limit is None:
        return cases
    return cases[:limit]


def _features_dataset(
    cases: Sequence[Case],
    *,
    tokenizer: Any,
    label2id: Mapping[str, int],
    max_length: int,
) -> Any:
    try:
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise ImportError("PIDR training requires the 'datasets' package.") from error

    features: list[dict[str, _TokenFeatureValue]] = [
        case_to_token_features(
            case,
            tokenizer,
            label2id,
            max_length=max_length,
        )
        for case in cases
    ]
    return Dataset.from_list(features)


def _training_arguments_kwargs(
    transformers: Any,
    settings: PIDRTrainingSettings,
) -> dict[str, Any]:
    import inspect

    signature: inspect.Signature = inspect.signature(transformers.TrainingArguments)
    parameters: set[str] = set(signature.parameters)
    kwargs: dict[str, Any] = {
        "output_dir": str(settings.output_dir),
        "learning_rate": settings.learning_rate,
        "per_device_train_batch_size": settings.per_device_train_batch_size,
        "per_device_eval_batch_size": settings.per_device_eval_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "num_train_epochs": settings.num_train_epochs,
        "weight_decay": settings.weight_decay,
        "warmup_ratio": settings.warmup_ratio,
        "logging_steps": settings.logging_steps,
        "save_total_limit": settings.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "token_f1",
        "greater_is_better": True,
        "report_to": list(settings.report_to),
        "seed": settings.random_seed,
    }
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"
    if "save_strategy" in parameters:
        kwargs["save_strategy"] = "epoch"
    return {key: value for key, value in kwargs.items() if key in parameters}


def _trainer_kwargs(
    *,
    transformers: Any,
    model: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
    compute_metrics: Any,
) -> dict[str, Any]:
    import inspect

    signature: inspect.Signature = inspect.signature(transformers.Trainer)
    parameters: set[str] = set(signature.parameters)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    return {key: value for key, value in kwargs.items() if key in parameters}


def _write_training_metadata(
    *,
    output_dir: Path,
    settings: PIDRTrainingSettings,
    label_schema: Sequence[str],
    train_samples: int,
    eval_samples: int,
    train_metrics: Mapping[str, float],
    eval_metrics: Mapping[str, float],
) -> None:
    label_schema_path: Path = output_dir / "label_schema.json"
    metadata_path: Path = output_dir / "anonmed_training.json"
    label_schema_path.write_text(
        json.dumps({"labels": list(label_schema)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "settings": asdict(settings),
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "train_metrics": dict(train_metrics),
        "eval_metrics": dict(eval_metrics),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _token_labels(
    *,
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Span],
    label2id: Mapping[str, int],
) -> list[int]:
    labels: list[int] = []
    started_spans: set[tuple[int, int, str]] = set()
    outside_id: int = int(label2id["O"])
    for offset in offsets:
        if len(offset) != 2:
            labels.append(-100)
            continue
        begin: int = int(offset[0])
        end: int = int(offset[1])
        if begin >= end:
            labels.append(-100)
            continue
        span: Span | None = _span_for_token(begin=begin, end=end, spans=spans)
        if span is None:
            labels.append(outside_id)
            continue
        span_key: tuple[int, int, str] = (span.begin, span.end, span.label)
        prefix: str = "I" if span_key in started_spans else "B"
        started_spans.add(span_key)
        labels.append(int(label2id.get(f"{prefix}-{span.label}", outside_id)))
    return labels


def _span_for_token(*, begin: int, end: int, spans: Sequence[Span]) -> Span | None:
    for span in spans:
        if span.begin <= begin and end <= span.end:
            return span
    return None


def _offset_mapping_from_encoded(encoded: Mapping[str, Any]) -> Sequence[Sequence[int]]:
    raw_offsets: Any = encoded.get("offset_mapping")
    if not isinstance(raw_offsets, Sequence) or isinstance(raw_offsets, (str, bytes, bytearray)):
        raise TypeError("Tokenizer output must include offset_mapping as a sequence.")
    return raw_offsets


def _list_or_int(value: Any) -> _TokenFeatureValue:
    if isinstance(value, int):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [int(item) for item in value]
    return int(value)


def _label2id(label_schema: Sequence[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(label_schema)}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _comma_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = [
    "DEFAULT_ENTITY_LABELS",
    "DEFAULT_LABEL_SCHEMA",
    "DEFAULT_LORA_MODULES_TO_SAVE",
    "DEFAULT_LORA_TARGET_MODULES",
    "DEFAULT_REPORT_TO",
    "PIDRTrainingResult",
    "PIDRTrainingSettings",
    "PathLike",
    "Representation",
    "build_parser",
    "case_to_token_features",
    "compute_token_metrics",
    "main",
    "select_training_spans",
    "train_pidr",
]
