from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Span, TextDocument
from anonmed.ml.models.base import PIIModel


_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH: Path = _REPOSITORY_ROOT / "models" / "PIDR"
DEFAULT_FINE_TUNED_MODEL_PATH: Path = (
    _REPOSITORY_ROOT / "models" / "PIDR_finetuned_in_the_wild_russian_pii_speech"
)
DEFAULT_MAX_LENGTH: int = 1024

DEFAULT_PIDR_LABEL_MAPPING: dict[str, str] = {
    "IPI-BODY_DESC": "body_description",
    "IPI-DETAILS": "details",
    "IPI-FAMILY": "family",
    "IPI-HEALTH_FCLT": "healthcare_facility",
    "IPI-RELATV_TIME": "relative_time",
    "IPI-SOCIO": "socio",
    "PHI-AGE": "age",
    "PHI-CONTACT_PHONE": "phone",
    "PHI-DATE": "birthdate",
    "PHI-ID": "id",
    "PHI-LOCATION_CITY": "city",
    "PHI-LOCATION_HOSPITAL": "hospital",
    "PHI-LOCATION_STREET": "street",
    "PHI-LOCATION_ZIP": "zipcode",
    "PHI-NAME": "full_name",
}


@dataclass(frozen=True, slots=True)
class _SpanDraft:
    begin: int
    end: int
    label: str
    scores: tuple[float, ...]


class PIDRModel(PIIModel):
    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        base_model_path: str | Path | None = None,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        min_score: float | None = None,
        device: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        output_label_mapping: Mapping[str, str] | None = None,
        local_files_only: bool = True,
    ) -> None:
        if max_length is not None and max_length <= 0:
            raise ValueError(f"max_length must be positive or None, got {max_length}")
        if min_score is not None and not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score}")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be provided together or omitted together.")

        self.model_path: str = str(model_path)
        self.base_model_path: str | None = None if base_model_path is None else str(base_model_path)
        self.max_length: int | None = max_length
        self.min_score: float | None = min_score
        self.device: str | None = _resolve_device(device)
        self.output_label_mapping: dict[str, str] = dict(
            DEFAULT_PIDR_LABEL_MAPPING if output_label_mapping is None else output_label_mapping
        )

        if tokenizer is not None and model is not None:
            self.tokenizer: Any = tokenizer
            self.model: Any = model
        else:
            loaded_model, loaded_tokenizer = _load_pidr_components(
                self.model_path,
                base_model_path=self.base_model_path,
                tokenizer_kwargs=tokenizer_kwargs or {},
                model_kwargs=model_kwargs or {},
                local_files_only=local_files_only,
            )
            self.model = loaded_model
            self.tokenizer = loaded_tokenizer

        if self.device is not None and hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.id2label: dict[int, str] = _id2label_from_model(self.model)

    def predict(self, document: TextDocument) -> AnnotationSet:
        annotation_lines: list[AnnotationSetLine] = []
        for line in document.lines:
            spans: list[Span] = self._predict_line(line.text, line.idx)
            annotation_line: AnnotationSetLine = AnnotationSetLine(
                idx=line.idx,
                role=line.role,
                spans=spans,
            )
            annotation_lines.append(annotation_line)
        return AnnotationSet(lines=tuple(annotation_lines), idx=document.sample_id)

    def _predict_line(self, text: str, line_idx: int) -> list[Span]:
        if text == "":
            return []

        tokenizer_kwargs: dict[str, Any] = {
            "return_offsets_mapping": True,
            "return_tensors": "pt",
            "truncation": self.max_length is not None,
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length

        encoded: Mapping[str, Any] = self.tokenizer(text, **tokenizer_kwargs)
        offsets: list[tuple[int, int]] = _offsets_from_encoded(encoded)
        model_inputs: dict[str, Any] = _model_inputs(encoded, device=self.device)

        with _no_grad_context():
            outputs: Any = self.model(**model_inputs)

        logits: Any = outputs.logits
        prediction_ids: list[int] = _first_int_row(_argmax_last_dim(logits))
        token_scores: list[float | None] = _selected_token_scores(
            logits,
            prediction_ids,
            enabled=self.min_score is not None,
        )
        return _spans_from_predictions(
            text=text,
            line_idx=line_idx,
            offsets=offsets,
            prediction_ids=prediction_ids,
            token_scores=token_scores,
            id2label=self.id2label,
            output_label_mapping=self.output_label_mapping,
            min_score=self.min_score,
        )


class FineTunedPIDRModel(PIDRModel):
    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_FINE_TUNED_MODEL_PATH,
        output_label_mapping: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_label_mapping: Mapping[str, str] = (
            {} if output_label_mapping is None else output_label_mapping
        )
        super().__init__(
            model_path=model_path,
            output_label_mapping=resolved_label_mapping,
            **kwargs,
        )


def _load_pidr_components(
    model_path: str,
    *,
    base_model_path: str | None,
    tokenizer_kwargs: Mapping[str, Any],
    model_kwargs: Mapping[str, Any],
    local_files_only: bool,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as error:
        message: str = (
            "PIDRModel requires 'transformers' and 'torch'. "
            "Install the ML extras or pass preloaded tokenizer and model objects."
        )
        raise ImportError(message) from error

    resolved_tokenizer_kwargs: dict[str, Any] = dict(tokenizer_kwargs)
    resolved_tokenizer_kwargs.setdefault("local_files_only", local_files_only)
    tokenizer_source: str = base_model_path or model_path
    tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_source, **resolved_tokenizer_kwargs)

    resolved_model_kwargs: dict[str, Any] = dict(model_kwargs)
    resolved_model_kwargs.setdefault("local_files_only", local_files_only)
    if _is_peft_adapter_dir(Path(model_path)):
        model: Any = _load_peft_token_classifier(
            model_path,
            base_model_path=base_model_path,
            model_kwargs=resolved_model_kwargs,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            **resolved_model_kwargs,
        )
    return model, tokenizer


def _load_peft_token_classifier(
    model_path: str,
    *,
    base_model_path: str | None,
    model_kwargs: Mapping[str, Any],
) -> Any:
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoConfig, AutoModelForTokenClassification
    except ImportError as error:
        message: str = (
            "Loading a PEFT fine-tuned PIDR model requires 'peft', "
            "'transformers', and 'torch'."
        )
        raise ImportError(message) from error

    peft_config: Any = PeftConfig.from_pretrained(model_path)
    resolved_base_model_path: str = base_model_path or str(peft_config.base_model_name_or_path)
    resolved_model_kwargs: dict[str, Any] = dict(model_kwargs)
    label_schema: tuple[str, ...] = _read_label_schema(Path(model_path))

    config: Any = AutoConfig.from_pretrained(
        resolved_base_model_path,
        local_files_only=bool(resolved_model_kwargs.get("local_files_only", True)),
    )
    if label_schema:
        config.id2label = {index: label for index, label in enumerate(label_schema)}
        config.label2id = {label: index for index, label in enumerate(label_schema)}
        config.num_labels = len(label_schema)

    base_model: Any = AutoModelForTokenClassification.from_pretrained(
        resolved_base_model_path,
        config=config,
        ignore_mismatched_sizes=bool(label_schema),
        **resolved_model_kwargs,
    )
    return PeftModel.from_pretrained(base_model, model_path)


def _is_peft_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def _read_label_schema(model_path: Path) -> tuple[str, ...]:
    schema_path: Path = model_path / "label_schema.json"
    if not schema_path.exists():
        return ()
    import json

    payload: Any = json.loads(schema_path.read_text(encoding="utf-8"))
    labels: Any = payload.get("labels") if isinstance(payload, Mapping) else None
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray)):
        return ()
    return tuple(str(label) for label in labels)


def _resolve_device(device: str | None) -> str | None:
    if device is not None:
        return device
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    mps_backend: Any = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _id2label_from_model(model: Any) -> dict[int, str]:
    config: Any = getattr(model, "config", None)
    raw_id2label: Any = getattr(config, "id2label", None)
    if isinstance(raw_id2label, Mapping):
        labels: dict[int, str] = {}
        for key, value in raw_id2label.items():
            labels[int(key)] = str(value)
        return labels

    num_labels: int = int(getattr(config, "num_labels", 0) or 0)
    return {index: str(index) for index in range(num_labels)}


def _offsets_from_encoded(encoded: Mapping[str, Any]) -> list[tuple[int, int]]:
    offset_mapping: Any = encoded.get("offset_mapping")
    first_row: Any = _first_row(offset_mapping)
    offsets: list[tuple[int, int]] = []
    for raw_offset in first_row:
        values: Sequence[Any] = _as_sequence(raw_offset)
        if len(values) != 2:
            offsets.append((0, 0))
            continue
        offsets.append((int(values[0]), int(values[1])))
    return offsets


def _model_inputs(encoded: Mapping[str, Any], *, device: str | None) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for key, value in encoded.items():
        if key == "offset_mapping":
            continue
        if device is not None and hasattr(value, "to"):
            inputs[key] = value.to(device)
        else:
            inputs[key] = value
    return inputs


def _argmax_last_dim(logits: Any) -> Any:
    try:
        return logits.argmax(dim=-1)
    except TypeError:
        return logits.argmax(-1)


def _selected_token_scores(
    logits: Any,
    prediction_ids: Sequence[int],
    *,
    enabled: bool,
) -> list[float | None]:
    if not enabled:
        return [None for _ in prediction_ids]
    try:
        probabilities: Any = logits.softmax(dim=-1)
        first_row: Any = _first_row(probabilities)
        return [float(first_row[index][label_id]) for index, label_id in enumerate(prediction_ids)]
    except (AttributeError, IndexError, TypeError, ValueError):
        return [None for _ in prediction_ids]


def _first_int_row(value: Any) -> list[int]:
    first_row: Any = _first_row(value)
    return [int(item) for item in _as_sequence(first_row)]


def _first_row(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    sequence: Sequence[Any] = _as_sequence(value)
    if not sequence:
        return []
    return sequence[0]


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"Expected sequence-like value, got {type(value).__name__}")


def _spans_from_predictions(
    *,
    text: str,
    line_idx: int,
    offsets: Sequence[tuple[int, int]],
    prediction_ids: Sequence[int],
    token_scores: Sequence[float | None],
    id2label: Mapping[int, str],
    output_label_mapping: Mapping[str, str],
    min_score: float | None,
) -> list[Span]:
    spans: list[Span] = []
    current: _SpanDraft | None = None
    for index, label_id in enumerate(prediction_ids):
        begin, end = offsets[index] if index < len(offsets) else (0, 0)
        score: float | None = token_scores[index] if index < len(token_scores) else None
        parsed_label: tuple[str, str] | None = _parse_model_label(
            id2label.get(label_id, str(label_id)),
            output_label_mapping=output_label_mapping,
        )
        if begin >= end or parsed_label is None or _below_threshold(score, min_score):
            current = _flush_span(text, line_idx, current, spans)
            continue

        prefix, label = parsed_label
        if (
            current is None
            or prefix == "B"
            or current.label != label
            or _has_non_whitespace_gap(text, current.end, begin)
        ):
            current = _flush_span(text, line_idx, current, spans)
            current = _SpanDraft(
                begin=begin,
                end=end,
                label=label,
                scores=() if score is None else (score,),
            )
            continue

        merged_scores: tuple[float, ...] = current.scores
        if score is not None:
            merged_scores = (*current.scores, score)
        current = _SpanDraft(
            begin=current.begin,
            end=max(current.end, end),
            label=current.label,
            scores=merged_scores,
        )

    _flush_span(text, line_idx, current, spans)
    return spans


def _parse_model_label(
    raw_label: str,
    *,
    output_label_mapping: Mapping[str, str],
) -> tuple[str, str] | None:
    if raw_label == "O" or raw_label == "":
        return None
    if raw_label.startswith(("B-", "I-")):
        prefix: str = raw_label[:1]
        entity_label: str = raw_label[2:]
    else:
        prefix = "B"
        entity_label = raw_label

    mapped_label: str = output_label_mapping.get(entity_label, entity_label)
    if mapped_label == "":
        return None
    return prefix, mapped_label


def _below_threshold(score: float | None, min_score: float | None) -> bool:
    return min_score is not None and score is not None and score < min_score


def _has_non_whitespace_gap(text: str, previous_end: int, next_begin: int) -> bool:
    if next_begin <= previous_end:
        return False
    return text[previous_end:next_begin].strip() != ""


def _flush_span(
    text: str,
    line_idx: int,
    current: _SpanDraft | None,
    spans: list[Span],
) -> _SpanDraft | None:
    if current is None:
        return None
    if 0 <= current.begin < current.end <= len(text):
        span: Span = Span(
            line_idx=line_idx,
            begin=current.begin,
            end=current.end,
            label=current.label,
            data=text[current.begin:current.end],
        )
        spans.append(span)
    return None


def _no_grad_context() -> Any:
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


__all__: list[str] = [
    "DEFAULT_FINE_TUNED_MODEL_PATH",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PIDR_LABEL_MAPPING",
    "FineTunedPIDRModel",
    "PIDRModel",
]
