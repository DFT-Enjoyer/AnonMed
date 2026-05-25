from __future__ import annotations

from types import SimpleNamespace
import unittest
from typing import Any

import numpy as np

from anonmed.ml.config import ModelConfig
from anonmed.ml.core.types import (
    AnnotationSet,
    AnnotationSetLine,
    Case,
    Role,
    Span,
    TextDocument,
    TextLine,
)
from anonmed.ml.models.PIDR import FineTunedPIDRModel, PIDRModel
from anonmed.ml.registry import build_model
from anonmed.ml.train.pidr import (
    PIDRTrainingSettings,
    build_parser,
    case_to_token_features,
    compute_token_metrics,
    select_training_spans,
)


class _FakeLogits:
    def __init__(self, prediction_ids: list[int]) -> None:
        self.prediction_ids: list[int] = prediction_ids

    def argmax(self, dim: int | None = None) -> list[list[int]]:
        return [self.prediction_ids]


class _FakeModel:
    def __init__(self, id2label: dict[int, str], prediction_ids: list[int]) -> None:
        self.config = SimpleNamespace(id2label=id2label, num_labels=len(id2label))
        self.prediction_ids: list[int] = prediction_ids
        self.eval_called: bool = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=_FakeLogits(self.prediction_ids))


class _FakeInferenceTokenizer:
    def __init__(self, offsets: list[tuple[int, int]]) -> None:
        self.offsets: list[tuple[int, int]] = offsets
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[list[int | tuple[int, int]]]]:
        self.kwargs = kwargs
        token_count: int = len(self.offsets)
        return {
            "input_ids": [list(range(token_count))],
            "attention_mask": [[1 for _ in range(token_count)]],
            "offset_mapping": [self.offsets],
        }


class _FakeTrainingTokenizer:
    def __init__(self, offsets: list[tuple[int, int]]) -> None:
        self.offsets: list[tuple[int, int]] = offsets

    def __call__(
        self,
        text: str,
        *,
        return_offsets_mapping: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        token_count: int = min(len(self.offsets), max_length)
        return {
            "input_ids": list(range(token_count)),
            "attention_mask": [1 for _ in range(token_count)],
            "offset_mapping": self.offsets[:token_count],
        }


def _case_with_spans(text: str, spans: list[Span]) -> Case:
    role = Role(name="text")
    line = TextLine(idx=0, role=role, text=text)
    document = TextDocument(lines=(line,), sample_id="sample-1")
    target_line = AnnotationSetLine(idx=0, role=role, spans=spans)
    target = AnnotationSet(lines=(target_line,), idx="sample-1")
    return Case(document=document, target=target)


class PIDRModelAndTrainingTests(unittest.TestCase):
    def test_pidr_model_maps_original_labels_to_dataset_labels(self) -> None:
        text: str = "Пациент Иван Иванов пришел."
        tokenizer = _FakeInferenceTokenizer(
            offsets=[(0, 0), (0, 7), (8, 12), (13, 19), (20, 26), (26, 27)]
        )
        fake_model = _FakeModel(
            id2label={0: "O", 1: "B-PHI-NAME", 2: "I-PHI-NAME"},
            prediction_ids=[0, 0, 1, 2, 0, 0],
        )
        model = PIDRModel(tokenizer=tokenizer, model=fake_model, device="cpu")
        document = TextDocument(lines=(TextLine(idx=0, role=Role(name="text"), text=text),))

        prediction = model.predict(document)

        self.assertTrue(fake_model.eval_called)
        self.assertEqual(tokenizer.kwargs["return_tensors"], "pt")
        spans = prediction.lines[0].spans
        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0].begin, spans[0].end, spans[0].label), (8, 19, "full_name"))
        self.assertEqual(spans[0].data, "Иван Иванов")

    def test_fine_tuned_pidr_model_keeps_trained_dataset_labels(self) -> None:
        text: str = "Пациент Иван Иванов пришел."
        tokenizer = _FakeInferenceTokenizer(
            offsets=[(0, 0), (0, 7), (8, 12), (13, 19), (20, 26), (26, 27)]
        )
        fake_model = _FakeModel(
            id2label={0: "O", 1: "B-full_name", 2: "I-full_name"},
            prediction_ids=[0, 0, 1, 2, 0, 0],
        )
        model = FineTunedPIDRModel(tokenizer=tokenizer, model=fake_model, device="cpu")
        document = TextDocument(lines=(TextLine(idx=0, role=Role(name="text"), text=text),))

        prediction = model.predict(document)

        span = prediction.lines[0].spans[0]
        self.assertEqual((span.begin, span.end, span.label), (8, 19, "full_name"))

    def test_pidr_model_is_registered(self) -> None:
        tokenizer = _FakeInferenceTokenizer(offsets=[(0, 0), (0, 4)])
        fake_model = _FakeModel(id2label={0: "O", 1: "B-PHI-NAME"}, prediction_ids=[0, 1])

        model = build_model(
            ModelConfig(
                id="PIDR",
                params={"tokenizer": tokenizer, "model": fake_model, "device": "cpu"},
            )
        )

        self.assertIsInstance(model, PIDRModel)

    def test_fine_tuned_pidr_model_is_a_separate_registered_wrapper(self) -> None:
        tokenizer = _FakeInferenceTokenizer(offsets=[(0, 0), (0, 4)])
        fake_model = _FakeModel(id2label={0: "O", 1: "B-full_name"}, prediction_ids=[0, 1])

        model = build_model(
            ModelConfig(
                id="PIDR_finetuned",
                params={"tokenizer": tokenizer, "model": fake_model, "device": "cpu"},
            )
        )

        self.assertIsInstance(model, FineTunedPIDRModel)
        self.assertNotIsInstance(model, PIDRModel)

    def test_training_span_selection_prefers_longer_overlapping_entities(self) -> None:
        spans: list[Span] = [
            Span(line_idx=0, begin=0, end=4, label="first_name", data="Иван"),
            Span(line_idx=0, begin=5, end=11, label="last_name", data="Иванов"),
            Span(line_idx=0, begin=0, end=11, label="full_name", data="Иван Иванов"),
            Span(line_idx=0, begin=19, end=31, label="phone", data="79991234567"),
        ]

        selected = select_training_spans(spans)

        self.assertEqual(
            [(span.begin, span.end, span.label) for span in selected],
            [(0, 11, "full_name"), (19, 31, "phone")],
        )

    def test_case_to_token_features_aligns_bio_labels_by_offsets(self) -> None:
        text: str = "Иван Иванов пришел"
        case: Case = _case_with_spans(
            text,
            [
                Span(line_idx=0, begin=0, end=4, label="first_name", data="Иван"),
                Span(line_idx=0, begin=0, end=11, label="full_name", data="Иван Иванов"),
                Span(line_idx=0, begin=5, end=11, label="last_name", data="Иванов"),
            ],
        )
        tokenizer = _FakeTrainingTokenizer(offsets=[(0, 0), (0, 4), (5, 11), (12, 18)])
        label2id: dict[str, int] = {"O": 0, "B-full_name": 1, "I-full_name": 2}

        features = case_to_token_features(case, tokenizer, label2id, max_length=16)

        self.assertEqual(features["input_ids"], [0, 1, 2, 3])
        self.assertEqual(features["labels"], [-100, 1, 2, 0])

    def test_compute_token_metrics_ignores_special_tokens(self) -> None:
        predictions = np.asarray([[1, 0, 2, 0, 0]], dtype=np.int64)
        labels = np.asarray([[1, 2, 2, 0, -100]], dtype=np.int64)

        metrics = compute_token_metrics(predictions, labels, ("O", "B-name", "I-name"))

        self.assertEqual(metrics["token_tp"], 2.0)
        self.assertEqual(metrics["token_fp"], 0.0)
        self.assertEqual(metrics["token_fn"], 1.0)
        self.assertAlmostEqual(metrics["token_recall"], 2.0 / 3.0)

    def test_training_parser_builds_settings_without_running_training(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--sample-size",
                "12",
                "--representation",
                "digits",
                "--max-train-samples",
                "8",
                "--max-eval-samples",
                "4",
                "--no-lora",
            ]
        )
        settings = PIDRTrainingSettings(
            sample_size=args.sample_size,
            representation=args.representation,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            use_lora=not bool(args.no_lora),
        )

        self.assertEqual(settings.sample_size, 12)
        self.assertEqual(settings.representation, "digits")
        self.assertEqual(settings.max_train_samples, 8)
        self.assertEqual(settings.max_eval_samples, 4)
        self.assertFalse(settings.use_lora)


__all__: list[str] = []
