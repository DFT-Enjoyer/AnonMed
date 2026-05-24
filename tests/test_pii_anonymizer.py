from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import unittest

from anonmed import (
    MLDetectionConfig,
    PIIAnonymizationResult,
    PIIAnonymizer,
    PIIAnonymizerConfig,
    anonymize,
)
from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Span
from anonmed.ml.models.base import PIIModel
from anonmed.ml.pipelines.runner import ModelRunner


class FakeNameModel(PIIModel):
    def predict(self, document: Any) -> AnnotationSet:
        line: Any = document.lines[0]
        entity_text: str = "Иванов Иван Иванович"
        start: int = line.text.index(entity_text)
        span = Span(
            line_idx=line.idx,
            begin=start,
            end=start + len(entity_text),
            label="PER",
            data=entity_text,
        )
        return AnnotationSet(
            lines=(AnnotationSetLine(idx=line.idx, role=line.role, spans=[span]),),
            idx=document.sample_id,
        )


class FakeMultiEntityModel(PIIModel):
    def predict(self, document: Any) -> AnnotationSet:
        line: Any = document.lines[0]
        spans: list[Span] = []
        entities: tuple[tuple[str, str], ...] = (
            ("PER", "Иванов Иван Иванович"),
            ("ADDRESS", "Москва улица Ленина 10"),
        )
        for label, entity_text in entities:
            start: int = line.text.index(entity_text)
            span = Span(
                line_idx=line.idx,
                begin=start,
                end=start + len(entity_text),
                label=label,
                data=entity_text,
            )
            spans.append(span)
        return AnnotationSet(
            lines=(AnnotationSetLine(idx=line.idx, role=line.role, spans=spans),),
            idx=document.sample_id,
        )


class PIIAnonymizerTests(unittest.TestCase):
    def test_rule_only_anonymizer_masks_original_text(self) -> None:
        anonymizer = PIIAnonymizer()

        result: PIIAnonymizationResult = anonymizer(
            "ну, телефон: восемь девять один три один два три четыре пять шесть семь!",
            use_ml=False,
        )

        self.assertEqual(result.preprocessed_text, "телефон 89131234567")
        self.assertEqual(result.masked_text, "ну, телефон: [PHONE]!")
        self.assertEqual(result.masked_preprocessed_text, "телефон [PHONE]")
        self.assertEqual([candidate.entity_type for candidate in result.rule_candidates], ["PHONE"])
        self.assertEqual(result.ml_candidates, ())

    def test_convenience_anonymize_function_runs_once(self) -> None:
        result: PIIAnonymizationResult = anonymize("телефон 89131234567", use_ml=False)

        self.assertEqual(result.masked_text, "телефон [PHONE]")

    def test_constructor_default_flags_can_disable_rules(self) -> None:
        anonymizer = PIIAnonymizer(use_rules=False)

        result: PIIAnonymizationResult = anonymizer("телефон 89131234567")

        self.assertEqual(result.rule_candidates, ())
        self.assertEqual(result.masked_text, "телефон 89131234567")

    def test_constructor_does_not_allocate_lazy_runtime_resources(self) -> None:
        anonymizer = PIIAnonymizer(ml_model="example")

        self.assertIsNone(getattr(anonymizer, "_ml_runner"))
        self.assertEqual(getattr(anonymizer, "_preprocessing_pipelines"), {})

    def test_preprocessing_pipeline_is_cached_for_same_config(self) -> None:
        anonymizer = PIIAnonymizer()
        text: str = "телефон восемь девять один три один два три четыре пять шесть семь"

        anonymizer(text, use_ml=False)
        preprocessing_pipelines: dict[object, object] = getattr(
            anonymizer,
            "_preprocessing_pipelines",
        )
        first_pipeline: object = next(iter(preprocessing_pipelines.values()))

        anonymizer(text, use_ml=False)

        self.assertEqual(len(preprocessing_pipelines), 1)
        self.assertIs(next(iter(preprocessing_pipelines.values())), first_pipeline)

    def test_call_kwargs_override_default_config(self) -> None:
        anonymizer = PIIAnonymizer(
            ml_model=FakeNameModel(),
            default_config=PIIAnonymizerConfig(ml=MLDetectionConfig(enabled=True)),
        )

        result: PIIAnonymizationResult = anonymizer(
            "Пациент Иванов Иван Иванович пришел.",
            use_ml=False,
        )

        self.assertEqual(result.ml_candidates, ())
        self.assertEqual(result.masked_text, "Пациент Иванов Иван Иванович пришел.")
        self.assertFalse(result.config.ml.enabled)

    def test_ml_detection_uses_model_runner_spans(self) -> None:
        anonymizer = PIIAnonymizer(ml_model=FakeNameModel())

        result: PIIAnonymizationResult = anonymizer(
            "Пациент Иванов Иван Иванович пришел.",
            use_rules=False,
            ml_labels=("PER",),
        )

        self.assertEqual([candidate.entity_type for candidate in result.ml_candidates], ["PER"])
        self.assertEqual(result.masked_text, "Пациент [PER] пришел.")

    def test_ml_runner_is_reused_between_calls_with_same_model_params(self) -> None:
        anonymizer = PIIAnonymizer(ml_model=FakeNameModel())
        text: str = "Пациент Иванов Иван Иванович пришел."

        anonymizer(text, use_rules=False, ml_labels=("PER",))
        first_runner: object | None = getattr(anonymizer, "_ml_runner")

        anonymizer(text, use_rules=False, ml_labels=("PER",))

        self.assertIsNotNone(first_runner)
        self.assertIs(getattr(anonymizer, "_ml_runner"), first_runner)

    def test_ml_enabled_without_model_raises_value_error(self) -> None:
        anonymizer = PIIAnonymizer()

        with self.assertRaisesRegex(ValueError, "ml_model must be provided"):
            anonymizer("телефон 89131234567", use_ml=True, use_rules=False)

    def test_normalize_numbers_flag_controls_rule_recall(self) -> None:
        anonymizer = PIIAnonymizer()
        text: str = "телефон восемь девять один три один два три четыре пять шесть семь"

        disabled_result: PIIAnonymizationResult = anonymizer(
            text,
            use_ml=False,
            normalize_numbers=False,
        )
        enabled_result: PIIAnonymizationResult = anonymizer(
            text,
            use_ml=False,
            normalize_numbers=True,
        )

        self.assertEqual(disabled_result.rule_candidates, ())
        self.assertNotIn("[PHONE]", disabled_result.masked_text)
        self.assertEqual(
            [candidate.entity_type for candidate in enabled_result.rule_candidates],
            ["PHONE"],
        )
        self.assertEqual(enabled_result.masked_text, "телефон [PHONE]")

    def test_restore_non_pii_false_returns_preprocessed_masked_text(self) -> None:
        anonymizer = PIIAnonymizer()
        text: str = "ну, телефон: восемь девять один три один два три четыре пять шесть семь!"

        result: PIIAnonymizationResult = anonymizer(
            text,
            use_ml=False,
            restore_non_pii=False,
        )

        self.assertEqual(result.masked_text, "телефон [PHONE]")
        self.assertEqual(result.masked_preprocessed_text, "телефон [PHONE]")
        self.assertEqual(result.masked_original_text, "ну, телефон: [PHONE]!")

    def test_disabled_postprocessing_warns_when_original_layer_cannot_be_restored(self) -> None:
        anonymizer = PIIAnonymizer()
        text: str = "ну, телефон: восемь девять один три один два три четыре пять шесть семь!"

        result: PIIAnonymizationResult = anonymizer(
            text,
            use_ml=False,
            use_postprocessing=False,
        )

        self.assertEqual(result.masked_preprocessed_text, "телефон [PHONE]")
        self.assertEqual(result.masked_original_text, text)
        self.assertEqual(result.masked_text, "телефон [PHONE]")
        self.assertIsNone(result.postprocessing_result)
        self.assertTrue(result.warnings)
        self.assertIn("postprocessing is disabled", result.warnings[0])

    def test_pii_types_filter_rule_candidates(self) -> None:
        anonymizer = PIIAnonymizer()
        text: str = "телефон 89131234567 снилс 12345678900"

        result: PIIAnonymizationResult = anonymizer(
            text,
            use_ml=False,
            pii_types=("PHONE",),
        )

        self.assertEqual(
            [candidate.entity_type for candidate in result.rule_candidates],
            ["PHONE"],
        )
        self.assertEqual(result.masked_text, "телефон [PHONE] снилс 12345678900")

    def test_replacement_by_type_controls_postprocessing_replacement(self) -> None:
        anonymizer = PIIAnonymizer()

        result: PIIAnonymizationResult = anonymizer(
            "телефон 89131234567",
            use_ml=False,
            replacement_by_type={"PHONE": "[ENC_PHONE]"},
        )

        self.assertEqual(result.masked_text, "телефон [ENC_PHONE]")
        self.assertEqual(result.postprocessed_mentions[0].replacement, "[ENC_PHONE]")

    def test_same_length_masking_strategy_masks_original_span_length(self) -> None:
        anonymizer = PIIAnonymizer()

        result: PIIAnonymizationResult = anonymizer(
            "телефон 89131234567",
            use_ml=False,
            masking_strategy="same_length",
        )

        self.assertEqual(result.masked_text, "телефон ***********")
        self.assertEqual(result.masked_original_text, "телефон ***********")

    def test_ml_labels_filter_ml_candidates(self) -> None:
        anonymizer = PIIAnonymizer(ml_model=FakeMultiEntityModel())
        text: str = "Пациент Иванов Иван Иванович живет Москва улица Ленина 10"

        result: PIIAnonymizationResult = anonymizer(
            text,
            use_rules=False,
            ml_labels=("PER",),
        )

        self.assertEqual(
            [candidate.entity_type for candidate in result.ml_candidates],
            ["PER"],
        )
        self.assertEqual(
            result.masked_text,
            "Пациент [PER] живет Москва улица Ленина 10",
        )

    def test_stage_methods_can_be_called_separately(self) -> None:
        anonymizer = PIIAnonymizer()
        preprocessing_result: Any = anonymizer.preprocess(
            "телефон восемь девять один три один два три четыре пять шесть семь"
        )
        candidates: tuple[Any, ...] = anonymizer.detect_by_rules(
            preprocessing_result.normalized_text
        )
        merged: tuple[Any, ...] = anonymizer.merge_candidates(
            preprocessing_result.normalized_text,
            candidates,
        )
        postprocessed: Any | None = anonymizer.postprocess(
            original_text=preprocessing_result.original_text,
            preprocessed_text=preprocessing_result.normalized_text,
            candidates=merged,
            preprocessing_result=preprocessing_result,
        )

        self.assertIsNotNone(postprocessed)
        self.assertEqual(postprocessed.masked_original_text, "телефон [PHONE]")

    def test_model_runner_exposes_prediction_result_and_keeps_call_compatibility(self) -> None:
        runner = ModelRunner(model=FakeNameModel())
        text: str = "Пациент Иванов Иван Иванович пришел."

        result: Any = runner.run(text)

        self.assertEqual(runner(text), "Пациент [PER] пришел.")
        self.assertEqual(runner.mask(text), "Пациент [PER] пришел.")
        self.assertEqual(len(runner.spans(text)), 1)
        self.assertEqual(result.masked_text, "Пациент [PER] пришел.")
        self.assertEqual(result.spans[0].label, "PER")
        self.assertEqual(len(result.annotation.lines[0].spans), 1)

    def test_importing_anonymizer_does_not_import_ml_until_ml_is_used(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "from anonmed.anonymizer import PIIAnonymizer; "
                    "a = PIIAnonymizer(ml_model='example'); "
                    "a('телефон 89131234567', use_ml=False); "
                    "forbidden = {'anonmed.ml.pipelines.runner', 'anonmed.ml.registry'} & set(sys.modules); "
                    "raise SystemExit(1 if forbidden else 0)"
                ),
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed_process.returncode,
            0,
            msg=completed_process.stderr or completed_process.stdout,
        )

    def test_top_level_import_does_not_import_ml_until_ml_is_used(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "from anonmed import PIIAnonymizer; "
                    "a = PIIAnonymizer(ml_model='example'); "
                    "a('телефон 89131234567', use_ml=False); "
                    "forbidden = {'anonmed.ml.pipelines.runner', 'anonmed.ml.registry'} & set(sys.modules); "
                    "raise SystemExit(1 if forbidden else 0)"
                ),
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed_process.returncode,
            0,
            msg=completed_process.stderr or completed_process.stdout,
        )

    def test_non_string_text_raises_type_error(self) -> None:
        anonymizer = PIIAnonymizer()

        with self.assertRaisesRegex(TypeError, "text must be str"):
            anonymizer(123)  # type: ignore[arg-type]

    def test_unknown_flag_raises_type_error(self) -> None:
        anonymizer = PIIAnonymizer()

        with self.assertRaisesRegex(TypeError, "Unknown PIIAnonymizer flag"):
            anonymizer("телефон 89131234567", flags={"unknown_flag": True})

    def test_invalid_masking_strategy_raises_value_error(self) -> None:
        anonymizer = PIIAnonymizer()

        with self.assertRaisesRegex(ValueError, "masking_strategy"):
            anonymizer(
                "телефон 89131234567",
                use_ml=False,
                masking_strategy="invalid",
            )

    def test_cli_anonymize_mode_masks_text(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-m",
                "anonmed.cli",
                "ну,",
                "телефон:",
                "восемь",
                "девять",
                "один",
                "три",
                "один",
                "два",
                "три",
                "четыре",
                "пять",
                "шесть",
                "семь!",
                "--anonymize",
                "--no-ml",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed_process.returncode,
            0,
            msg=completed_process.stderr or completed_process.stdout,
        )
        self.assertEqual(completed_process.stdout.strip(), "ну, телефон: [PHONE]!")

    def test_cli_anonymize_json_mode_returns_structured_payload(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-m",
                "anonmed.cli",
                "телефон",
                "89131234567",
                "--anonymize-json",
                "--no-ml",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed_process.returncode,
            0,
            msg=completed_process.stderr or completed_process.stdout,
        )

        payload: dict[str, object] = json.loads(completed_process.stdout)
        self.assertEqual(payload["masked_text"], "телефон [PHONE]")
        self.assertIn("rule_candidates", payload)
        self.assertIn("config", payload)


__all__: list[str] = []
