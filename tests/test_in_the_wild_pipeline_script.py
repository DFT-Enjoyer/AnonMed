from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from typing import Any, Final

from anonmed.ml.core.types import (
    AnnotationSet,
    AnnotationSetLine,
    Case,
    ParticipantKind,
    Role,
    Span,
    TextDocument,
    TextLine,
)

_SCRIPT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_in_the_wild_1_pipeline.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_in_the_wild_1_pipeline", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evaluate_in_the_wild_1_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case() -> Case:
    role = Role(name="text", kind=ParticipantKind.UNKNOWN)
    line = TextLine(idx=0, role=role, text="Пациент Иван телефон 89131234567")
    document = TextDocument(lines=(line,), sample_id="sample-1")
    target_line = AnnotationSetLine(
        idx=0,
        role=role,
        spans=[
            Span(line_idx=0, begin=8, end=12, label="full_name", data="Иван"),
            Span(line_idx=0, begin=21, end=32, label="phone", data="89131234567"),
        ],
    )
    target = AnnotationSet(lines=(target_line,), idx="sample-1")
    return Case(document=document, target=target)


class _FakeAnonymizer:
    def __call__(self, text: str, **_kwargs: object) -> Any:
        return SimpleNamespace(
            original_text=text,
            preprocessed_text=text,
            masked_original_text="Пациент [PER] телефон [PHONE]",
            warnings=(),
            postprocessed_mentions=(
                SimpleNamespace(
                    original_start=8,
                    original_end=12,
                    entity_type="PER",
                    source="ml",
                    rule_id="ml:PER",
                ),
                SimpleNamespace(
                    original_start=21,
                    original_end=32,
                    entity_type="PHONE",
                    source="regex",
                    rule_id="russian_mobile_phone",
                ),
            ),
        )


class InTheWildPipelineScriptTests(unittest.TestCase):
    def test_normalizes_dataset_labels_and_evaluates_fake_pipeline(self) -> None:
        script = _load_script()
        cases: tuple[Case, ...] = (script.normalize_case_labels(_case()),)

        _predictions, output = script.evaluate_cases(
            cases,
            anonymizer=_FakeAnonymizer(),
            soft_iou_threshold=0.5,
            show_progress=False,
        )

        self.assertEqual(output.stats.samples, 1)
        self.assertEqual(output.stats.gt_by_type, {"PER": 1, "PHONE": 1})
        self.assertEqual(output.stats.pred_by_type, {"PER": 1, "PHONE": 1})
        self.assertEqual(output.metrics["entity_hard"]["f1"], 1.0)
        self.assertEqual(output.per_type["PHONE"]["entity_soft"]["tp"], 1)


__all__: list[str] = []
