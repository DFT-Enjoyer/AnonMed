from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from anonmed.ml import ModelRunner as PublicModelRunner
from anonmed.ml.config import DatasetConfig, MetricConfig
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
from anonmed.ml.data.base import Dataset
from anonmed.ml.data.in_the_wild_datasets import (
    InTheWildComprehensivePIIDataset,
    InTheWildControlledPIIDataset,
    InTheWildDialogPIIDataset,
    InTheWildMedicalNotesPIIDataset,
    InTheWildNamesAddressesDataset,
    InTheWildNewsEntityDataset,
)
from anonmed.ml.metrics.soft import EntitySoftPrecisionMetric
from anonmed.ml.models.base import PIIModel
from anonmed.ml.registry import build_dataset, build_metric


class _StaticDataset(Dataset):
    def _load(self) -> None:
        object.__setattr__(self, "_row_data", None)

    def _convert(self) -> None:
        role = Role(name="text", kind=ParticipantKind.UNKNOWN)
        line = TextLine(idx=0, role=role, text="0123456789abcdef")
        document = TextDocument(lines=(line,), sample_id="sample-1")
        target_span = Span(line_idx=0, begin=0, end=10, label="PER", data="0123456789")
        target_line = AnnotationSetLine(idx=0, role=role, spans=[target_span])
        target = AnnotationSet(lines=(target_line,), idx="sample-1")
        object.__setattr__(self, "cases", (Case(document=document, target=target),))


def _prediction(begin: int, end: int) -> AnnotationSet:
    role = Role(name="text", kind=ParticipantKind.UNKNOWN)
    span = Span(line_idx=0, begin=begin, end=end, label="PER", data="x")
    line = AnnotationSetLine(idx=0, role=role, spans=[span])
    return AnnotationSet(lines=(line,), idx="sample-1")


class _StaticModel(PIIModel):
    def predict(self, document: TextDocument) -> AnnotationSet:
        line = document.lines[0]
        begin = line.text.index("Иван")
        end = begin + len("Иван")
        span = Span(line_idx=line.idx, begin=begin, end=end, label="PER", data="Иван")
        target_line = AnnotationSetLine(idx=line.idx, role=line.role, spans=[span])
        return AnnotationSet(lines=(target_line,), idx=document.sample_id)


class MLDatasetMetricRunnerTests(unittest.TestCase):
    def test_in_the_wild_wrappers_read_brat_and_json_entities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            letters_dir = root / "final_version" / "2" / "letters"
            digits_dir = root / "final_version" / "1" / "digits"
            letters_dir.mkdir(parents=True)
            digits_dir.mkdir(parents=True)
            (letters_dir / "letters_data.jsonl").write_text(
                json.dumps(
                    {
                        "id": "brat-1",
                        "text": "Привет Анна",
                        "entities": ["T1\tPERSON 7 11\tАнна"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (digits_dir / "digits_data.jsonl").write_text(
                json.dumps(
                    {
                        "source_text": "телефон 89131234567",
                        "entities": [
                            {
                                "start": 8,
                                "end": 19,
                                "text": "89131234567",
                                "type": "phone",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            brat_dataset = InTheWildNewsEntityDataset(
                root=root,
                representation="letters",
            )
            json_dataset = InTheWildComprehensivePIIDataset(
                root=root,
                representation="digits",
            )

        brat_span = brat_dataset.cases[0].target.lines[0].spans[0]
        json_span = json_dataset.cases[0].target.lines[0].spans[0]
        self.assertEqual((brat_span.begin, brat_span.end, brat_span.label), (7, 11, "PERSON"))
        self.assertEqual((json_span.begin, json_span.end, json_span.label), (8, 19, "phone"))

    def test_dataset_registry_builds_in_the_wild_wrappers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version_dir = root / "final_version" / "6" / "letters"
            version_dir.mkdir(parents=True)
            (version_dir / "letters_data.jsonl").write_text(
                json.dumps(
                    {
                        "filename": "note-1",
                        "text": "Пациент Иван",
                        "entities": [
                            {
                                "start": 8,
                                "end": 12,
                                "text": "Иван",
                                "type": "full_name",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = build_dataset(
                DatasetConfig(
                    id=InTheWildMedicalNotesPIIDataset.name,
                    params={"root": str(root)},
                )
            )

        self.assertIsInstance(dataset, InTheWildMedicalNotesPIIDataset)
        span = dataset.cases[0].target.lines[0].spans[0]
        self.assertEqual((span.begin, span.end, span.label), (8, 12, "full_name"))

    def test_all_in_the_wild_wrappers_have_fixed_versions(self) -> None:
        wrappers = (
            (InTheWildComprehensivePIIDataset, "1", "in_the_wild_russian_pii_speech"),
            (InTheWildNewsEntityDataset, "2", "in_the_wild_russian_news_ner"),
            (InTheWildNamesAddressesDataset, "3", "in_the_wild_russian_names_addresses"),
            (InTheWildDialogPIIDataset, "4", "in_the_wild_dialog_pii"),
            (InTheWildControlledPIIDataset, "5", "in_the_wild_controlled_synthetic_pii"),
            (InTheWildMedicalNotesPIIDataset, "6", "in_the_wild_medical_notes_pii"),
        )

        for wrapper, directory, name in wrappers:
            with self.subTest(wrapper=wrapper.__name__):
                self.assertEqual(wrapper.dataset_directory, directory)
                self.assertEqual(wrapper.name, name)

    def test_entity_soft_metric_uses_iou_threshold(self) -> None:
        dataset = _StaticDataset()
        loose_metric = EntitySoftPrecisionMetric(threshold=0.3)
        strict_metric = EntitySoftPrecisionMetric(threshold=0.5)

        loose_result = loose_metric.compute(dataset, (_prediction(5, 15),))
        strict_result = strict_metric.compute(dataset, (_prediction(5, 15),))

        self.assertEqual(loose_result["tp"], 1)
        self.assertEqual(strict_result["tp"], 0)
        self.assertEqual(strict_result["fp"], 1)
        self.assertEqual(strict_result["fn"], 1)

    def test_metric_registry_exports_soft_iou_params(self) -> None:
        metric = build_metric(MetricConfig(id="entity_soft_f1", params={"threshold": 0.75}))
        self.assertEqual(metric.name, "entity_soft_f1")
        self.assertEqual(metric.threshold, 0.75)

    def test_public_model_runner_uses_ml_model_only(self) -> None:
        runner = PublicModelRunner(model=_StaticModel())
        masked_text = runner("Пациент Иван пришел")
        self.assertIsInstance(masked_text, str)
        self.assertEqual(masked_text, "Пациент [PER] пришел")


__all__: list[str] = []
