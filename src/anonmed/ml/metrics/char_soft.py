from anonmed.ml.core.types import AnnotationSet
from anonmed.ml.datasets.base import Dataset
from anonmed.ml.metrics.base import Metric
from anonmed.ml.metrics.utils import accuracy_without_tn, aggregate_counts, f1, precision, recall


class CharSoftPrecisionMetric(Metric):
    @property
    def name(self) -> str:
        return "char_soft_precision"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_soft")
        return {"value": precision(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class CharSoftRecallMetric(Metric):
    @property
    def name(self) -> str:
        return "char_soft_recall"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_soft")
        return {"value": recall(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class CharSoftF1Metric(Metric):
    @property
    def name(self) -> str:
        return "char_soft_f1"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_soft")
        return {"value": f1(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class CharSoftAccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "char_soft_accuracy"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_soft")
        return {"value": accuracy_without_tn(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}
