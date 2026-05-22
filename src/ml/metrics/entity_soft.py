from ml.core.types import AnnotationSet
from ml.datasets.base import Dataset
from ml.metrics.base import Metric
from ml.metrics.utils import accuracy_without_tn, aggregate_counts, f1, precision, recall


class EntitySoftPrecisionMetric(Metric):
    @property
    def name(self) -> str:
        return "entity_soft_precision"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="entity_soft")
        return {"value": precision(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class EntitySoftRecallMetric(Metric):
    @property
    def name(self) -> str:
        return "entity_soft_recall"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="entity_soft")
        return {"value": recall(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class EntitySoftF1Metric(Metric):
    @property
    def name(self) -> str:
        return "entity_soft_f1"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="entity_soft")
        return {"value": f1(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class EntitySoftAccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "entity_soft_accuracy"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="entity_soft")
        return {"value": accuracy_without_tn(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}
