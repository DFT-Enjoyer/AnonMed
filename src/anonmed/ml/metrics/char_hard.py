from anonmed.ml.core.types import AnnotationSet
from anonmed.ml.datasets.base import Dataset
from anonmed.ml.metrics.base import Metric
from anonmed.ml.metrics.utils import accuracy_without_tn, aggregate_counts, f1, precision, recall


class CharHardPrecisionMetric(Metric):
    @property
    def name(self) -> str:
        return "char_hard_precision"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_hard")
        return {"value": precision(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class CharHardRecallMetric(Metric):
    @property
    def name(self) -> str:
        return "char_hard_recall"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_hard")
        return {"value": recall(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class CharHardF1Metric(Metric):
    @property
    def name(self) -> str:
        return "char_hard_f1"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_hard")
        return {"value": f1(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}


class CharHardAccuracyMetric(Metric):
    @property
    def name(self) -> str:
        return "char_hard_accuracy"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(dataset.cases, predictions, mode="char_hard")
        return {"value": accuracy_without_tn(counts), "tp": counts.tp, "fp": counts.fp, "fn": counts.fn}
