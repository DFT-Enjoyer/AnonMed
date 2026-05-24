from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from anonmed.ml.core.types import AnnotationSet
from anonmed.ml.data.base import Dataset
from anonmed.ml.metrics.base import Metric
from anonmed.ml.metrics.utils import (
    Counts,
    accuracy_without_tn,
    aggregate_counts,
    f1,
    precision,
    recall,
)

_ScoreFunction = Callable[[Counts], float]


@dataclass(frozen=True)
class _EntitySoftMetric(Metric):
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")

    def reset(self) -> None:
        return None

    def _compute_score(
        self,
        dataset: Dataset,
        predictions: tuple[AnnotationSet, ...],
        score: _ScoreFunction,
    ) -> dict[str, float | int]:
        self._validate_inputs(dataset, predictions)
        counts = aggregate_counts(
            dataset.cases,
            predictions,
            mode="entity_soft",
            entity_iou_threshold=self.threshold,
        )
        return {
            "value": score(counts),
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "threshold": self.threshold,
        }


class EntitySoftPrecisionMetric(_EntitySoftMetric):
    @property
    def name(self) -> str:
        return "entity_soft_precision"

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        return self._compute_score(dataset, predictions, precision)


class EntitySoftRecallMetric(_EntitySoftMetric):
    @property
    def name(self) -> str:
        return "entity_soft_recall"

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        return self._compute_score(dataset, predictions, recall)


class EntitySoftF1Metric(_EntitySoftMetric):
    @property
    def name(self) -> str:
        return "entity_soft_f1"

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        return self._compute_score(dataset, predictions, f1)


class EntitySoftAccuracyMetric(_EntitySoftMetric):
    @property
    def name(self) -> str:
        return "entity_soft_accuracy"

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float | int]:
        return self._compute_score(dataset, predictions, accuracy_without_tn)


__all__: list[str] = [
    "EntitySoftAccuracyMetric",
    "EntitySoftF1Metric",
    "EntitySoftPrecisionMetric",
    "EntitySoftRecallMetric",
]
