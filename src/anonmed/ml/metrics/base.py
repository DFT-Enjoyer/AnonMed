from abc import ABC, abstractmethod
from typing import Any

from anonmed.ml.data.base import Dataset
from anonmed.ml.core.types import AnnotationSet


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def compute(
        self,
        dataset: Dataset,
        predictions: tuple[AnnotationSet, ...],
    ) -> dict[str, Any]:
        ...

    def _validate_inputs(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be Dataset, got {type(dataset).__name__}")
        if not isinstance(predictions, tuple):
            raise TypeError(f"predictions must be tuple[AnnotationSet, ...], got {type(predictions).__name__}")
        for prediction in predictions:
            if not isinstance(prediction, AnnotationSet):
                raise TypeError(f"predictions must contain AnnotationSet, got {type(prediction).__name__}")
        if len(predictions) != len(dataset.cases):
            raise ValueError(
                "predictions and dataset size mismatch: "
                f"len(predictions)={len(predictions)} != len(dataset.cases)={len(dataset.cases)}"
            )
