from abc import ABC, abstractmethod
from typing import Any

from ml.datasets.base import Dataset
from ml.core.types import AnnotationSet


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
