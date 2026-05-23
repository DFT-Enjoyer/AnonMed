from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from anonmed.ml.core.types import AnnotationSet, TextDocument

if TYPE_CHECKING:
    from anonmed.ml.data.base import Dataset


class PIIModel(ABC):
    def process(self, documents: tuple[TextDocument, ...]) -> tuple[AnnotationSet, ...]:
        if not isinstance(documents, tuple):
            raise TypeError(f"documents must be tuple[TextDocument, ...], got {type(documents).__name__}")
        predictions: list[AnnotationSet] = []
        for document in documents:
            if not isinstance(document, TextDocument):
                raise TypeError(f"documents must contain TextDocument, got {type(document).__name__}")
            prediction: AnnotationSet = self.predict(document)
            if not isinstance(prediction, AnnotationSet):
                raise TypeError(f"predict() must return AnnotationSet, got {type(prediction).__name__}")
            predictions.append(prediction)
        return tuple(predictions)

    @abstractmethod
    def predict(self, document) -> AnnotationSet:
        ...


@runtime_checkable
class TrainablePIIModel(Protocol):
    def fit(self, dataset: "Dataset") -> object:
        ...
