from abc import ABC, abstractmethod

from ml.core.types import AnnotationSet, TextDocument


class PIIModel(ABC):
    def process(self, documents: tuple[TextDocument, ...]) -> tuple[AnnotationSet, ...]:
        predictions: list[AnnotationSet] = []
        for document in documents:
            prediction: AnnotationSet = self.predict(document)
            predictions.append(prediction)
        return tuple(predictions)

    @abstractmethod
    def predict(self, document) -> AnnotationSet:
        ...
