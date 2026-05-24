from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from anonmed.ml.core.types import AnnotationSet, TextDocument

if TYPE_CHECKING:
    from anonmed.ml.data.base import Dataset

_T = TypeVar("_T")


class PIIModel(ABC):
    def process(
        self,
        documents: tuple[TextDocument, ...],
        *,
        show_progress: bool = False,
        progress_description: str = "Predicting",
    ) -> tuple[AnnotationSet, ...]:
        if not isinstance(documents, tuple):
            raise TypeError(
                "documents must be tuple[TextDocument, ...], "
                f"got {type(documents).__name__}"
            )
        predictions: list[AnnotationSet] = []
        for document in _progress_iterable(
            documents,
            enabled=show_progress,
            description=progress_description,
        ):
            if not isinstance(document, TextDocument):
                raise TypeError(
                    "documents must contain TextDocument, "
                    f"got {type(document).__name__}"
                )
            prediction: AnnotationSet = self.predict(document)
            if not isinstance(prediction, AnnotationSet):
                raise TypeError(
                    "predict() must return AnnotationSet, "
                    f"got {type(prediction).__name__}"
                )
            predictions.append(prediction)
        return tuple(predictions)

    @abstractmethod
    def predict(self, document) -> AnnotationSet:
        ...


def _progress_iterable(
    iterable: tuple[_T, ...],
    *,
    enabled: bool,
    description: str,
) -> Iterable[_T]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=len(iterable), desc=description, unit="doc")


@runtime_checkable
class TrainablePIIModel(Protocol):
    def fit(self, dataset: "Dataset") -> object:
        ...
