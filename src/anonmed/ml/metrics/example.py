from anonmed.ml.core.types import AnnotationSet
from anonmed.ml.data.base import Dataset
from anonmed.ml.metrics.base import Metric


class ExampleCountMetric(Metric):
    @property
    def name(self) -> str:
        return "example_count"

    def reset(self):
        return None

    def compute(
        self,
        dataset: Dataset,
        predictions: tuple[AnnotationSet, ...],
    ) -> dict[str, int]:
        self._validate_inputs(dataset, predictions)
        return {
            "predictions_count": len(predictions),
            "cases_count": len(dataset.cases),
        }


__all__: list[str] = ["ExampleCountMetric"]
