from ml.core.types import AnnotationSet
from ml.datasets.base import Dataset
from ml.metrics.base import Metric


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
        return {
            "predictions_count": len(predictions),
            "cases_count": len(dataset.cases),
        }
