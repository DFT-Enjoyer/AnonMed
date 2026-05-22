from ml.core.types import AnnotationSet
from ml.datasets.base import Dataset
from ml.metrics.base import Metric
from ml.metrics.utils import coverage_percent


class CoveragePercentMetric(Metric):
    @property
    def name(self) -> str:
        return "coverage_percent"

    def reset(self):
        return None

    def compute(self, dataset: Dataset, predictions: tuple[AnnotationSet, ...]) -> dict[str, float]:
        self._validate_inputs(dataset, predictions)
        coverage, over_coverage = coverage_percent(dataset.cases, predictions)
        return {
            "coverage_percent": coverage,
            "over_coverage_percent": over_coverage,
        }
