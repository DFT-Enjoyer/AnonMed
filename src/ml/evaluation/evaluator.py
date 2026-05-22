from collections.abc import Sequence

from ml.datasets.base import Dataset
from ml.metrics.base import Metric
from ml.models.base import PIIModel
from ml.core.types import AnnotationSet, EvaluationReport, MetricResult


class Evaluator:
    def __init__(self, dataset: Dataset):
        self._dataset: Dataset = dataset

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def eval_model(self, model: PIIModel, metrics: Sequence[Metric]) -> EvaluationReport:
        for metric in metrics:
            metric.reset()

        predictions: tuple[AnnotationSet, ...] = model.process(self.dataset.documents)

        results: dict[str, MetricResult] = {}
        for metric in metrics:
            metric_values: MetricResult = metric.compute(
                dataset=self._dataset,
                predictions=predictions,
            )
            results[metric.name] = metric_values

        report = EvaluationReport(metrics=results, samples_count=len(self._dataset.cases))
        return report
