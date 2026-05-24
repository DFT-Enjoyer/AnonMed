from collections.abc import Sequence
from dataclasses import dataclass

from anonmed.ml.data.base import Dataset
from anonmed.ml.metrics.base import Metric
from anonmed.ml.models.base import PIIModel
from anonmed.ml.core.types import AnnotationSet, EvaluationReport, MetricResult


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    report: EvaluationReport
    predictions: tuple[AnnotationSet, ...]


class Evaluator:
    def __init__(self, dataset: Dataset):
        self._dataset: Dataset = dataset

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def eval_model(
        self,
        model: PIIModel,
        metrics: Sequence[Metric],
        *,
        show_progress: bool = False,
    ) -> EvaluationReport:
        return self.eval_model_with_predictions(
            model=model,
            metrics=metrics,
            show_progress=show_progress,
        ).report

    def eval_model_with_predictions(
        self,
        model: PIIModel,
        metrics: Sequence[Metric],
        *,
        show_progress: bool = False,
    ) -> EvaluationResult:
        for metric in metrics:
            metric.reset()

        predictions: tuple[AnnotationSet, ...] = model.process(
            self.dataset.documents,
            show_progress=show_progress,
        )

        results: dict[str, MetricResult] = {}
        for metric in metrics:
            metric_values: MetricResult = metric.compute(
                dataset=self._dataset,
                predictions=predictions,
            )
            results[metric.name] = metric_values

        report = EvaluationReport(metrics=results, samples_count=len(self._dataset.cases))
        return EvaluationResult(report=report, predictions=predictions)
