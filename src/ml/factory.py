from collections.abc import Sequence

from ml.core.snapshot import DatasetSnapshotWriter
from ml.core.types import EvaluationReport
from ml.datasets.base import Dataset
from ml.evaluation.evaluator import Evaluator
from ml.metrics.base import Metric
from ml.models.base import PIIModel


def create_evaluator(dataset: Dataset) -> Evaluator:
    return Evaluator(dataset=dataset)


def evaluate(
    dataset: Dataset,
    model: PIIModel,
    metrics: Sequence[Metric],
) -> EvaluationReport:
    evaluator = create_evaluator(dataset=dataset)
    return evaluator.eval_model(model=model, metrics=metrics)


def create_dataset_snapshot_writer() -> DatasetSnapshotWriter:
    return DatasetSnapshotWriter()


__all__ = [
    "Dataset",
    "DatasetSnapshotWriter",
    "Evaluator",
    "Metric",
    "PIIModel",
    "create_dataset_snapshot_writer",
    "create_evaluator",
    "evaluate",
]
