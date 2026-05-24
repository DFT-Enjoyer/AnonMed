from collections.abc import Sequence

from anonmed.ml.core.snapshot import DatasetSnapshotWriter
from anonmed.ml.core.types import EvaluationReport
from anonmed.ml.data.base import Dataset
from anonmed.ml.evaluation.evaluator import Evaluator
from anonmed.ml.metrics.base import Metric
from anonmed.ml.models.base import PIIModel


def create_evaluator(dataset: Dataset) -> Evaluator:
    return Evaluator(dataset=dataset)


def evaluate(
    dataset: Dataset,
    model: PIIModel,
    metrics: Sequence[Metric],
    *,
    show_progress: bool = False,
) -> EvaluationReport:
    evaluator = create_evaluator(dataset=dataset)
    return evaluator.eval_model(model=model, metrics=metrics, show_progress=show_progress)


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
