from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from anonmed.ml.config import DatasetConfig, MetricConfig, ModelConfig, PipelineConfig
from anonmed.ml.datasets.base import Dataset
from anonmed.ml.datasets.example import build_example_dataset
from anonmed.ml.metrics.base import Metric
from anonmed.ml.metrics.example import ExampleCountMetric
from anonmed.ml.models.base import PIIModel
from anonmed.ml.models.example import ExamplePIIModel


DatasetBuilder = Callable[[DatasetConfig], Dataset]
ModelBuilder = Callable[[ModelConfig], PIIModel]
MetricBuilder = Callable[[MetricConfig], Metric]


@dataclass(frozen=True, slots=True)
class PipelineComponents:
    dataset: Dataset
    model: PIIModel
    metrics: tuple[Metric, ...]


class RegistryError(ValueError):
    pass


def _reject_params(component: str, params: object) -> None:
    if params:
        raise RegistryError(f"Component {component!r} does not accept params in the built-in registry.")


def _build_example_dataset(config: DatasetConfig) -> Dataset:
    _reject_params(config.id, config.params)
    return build_example_dataset()


def _build_example_model(config: ModelConfig) -> PIIModel:
    _reject_params(config.id, config.params)
    return ExamplePIIModel()


def _build_example_count_metric(config: MetricConfig) -> Metric:
    _reject_params(config.id, config.params)
    return ExampleCountMetric()


DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "example": _build_example_dataset,
}

MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "example": _build_example_model,
}

METRIC_BUILDERS: dict[str, MetricBuilder] = {
    "example_count": _build_example_count_metric,
}


def build_dataset(config: DatasetConfig) -> Dataset:
    builder = DATASET_BUILDERS.get(config.id)
    if builder is None:
        raise RegistryError(f"Unknown dataset id: {config.id}")
    return builder(config)


def build_model(config: ModelConfig) -> PIIModel:
    builder = MODEL_BUILDERS.get(config.id)
    if builder is None:
        raise RegistryError(f"Unknown model id: {config.id}")
    return builder(config)


def build_metric(config: MetricConfig) -> Metric:
    builder = METRIC_BUILDERS.get(config.id)
    if builder is None:
        raise RegistryError(f"Unknown metric id: {config.id}")
    return builder(config)


def build_metrics(configs: tuple[MetricConfig, ...]) -> tuple[Metric, ...]:
    return tuple(build_metric(config) for config in configs)


def build_pipeline_components(config: PipelineConfig) -> PipelineComponents:
    dataset = build_dataset(config.dataset)
    model = build_model(config.model)
    metrics = build_metrics(config.metrics)
    return PipelineComponents(dataset=dataset, model=model, metrics=metrics)


__all__ = [
    "DATASET_BUILDERS",
    "METRIC_BUILDERS",
    "MODEL_BUILDERS",
    "PipelineComponents",
    "RegistryError",
    "build_dataset",
    "build_metric",
    "build_metrics",
    "build_model",
    "build_pipeline_components",
]
