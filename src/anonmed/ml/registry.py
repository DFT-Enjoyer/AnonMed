from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from anonmed.ml.config import DatasetConfig, MetricConfig, ModelConfig, PipelineConfig

from anonmed.ml.data.base import Dataset
from anonmed.ml.data.example import build_example_dataset

from anonmed.ml.metrics.base import Metric
from anonmed.ml.metrics.example import ExampleCountMetric
from anonmed.ml.metrics.entity_hard import EntityHardF1Metric
from anonmed.ml.metrics.char_hard import CharHardF1Metric

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

def _build_russian_pii_dataset(config: DatasetConfig) -> Dataset:
    from anonmed.ml.data.russian_pii_66k import RussianPIIDataset

    params = config.params
    sample_size = params.get("sample_size", 2000)
    random_seed = params.get("random_seed", 42)
    return RussianPIIDataset(sample_size=sample_size, random_seed=random_seed)

def _build_natasha_per_model(config: ModelConfig) -> PIIModel:
    from anonmed.ml.models.natasha_per import NatashaPERModel

    _reject_params(config.id, config.params)
    return NatashaPERModel()

def _build_entity_hard_f1(config: MetricConfig) -> Metric:
    _reject_params(config.id, config.params)
    return EntityHardF1Metric()

def _build_char_hard_f1(config: MetricConfig) -> Metric:
    _reject_params(config.id, config.params)
    return CharHardF1Metric()


DATASET_BUILDERS = {
    "example": _build_example_dataset,
    "russian_pii_66k": _build_russian_pii_dataset,
}

MODEL_BUILDERS = {
    "example": _build_example_model,
    "natasha_per": _build_natasha_per_model,
}

METRIC_BUILDERS = {
    "example_count": _build_example_count_metric,
    "entity_hard_f1": _build_entity_hard_f1,
    "char_hard_f1": _build_char_hard_f1,
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
