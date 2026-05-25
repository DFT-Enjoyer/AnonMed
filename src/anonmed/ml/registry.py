from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from anonmed.ml.config import DatasetConfig, MetricConfig, ModelConfig, PipelineConfig

from anonmed.ml.data.base import Dataset
from anonmed.ml.data.example import build_example_dataset
from anonmed.ml.data.in_the_wild_datasets import (
    InTheWildComprehensivePIIDataset,
    InTheWildControlledPIIDataset,
    InTheWildDataset,
    InTheWildDialogPIIDataset,
    InTheWildMedicalNotesPIIDataset,
    InTheWildNamesAddressesDataset,
    InTheWildNewsEntityDataset,
)

from anonmed.ml.metrics import (
    CharHardAccuracyMetric,
    CharHardF1Metric,
    CharHardPrecisionMetric,
    CharHardRecallMetric,
    CharSoftAccuracyMetric,
    CharSoftF1Metric,
    CharSoftPrecisionMetric,
    CharSoftRecallMetric,
    CoveragePercentMetric,
    EntityHardAccuracyMetric,
    EntityHardF1Metric,
    EntityHardPrecisionMetric,
    EntityHardRecallMetric,
    EntitySoftAccuracyMetric,
    EntitySoftF1Metric,
    EntitySoftPrecisionMetric,
    EntitySoftRecallMetric,
    ExampleCountMetric,
    Metric,
)

from anonmed.ml.models.base import PIIModel
from anonmed.ml.models.example import ExamplePIIModel


DatasetBuilder = Callable[[DatasetConfig], Dataset]
ModelBuilder = Callable[[ModelConfig], PIIModel]
MetricBuilder = Callable[[MetricConfig], Metric]
MetricType = type[Metric]
InTheWildDatasetType = type[InTheWildDataset]


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


def _build_in_the_wild_dataset_from_class(
    config: DatasetConfig,
    dataset_class: InTheWildDatasetType,
) -> Dataset:
    try:
        return dataset_class(**dict(config.params))
    except TypeError as error:
        raise RegistryError(
            f"Invalid params for dataset {config.id!r}: {dict(config.params)!r}"
        ) from error


def _in_the_wild_dataset_builder(dataset_class: InTheWildDatasetType) -> DatasetBuilder:
    def build(config: DatasetConfig) -> Dataset:
        return _build_in_the_wild_dataset_from_class(config, dataset_class)

    return build


def _build_example_model(config: ModelConfig) -> PIIModel:
    _reject_params(config.id, config.params)
    return ExamplePIIModel()


def _build_metric_from_class(config: MetricConfig, metric_class: MetricType) -> Metric:
    try:
        return metric_class(**dict(config.params))
    except TypeError as error:
        raise RegistryError(
            f"Invalid params for metric {config.id!r}: {dict(config.params)!r}"
        ) from error


def _build_russian_pii_dataset(config: DatasetConfig) -> Dataset:
    from anonmed.ml.data.russian_pii_66k import RussianPIIDataset

    params = config.params
    sample_size = params.get("sample_size", 2000)
    random_seed = params.get("random_seed", 42)
    return RussianPIIDataset(sample_size=sample_size, random_seed=random_seed)


def _build_gt_asr(config: DatasetConfig) -> Dataset:
    from anonmed.ml.data.gt_asr import GTASRDataset

    params = dict(config.params)
    if "annotation_types" in params and isinstance(params["annotation_types"], list):
        params["annotation_types"] = tuple(str(value) for value in params["annotation_types"])
    try:
        return GTASRDataset(**params)
    except TypeError as error:
        raise RegistryError(
            f"Invalid params for dataset {config.id!r}: {dict(config.params)!r}"
        ) from error


def _build_natasha_per_model(config: ModelConfig) -> PIIModel:
    from anonmed.ml.models.natasha_per import NatashaPERModel

    _reject_params(config.id, config.params)
    return NatashaPERModel()


def _build_qwen06b_model(config: ModelConfig) -> PIIModel:
    from anonmed.ml.models.Qwen06B import Qwen06BModel

    try:
        return Qwen06BModel(**dict(config.params))
    except TypeError as error:
        raise RegistryError(
            f"Invalid params for model {config.id!r}: {dict(config.params)!r}"
        ) from error

def _metric_builder(metric_class: MetricType) -> MetricBuilder:
    def build(config: MetricConfig) -> Metric:
        return _build_metric_from_class(config, metric_class)

    return build


def _build_gliner2_model(config: ModelConfig) -> PIIModel:
    from anonmed.ml.models.GLiNER2 import GLiNER2Model

    return GLiNER2Model(**dict(config.params))


DATASET_BUILDERS = {
    "example": _build_example_dataset,
    InTheWildComprehensivePIIDataset.name: _in_the_wild_dataset_builder(
        InTheWildComprehensivePIIDataset
    ),
    InTheWildControlledPIIDataset.name: _in_the_wild_dataset_builder(
        InTheWildControlledPIIDataset
    ),
    InTheWildDialogPIIDataset.name: _in_the_wild_dataset_builder(InTheWildDialogPIIDataset),
    InTheWildMedicalNotesPIIDataset.name: _in_the_wild_dataset_builder(
        InTheWildMedicalNotesPIIDataset
    ),
    InTheWildNamesAddressesDataset.name: _in_the_wild_dataset_builder(
        InTheWildNamesAddressesDataset
    ),
    InTheWildNewsEntityDataset.name: _in_the_wild_dataset_builder(InTheWildNewsEntityDataset),
    "russian_pii_66k": _build_russian_pii_dataset,
    "gt_asr": _build_gt_asr,
}

MODEL_BUILDERS = {
    "example": _build_example_model,
    "natasha_per": _build_natasha_per_model,
    "GLiNER2": _build_gliner2_model,
    "Qwen06B": _build_qwen06b_model,
}

METRIC_BUILDERS = {
    "char_hard_accuracy": _metric_builder(CharHardAccuracyMetric),
    "char_hard_f1": _metric_builder(CharHardF1Metric),
    "char_hard_precision": _metric_builder(CharHardPrecisionMetric),
    "char_hard_recall": _metric_builder(CharHardRecallMetric),
    "char_soft_accuracy": _metric_builder(CharSoftAccuracyMetric),
    "char_soft_f1": _metric_builder(CharSoftF1Metric),
    "char_soft_precision": _metric_builder(CharSoftPrecisionMetric),
    "char_soft_recall": _metric_builder(CharSoftRecallMetric),
    "coverage_percent": _metric_builder(CoveragePercentMetric),
    "entity_hard_accuracy": _metric_builder(EntityHardAccuracyMetric),
    "entity_hard_f1": _metric_builder(EntityHardF1Metric),
    "entity_hard_precision": _metric_builder(EntityHardPrecisionMetric),
    "entity_hard_recall": _metric_builder(EntityHardRecallMetric),
    "entity_soft_accuracy": _metric_builder(EntitySoftAccuracyMetric),
    "entity_soft_f1": _metric_builder(EntitySoftF1Metric),
    "entity_soft_precision": _metric_builder(EntitySoftPrecisionMetric),
    "entity_soft_recall": _metric_builder(EntitySoftRecallMetric),
    "example_count": _metric_builder(ExampleCountMetric),
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
