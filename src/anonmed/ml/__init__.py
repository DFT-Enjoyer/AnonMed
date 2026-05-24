"""ML primitives and pipelines for AnonMed."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AnnotationSet",
    "AnnotationSetLine",
    "Case",
    "ConfigMapping",
    "Dataset",
    "DatasetConfig",
    "DatasetSnapshotWriter",
    "EvaluationConfig",
    "EvaluationReport",
    "Evaluator",
    "FineTunedPIDRModel",
    "GLiNER2Model",
    "Metric",
    "MetricConfig",
    "MetricResult",
    "MetricValue",
    "ModelRunner",
    "ModelRunnerResult",
    "ModelConfig",
    "OutputConfig",
    "PIDRModel",
    "Qwen06BModel",
    "PIIModel",
    "ParticipantKind",
    "PipelineComponents",
    "PipelineConfig",
    "RegistryError",
    "Role",
    "RunConfig",
    "Span",
    "TextDocument",
    "TextLine",
    "TrainablePIIModel",
    "TrainingConfig",
    "build_dataset",
    "build_metric",
    "build_metrics",
    "build_model",
    "build_pipeline_components",
    "create_dataset_snapshot_writer",
    "create_evaluator",
    "evaluate",
    "load_pipeline_config",
    "pipeline_config_from_mapping",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AnnotationSet": ("anonmed.ml.core.types", "AnnotationSet"),
    "AnnotationSetLine": ("anonmed.ml.core.types", "AnnotationSetLine"),
    "Case": ("anonmed.ml.core.types", "Case"),
    "ConfigMapping": ("anonmed.ml.config", "ConfigMapping"),
    "Dataset": ("anonmed.ml.data.base", "Dataset"),
    "DatasetConfig": ("anonmed.ml.config", "DatasetConfig"),
    "DatasetSnapshotWriter": ("anonmed.ml.core.snapshot", "DatasetSnapshotWriter"),
    "EvaluationConfig": ("anonmed.ml.config", "EvaluationConfig"),
    "EvaluationReport": ("anonmed.ml.core.types", "EvaluationReport"),
    "Evaluator": ("anonmed.ml.evaluation.evaluator", "Evaluator"),
    "FineTunedPIDRModel": ("anonmed.ml.models.PIDR", "FineTunedPIDRModel"),
    "GLiNER2Model": ("anonmed.ml.models.GLiNER2", "GLiNER2Model"),
    "Metric": ("anonmed.ml.metrics.base", "Metric"),
    "MetricConfig": ("anonmed.ml.config", "MetricConfig"),
    "MetricResult": ("anonmed.ml.core.types", "MetricResult"),
    "MetricValue": ("anonmed.ml.core.types", "MetricValue"),
    "ModelRunner": ("anonmed.ml.pipelines.runner", "ModelRunner"),
    "ModelRunnerResult": ("anonmed.ml.pipelines.runner", "ModelRunnerResult"),
    "ModelConfig": ("anonmed.ml.config", "ModelConfig"),
    "OutputConfig": ("anonmed.ml.config", "OutputConfig"),
    "PIDRModel": ("anonmed.ml.models.PIDR", "PIDRModel"),
    "Qwen06BModel": ("anonmed.ml.models.Qwen06B", "Qwen06BModel"),
    "PIIModel": ("anonmed.ml.models.base", "PIIModel"),
    "ParticipantKind": ("anonmed.ml.core.types", "ParticipantKind"),
    "PipelineComponents": ("anonmed.ml.registry", "PipelineComponents"),
    "PipelineConfig": ("anonmed.ml.config", "PipelineConfig"),
    "RegistryError": ("anonmed.ml.registry", "RegistryError"),
    "Role": ("anonmed.ml.core.types", "Role"),
    "RunConfig": ("anonmed.ml.config", "RunConfig"),
    "Span": ("anonmed.ml.core.types", "Span"),
    "TextDocument": ("anonmed.ml.core.types", "TextDocument"),
    "TextLine": ("anonmed.ml.core.types", "TextLine"),
    "TrainablePIIModel": ("anonmed.ml.models.base", "TrainablePIIModel"),
    "TrainingConfig": ("anonmed.ml.config", "TrainingConfig"),
    "build_dataset": ("anonmed.ml.registry", "build_dataset"),
    "build_metric": ("anonmed.ml.registry", "build_metric"),
    "build_metrics": ("anonmed.ml.registry", "build_metrics"),
    "build_model": ("anonmed.ml.registry", "build_model"),
    "build_pipeline_components": ("anonmed.ml.registry", "build_pipeline_components"),
    "create_dataset_snapshot_writer": (
        "anonmed.ml.factory",
        "create_dataset_snapshot_writer",
    ),
    "create_evaluator": ("anonmed.ml.factory", "create_evaluator"),
    "evaluate": ("anonmed.ml.factory", "evaluate"),
    "load_pipeline_config": ("anonmed.ml.config", "load_pipeline_config"),
    "pipeline_config_from_mapping": ("anonmed.ml.config", "pipeline_config_from_mapping"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
