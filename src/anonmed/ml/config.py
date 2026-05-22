from __future__ import annotations

from dataclasses import dataclass, field
import ast
from pathlib import Path
from typing import Any, Mapping, Sequence


ConfigMapping = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "default"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    id: str
    params: ConfigMapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id: str
    params: ConfigMapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricConfig:
    id: str
    params: ConfigMapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    enabled: bool = False
    trainer: str | None = None
    params: ConfigMapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class OutputConfig:
    artifacts_dir: str = "artifacts/ml"
    report_filename: str = "report.json"
    write_dataset_snapshot_json: bool = False
    write_dataset_snapshot_parquet: bool = False


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    run: RunConfig
    dataset: DatasetConfig
    model: ModelConfig
    metrics: tuple[MetricConfig, ...]
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    payload = _load_yaml_mapping(config_path)
    return pipeline_config_from_mapping(payload)


def pipeline_config_from_mapping(payload: ConfigMapping) -> PipelineConfig:
    run = _run_config(payload.get("run", {}))
    dataset = _component_config(payload.get("dataset"), "dataset", DatasetConfig)
    model = _component_config(payload.get("model"), "model", ModelConfig)
    metrics = _metric_configs(payload.get("metrics", ()))
    training = _training_config(payload.get("training", {}))
    evaluation = _evaluation_config(payload.get("evaluation", {}))
    outputs = _output_config(payload.get("outputs", {}))
    return PipelineConfig(
        run=run,
        dataset=dataset,
        model=model,
        metrics=metrics,
        training=training,
        evaluation=evaluation,
        outputs=outputs,
    )


def _load_yaml_mapping(path: Path) -> ConfigMapping:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        data = _load_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Config {path!s} must contain a mapping at the top level.")
    return data


def _load_simple_yaml(text: str) -> ConfigMapping:
    root: dict[str, Any] = {}
    current_parent: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            key, value = _split_yaml_key_value(stripped)
            if value is None:
                root[key] = {}
                current_parent = key
            else:
                root[key] = _parse_scalar(value)
                current_parent = None
            continue
        if indent == 2 and current_parent is not None:
            parent = root[current_parent]
            if not isinstance(parent, dict):
                raise ValueError(f"Cannot add nested key under scalar section {current_parent!r}.")
            key, value = _split_yaml_key_value(stripped)
            if value is None:
                parent[key] = {}
            else:
                parent[key] = _parse_scalar(value)
            continue
        raise ValueError("Install PyYAML for nested or advanced YAML config files.")
    return root


def _split_yaml_key_value(line: str) -> tuple[str, str | None]:
    if ":" not in line:
        raise ValueError(f"Expected YAML key/value line, got {line!r}.")
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Expected non-empty YAML key in line {line!r}.")
    return key, value or None


def _parse_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _run_config(raw: object) -> RunConfig:
    if raw is None:
        return RunConfig()
    if isinstance(raw, str):
        return RunConfig(name=raw)
    mapping = _require_mapping(raw, "run")
    return RunConfig(name=str(mapping.get("name", "default")))


def _component_config(raw: object, section: str, cls: type[DatasetConfig] | type[ModelConfig]):
    if raw is None:
        raise ValueError(f"Missing required config section: {section}.")
    if isinstance(raw, str):
        return cls(id=raw)
    mapping = _require_mapping(raw, section)
    component_id = mapping.get("id")
    if not isinstance(component_id, str) or not component_id:
        raise ValueError(f"Config section {section!r} must define a non-empty string id.")
    params = mapping.get("params", {})
    return cls(id=component_id, params=_require_mapping(params, f"{section}.params"))


def _metric_configs(raw: object) -> tuple[MetricConfig, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (MetricConfig(id=raw),)
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raise TypeError("Config section 'metrics' must be a list of ids or mappings.")
    metrics: list[MetricConfig] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            metrics.append(MetricConfig(id=item))
            continue
        mapping = _require_mapping(item, f"metrics[{index}]")
        metric_id = mapping.get("id")
        if not isinstance(metric_id, str) or not metric_id:
            raise ValueError(f"Metric config at index {index} must define a non-empty string id.")
        params = mapping.get("params", {})
        metrics.append(
            MetricConfig(id=metric_id, params=_require_mapping(params, f"metrics[{index}].params"))
        )
    return tuple(metrics)


def _training_config(raw: object) -> TrainingConfig:
    if raw is None:
        return TrainingConfig()
    mapping = _require_mapping(raw, "training")
    params = mapping.get("params", {})
    trainer = mapping.get("trainer")
    return TrainingConfig(
        enabled=bool(mapping.get("enabled", False)),
        trainer=str(trainer) if trainer is not None else None,
        params=_require_mapping(params, "training.params"),
    )


def _evaluation_config(raw: object) -> EvaluationConfig:
    if raw is None:
        return EvaluationConfig()
    mapping = _require_mapping(raw, "evaluation")
    return EvaluationConfig(enabled=bool(mapping.get("enabled", True)))


def _output_config(raw: object) -> OutputConfig:
    if raw is None:
        return OutputConfig()
    mapping = _require_mapping(raw, "outputs")
    return OutputConfig(
        artifacts_dir=str(mapping.get("artifacts_dir", "artifacts/ml")),
        report_filename=str(mapping.get("report_filename", "report.json")),
        write_dataset_snapshot_json=bool(mapping.get("write_dataset_snapshot_json", False)),
        write_dataset_snapshot_parquet=bool(mapping.get("write_dataset_snapshot_parquet", False)),
    )


def _require_mapping(raw: object, section: str) -> ConfigMapping:
    if not isinstance(raw, Mapping):
        raise TypeError(f"Config section {section!r} must be a mapping.")
    return raw


__all__ = [
    "DatasetConfig",
    "EvaluationConfig",
    "MetricConfig",
    "ModelConfig",
    "OutputConfig",
    "PipelineConfig",
    "RunConfig",
    "TrainingConfig",
    "load_pipeline_config",
    "pipeline_config_from_mapping",
]
