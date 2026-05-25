from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from anonmed.ml.config import PipelineConfig, RunConfig, load_pipeline_config
from anonmed.ml.core.snapshot import DatasetSnapshotWriter
from anonmed.ml.factory import evaluate_with_predictions
from anonmed.ml.models.GLiNER2 import DEFAULT_ENTITY_DESCRIPTION, GLiNER2Model
from anonmed.ml.outputs import build_run_instance_dir
from anonmed.ml.registry import build_dataset, build_metrics, build_model


DEFAULT_PROMPTS: tuple[str, ...] = (
    DEFAULT_ENTITY_DESCRIPTION,
    (
        "Person full name in Russian medical or administrative text. Extract only complete "
        "mentions of real people: surname, given name, patronymic, or their contiguous "
        "combination. Treat Russian ФИО as one entity span. Include inflected Russian names "
        "and initials only when they are part of the same personal-name mention. Exclude "
        "roles, professions, organizations, locations, dates, document ids, phone numbers, "
        "diagnoses, and generic words such as пациент or врач."
    ),
    (
        "A Russian personal-name mention identifying an individual person. The entity may be "
        "a full ФИО, surname plus initials, name plus patronymic, or a single surname/name "
        "when it clearly refers to a person. Return one continuous span per person mention. "
        "Do not split name components. Do not mark hospitals, departments, job titles, "
        "addresses, numbers, dates, or medical terms."
    ),
    (
        "Full name of a human being, especially Russian ФИО in clinical notes, dialogs, "
        "forms, and transcripts. Extract the exact text span that names the person, including "
        "surname/name/patronymic/initials when adjacent. Avoid overmatching surrounding words "
        "such as patient, doctor, relative, signature, diagnosis, organization, address, or "
        "document fields."
    ),
)


PRECISION_METRIC = "entity_hard_precision"
RECALL_METRIC = "entity_hard_recall"
F1_METRIC = "entity_hard_f1"


def _load_optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        message = (
            "GLiNER2 threshold search requires the 'optuna' package. "
            "Install it with `pip install optuna` or install the ML extras."
        )
        raise ImportError(message) from error
    return optuna


def run_threshold_search(
    config: PipelineConfig,
    *,
    n_trials: int | None = None,
    min_recall: float = 0.9,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.05,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    study_name: str = "gliner2-threshold-search",
    storage: str | None = None,
    sampler_name: str = "grid",
    show_progress: bool = True,
    show_document_progress: bool = False,
) -> dict[str, Any]:
    optuna = _load_optuna()
    dataset = build_dataset(config.dataset)
    metrics = build_metrics(config.metrics)
    base_model = build_model(config.model)
    if not isinstance(base_model, GLiNER2Model):
        raise TypeError(
            "GLiNER2 threshold search requires config.model.id='GLiNER2', "
            f"got {config.model.id!r}."
        )
    if not prompts:
        raise ValueError("prompts must contain at least one entity description.")

    threshold_values = _threshold_values(threshold_min, threshold_max, threshold_step)
    extractor = base_model._extractor
    base_params = dict(config.model.params)
    base_params.pop("threshold", None)
    base_params.pop("entity_description", None)

    def objective(trial: Any) -> float:
        threshold = trial.suggest_categorical(
            "threshold",
            threshold_values,
        )
        prompt_index = trial.suggest_categorical(
            "prompt_index",
            list(range(len(prompts))),
        )
        prompt = prompts[int(prompt_index)]
        model = GLiNER2Model(
            **base_params,
            threshold=float(threshold),
            entity_description=prompt,
            extractor=extractor,
        )
        result = evaluate_with_predictions(
            dataset,
            model,
            metrics,
            show_progress=show_document_progress,
        )
        precision = _metric_value(result.report.metrics, PRECISION_METRIC)
        recall = _metric_value(result.report.metrics, RECALL_METRIC)
        f1 = _metric_value(result.report.metrics, F1_METRIC)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("f1", f1)
        trial.set_user_attr("metrics", result.report.metrics)
        trial.set_user_attr("entity_description", prompt)
        trial.set_user_attr("feasible", recall >= min_recall)
        if recall >= min_recall:
            return precision
        return recall - min_recall - 1.0

    sampler = _build_sampler(
        optuna,
        sampler_name=sampler_name,
        threshold_values=threshold_values,
        prompts_count=len(prompts),
        seed=_random_seed(config),
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    resolved_trials_count = _resolve_trials_count(
        n_trials=n_trials,
        sampler_name=sampler_name,
        threshold_values=threshold_values,
        prompts_count=len(prompts),
    )
    progress_bar = _trial_progress_bar(
        enabled=show_progress,
        total=resolved_trials_count,
        description="GLiNER2 threshold search",
    )
    try:
        study.optimize(
            objective,
            n_trials=resolved_trials_count,
            callbacks=[_progress_callback(progress_bar)] if progress_bar is not None else None,
            show_progress_bar=False,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    best_trial = _best_feasible_trial(study.trials, min_recall) or study.best_trial
    best_prompt = str(best_trial.user_attrs.get("entity_description", prompts[0]))
    best_threshold = float(best_trial.params["threshold"])
    best_params = {
        **base_params,
        "threshold": best_threshold,
        "entity_description": best_prompt,
    }
    best_model = GLiNER2Model(**best_params, extractor=extractor)
    best_result = evaluate_with_predictions(
        dataset,
        best_model,
        metrics,
        show_progress=show_document_progress,
    )

    output_config = replace(
        config,
        run=RunConfig(name=f"{config.run.name}_threshold_search"),
    )
    instance_dir = build_run_instance_dir(output_config)
    instance_dir.mkdir(parents=True, exist_ok=True)

    snapshot_writer = DatasetSnapshotWriter()
    evaluation_snapshot_path = instance_dir / "evaluation_snapshot.json"
    snapshot_writer.write_evaluation_json(
        dataset,
        best_result.predictions,
        evaluation_snapshot_path,
    )
    if config.outputs.write_dataset_snapshot_json:
        dataset_snapshot_path = instance_dir / "dataset_snapshot.json"
        snapshot_writer.write_json(dataset, dataset_snapshot_path)
    else:
        dataset_snapshot_path = None

    trials_payload = [_trial_to_dict(trial) for trial in study.trials]
    trials_path = instance_dir / "trials.json"
    trials_path.write_text(
        json.dumps(trials_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "run": {"name": output_config.run.name},
        "dataset": asdict(config.dataset),
        "model": {"id": config.model.id, "params": best_params},
        "metrics": [asdict(metric_config) for metric_config in config.metrics],
        "training": asdict(config.training),
        "evaluation": asdict(config.evaluation),
        "samples_count": best_result.report.samples_count,
        "metric_results": best_result.report.metrics,
        "optimization": {
            "library": "optuna",
            "sampler": sampler_name,
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "threshold_values": threshold_values,
            "prompts_count": len(prompts),
            "min_recall": min_recall,
            "objective": f"maximize {PRECISION_METRIC} subject to {RECALL_METRIC} >= {min_recall}",
            "best_trial_number": best_trial.number,
            "best_trial_feasible": bool(best_trial.user_attrs.get("feasible", False)),
            "best_params": best_trial.params,
            "best_precision": _metric_value(best_result.report.metrics, PRECISION_METRIC),
            "best_recall": _metric_value(best_result.report.metrics, RECALL_METRIC),
            "best_f1": _metric_value(best_result.report.metrics, F1_METRIC),
        },
        "instance": {
            "run_dir": str(instance_dir),
            "report": str(instance_dir / config.outputs.report_filename),
            "trials_json": str(trials_path),
            "evaluation_snapshot_json": str(evaluation_snapshot_path),
        },
    }
    if dataset_snapshot_path is not None:
        report["instance"]["dataset_snapshot_json"] = str(dataset_snapshot_path)

    report_path = instance_dir / config.outputs.report_filename
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _metric_value(metrics: Mapping[str, Mapping[str, Any]], name: str) -> float:
    raw_value = metrics.get(name, {}).get("value", 0.0)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0


def _threshold_values(
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> list[float]:
    if threshold_step <= 0:
        raise ValueError("threshold_step must be positive.")
    if threshold_min > threshold_max:
        raise ValueError("threshold_min must be less than or equal to threshold_max.")
    values: list[float] = []
    current = threshold_min
    epsilon = threshold_step / 1_000_000
    while current <= threshold_max + epsilon:
        values.append(round(current, 10))
        current += threshold_step
    return values


def _build_sampler(
    optuna: Any,
    *,
    sampler_name: str,
    threshold_values: Sequence[float],
    prompts_count: int,
    seed: int | None,
) -> Any:
    normalized_name = sampler_name.lower()
    if normalized_name == "grid":
        return optuna.samplers.GridSampler(
            {
                "threshold": list(threshold_values),
                "prompt_index": list(range(prompts_count)),
            }
        )
    if normalized_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if normalized_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(
        "Unknown sampler. Supported values are: grid, tpe, random."
    )


def _resolve_trials_count(
    *,
    n_trials: int | None,
    sampler_name: str,
    threshold_values: Sequence[float],
    prompts_count: int,
) -> int:
    if n_trials is not None:
        if n_trials <= 0:
            raise ValueError("n_trials must be positive when provided.")
        return n_trials
    if sampler_name.lower() == "grid":
        return len(threshold_values) * prompts_count
    return 30


def _trial_progress_bar(
    *,
    enabled: bool,
    total: int,
    description: str,
) -> Any | None:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _SimpleProgressBar(total=total, description=description)
    return tqdm(total=total, desc=description, unit="trial")


def _progress_callback(progress_bar: Any) -> Any:
    def callback(_study: Any, trial: Any) -> None:
        progress_bar.update(1)
        postfix = {
            "precision": _format_progress_value(trial.user_attrs.get("precision")),
            "recall": _format_progress_value(trial.user_attrs.get("recall")),
            "best": _format_progress_value(_study.best_value),
        }
        if hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(postfix)
        elif hasattr(progress_bar, "set_message"):
            progress_bar.set_message(
                " ".join(f"{key}={value}" for key, value in postfix.items())
            )

    return callback


def _format_progress_value(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


class _SimpleProgressBar:
    def __init__(self, *, total: int, description: str) -> None:
        self.total = total
        self.description = description
        self.current = 0
        self.message = ""

    def update(self, value: int) -> None:
        self.current += value
        percent = 100 * self.current / self.total if self.total else 100
        suffix = f" {self.message}" if self.message else ""
        print(
            f"\r{self.description}: {self.current}/{self.total} ({percent:5.1f}%){suffix}",
            end="",
            flush=True,
        )

    def set_message(self, message: str) -> None:
        self.message = message

    def close(self) -> None:
        print()


def _best_feasible_trial(trials: Sequence[Any], min_recall: float) -> Any | None:
    feasible = [
        trial
        for trial in trials
        if trial.user_attrs.get("recall", 0.0) >= min_recall
        and trial.user_attrs.get("precision") is not None
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda trial: (
            float(trial.user_attrs.get("precision", 0.0)),
            float(trial.user_attrs.get("recall", 0.0)),
            float(trial.user_attrs.get("f1", 0.0)),
        ),
    )


def _trial_to_dict(trial: Any) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": str(trial.state),
        "value": trial.value,
        "params": dict(trial.params),
        "precision": trial.user_attrs.get("precision"),
        "recall": trial.user_attrs.get("recall"),
        "f1": trial.user_attrs.get("f1"),
        "feasible": trial.user_attrs.get("feasible"),
        "entity_description": trial.user_attrs.get("entity_description"),
        "metrics": trial.user_attrs.get("metrics"),
    }


def _random_seed(config: PipelineConfig) -> int | None:
    seed = config.dataset.params.get("random_seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "GLiNER2.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune GLiNER2 threshold and entity prompt with Optuna.",
    )
    parser.add_argument(
        "--config",
        default=str(_default_config_path()),
        help="Path to the GLiNER2 evaluation config.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Number of Optuna trials. Defaults to exhaustive grid size for --sampler grid "
            "and 30 for stochastic samplers."
        ),
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.9,
        help="Minimum recall constraint for selecting the best precision trial.",
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--sampler",
        choices=("grid", "tpe", "random"),
        default="grid",
        help=(
            "Optuna sampler. grid is exhaustive and best for the default small "
            "threshold/prompt search space."
        ),
    )
    parser.add_argument("--study-name", default="gliner2-threshold-search")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the trial-level progress bar.",
    )
    parser.add_argument(
        "--document-progress",
        action="store_true",
        help="Also show per-document prediction progress inside each trial.",
    )
    parser.add_argument("--json", action="store_true", help="Print the final report as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_pipeline_config(args.config)
    show_progress = not args.no_progress and not args.json
    report = run_threshold_search(
        config,
        n_trials=args.n_trials,
        min_recall=args.min_recall,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        study_name=args.study_name,
        storage=args.storage,
        sampler_name=args.sampler,
        show_progress=show_progress,
        show_document_progress=args.document_progress,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        optimization = report["optimization"]
        print(f"report: {report['instance']['report']}")
        print(f"trials: {report['instance']['trials_json']}")
        print(f"sampler: {optimization['sampler']}")
        print(f"best threshold: {optimization['best_params']['threshold']}")
        print(f"best precision: {optimization['best_precision']:.4f}")
        print(f"best recall: {optimization['best_recall']:.4f}")
        print(f"best feasible: {optimization['best_trial_feasible']}")
    return 0


__all__ = [
    "DEFAULT_PROMPTS",
    "run_threshold_search",
]


if __name__ == "__main__":
    raise SystemExit(main())
