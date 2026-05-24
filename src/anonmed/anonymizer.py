from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence


_POST_PROCESSING_MODES: frozenset[str] = frozenset(
    {"balanced", "conservative", "production_safe"}
)
_MASKING_STRATEGIES: frozenset[str] = frozenset({"type", "same_length"})


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    enabled: bool | None = None
    remove_disfluency: bool | None = None
    remove_punctuation: bool | None = None
    normalize_numbers: bool | None = None
    normalize_document_numbers: bool | None = None
    normalize_contacts: bool | None = None
    normalize_dates: bool | None = None
    deduplicate_repetitions: bool | None = None


@dataclass(frozen=True, slots=True)
class RuleDetectionConfig:
    enabled: bool | None = None
    pii_types: Sequence[str] | None = None


@dataclass(frozen=True, slots=True)
class MLDetectionConfig:
    enabled: bool | None = None
    labels: Sequence[str] | None = None
    model_params: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PostProcessingConfig:
    enabled: bool | None = None
    restore_non_pii: bool | None = None
    mode: str | None = None
    masking_strategy: str | None = None
    replacement_by_type: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class PIIAnonymizerConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    rules: RuleDetectionConfig = field(default_factory=RuleDetectionConfig)
    ml: MLDetectionConfig = field(default_factory=MLDetectionConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)


@dataclass(frozen=True, slots=True)
class ResolvedPreprocessingConfig:
    enabled: bool
    remove_disfluency: bool
    remove_punctuation: bool
    normalize_numbers: bool
    normalize_document_numbers: bool
    normalize_contacts: bool
    normalize_dates: bool
    deduplicate_repetitions: bool


@dataclass(frozen=True, slots=True)
class ResolvedRuleDetectionConfig:
    enabled: bool
    pii_types: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ResolvedMLDetectionConfig:
    enabled: bool
    labels: tuple[str, ...] | None
    model_params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ResolvedPostProcessingConfig:
    enabled: bool
    restore_non_pii: bool
    mode: str
    masking_strategy: str
    replacement_by_type: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ResolvedPIIAnonymizerConfig:
    preprocessing: ResolvedPreprocessingConfig
    rules: ResolvedRuleDetectionConfig
    ml: ResolvedMLDetectionConfig
    postprocessing: ResolvedPostProcessingConfig


@dataclass(frozen=True, slots=True)
class PIIAnonymizationResult:
    original_text: str
    preprocessed_text: str
    masked_text: str
    masked_preprocessed_text: str
    masked_original_text: str
    candidates: tuple[Any, ...]
    rule_candidates: tuple[Any, ...]
    ml_candidates: tuple[Any, ...]
    postprocessed_mentions: tuple[Any, ...]
    entity_groups: tuple[Any, ...]
    preprocessing_result: Any | None
    postprocessing_result: Any | None
    config: ResolvedPIIAnonymizerConfig
    warnings: tuple[str, ...] = ()

    def to_dict(self, *, include_debug: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "original_text": self.original_text,
            "preprocessed_text": self.preprocessed_text,
            "masked_text": self.masked_text,
            "masked_preprocessed_text": self.masked_preprocessed_text,
            "masked_original_text": self.masked_original_text,
            "config": asdict(self.config),
            "warnings": list(self.warnings),
        }
        if include_debug:
            payload.update(
                {
                    "candidates": [asdict(candidate) for candidate in self.candidates],
                    "rule_candidates": [asdict(candidate) for candidate in self.rule_candidates],
                    "ml_candidates": [asdict(candidate) for candidate in self.ml_candidates],
                    "postprocessed_mentions": [
                        asdict(mention) for mention in self.postprocessed_mentions
                    ],
                    "entity_groups": [asdict(group) for group in self.entity_groups],
                }
            )
        return payload


class PIIAnonymizer:
    def __init__(
        self,
        ml_model: object | None = None,
        *,
        device: str | None = None,
        default_config: PIIAnonymizerConfig | None = None,
        lazy_imports: bool = True,
        flags: Mapping[str, object] | None = None,
        **default_overrides: object,
    ) -> None:
        self._ml_model: object | None = ml_model
        self._device: str | None = device
        self._default_config: PIIAnonymizerConfig | None = default_config
        self._lazy_imports: bool = lazy_imports
        self._default_overrides: dict[str, object] = _merge_override_mappings(
            flags,
            default_overrides,
        )
        _validate_overrides(self._default_overrides)
        self._preprocessing_pipelines: dict[
            ResolvedPreprocessingConfig,
            object,
        ] = {}
        self._ml_runner: object | None = None
        self._ml_runner_key: tuple[tuple[str, str], ...] | None = None

        if not lazy_imports:
            self._eager_import_lightweight_components()

    def __call__(
        self,
        text: str,
        *,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        use_preprocessing: bool | None = None,
        use_rules: bool | None = None,
        use_regulars: bool | None = None,
        use_ml: bool | None = None,
        use_postprocessing: bool | None = None,
        remove_disfluency: bool | None = None,
        remove_punctuation: bool | None = None,
        normalize_numbers: bool | None = None,
        normalize_document_numbers: bool | None = None,
        normalize_contacts: bool | None = None,
        normalize_dates: bool | None = None,
        deduplicate_repetitions: bool | None = None,
        restore_non_pii: bool | None = None,
        pii_types: Sequence[str] | None = None,
        ml_labels: Sequence[str] | None = None,
        masking_strategy: str | None = None,
        post_processing_mode: str | None = None,
        replacement_by_type: Mapping[str, str] | None = None,
    ) -> PIIAnonymizationResult:
        overrides: dict[str, object] = _drop_none(
            {
                "use_preprocessing": use_preprocessing,
                "use_rules": use_rules,
                "use_regulars": use_regulars,
                "use_ml": use_ml,
                "use_postprocessing": use_postprocessing,
                "remove_disfluency": remove_disfluency,
                "remove_punctuation": remove_punctuation,
                "normalize_numbers": normalize_numbers,
                "normalize_document_numbers": normalize_document_numbers,
                "normalize_contacts": normalize_contacts,
                "normalize_dates": normalize_dates,
                "deduplicate_repetitions": deduplicate_repetitions,
                "restore_non_pii": restore_non_pii,
                "pii_types": pii_types,
                "ml_labels": ml_labels,
                "masking_strategy": masking_strategy,
                "post_processing_mode": post_processing_mode,
                "replacement_by_type": replacement_by_type,
            }
        )
        return self._run(text, config=config, flags=flags, overrides=overrides)

    def anonymize(
        self,
        text: str,
        *,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        **overrides: object,
    ) -> PIIAnonymizationResult:
        return self._run(text, config=config, flags=flags, overrides=overrides)

    def detect_pii(
        self,
        text: str,
        *,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        **overrides: object,
    ) -> PIIAnonymizationResult:
        return self._run(text, config=config, flags=flags, overrides=overrides)

    def preprocess(
        self,
        text: str,
        *,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        enabled: bool | None = None,
        remove_disfluency: bool | None = None,
        remove_punctuation: bool | None = None,
        normalize_numbers: bool | None = None,
        normalize_document_numbers: bool | None = None,
        normalize_contacts: bool | None = None,
        normalize_dates: bool | None = None,
        deduplicate_repetitions: bool | None = None,
    ) -> Any:
        overrides: dict[str, object] = _drop_none(
            {
                "use_preprocessing": enabled,
                "remove_disfluency": remove_disfluency,
                "remove_punctuation": remove_punctuation,
                "normalize_numbers": normalize_numbers,
                "normalize_document_numbers": normalize_document_numbers,
                "normalize_contacts": normalize_contacts,
                "normalize_dates": normalize_dates,
                "deduplicate_repetitions": deduplicate_repetitions,
            }
        )
        resolved: ResolvedPIIAnonymizerConfig = self._resolve_config(
            config=config,
            flags=flags,
            overrides=overrides,
        )
        return self._run_preprocessing(text, resolved.preprocessing)

    def detect_by_rules(
        self,
        text: str,
        *,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        enabled: bool | None = None,
        pii_types: Sequence[str] | None = None,
        rules: Sequence[object] | None = None,
    ) -> tuple[Any, ...]:
        overrides: dict[str, object] = _drop_none(
            {
                "use_rules": enabled,
                "pii_types": pii_types,
            }
        )
        resolved: ResolvedPIIAnonymizerConfig = self._resolve_config(
            config=config,
            flags=flags,
            overrides=overrides,
        )
        return self._detect_by_rules(text, resolved.rules, rules=rules)

    def detect_by_ml(
        self,
        text: str,
        *,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        enabled: bool | None = None,
        labels: Sequence[str] | None = None,
    ) -> tuple[Any, ...]:
        overrides: dict[str, object] = _drop_none(
            {
                "use_ml": enabled,
                "ml_labels": labels,
            }
        )
        resolved: ResolvedPIIAnonymizerConfig = self._resolve_config(
            config=config,
            flags=flags,
            overrides=overrides,
        )
        return self._detect_by_ml(text, resolved.ml)

    def merge_candidates(
        self,
        text: str,
        candidates: Sequence[Any],
        *,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        mode: str | None = None,
    ) -> tuple[Any, ...]:
        del text
        overrides: dict[str, object] = _drop_none({"post_processing_mode": mode})
        resolved: ResolvedPIIAnonymizerConfig = self._resolve_config(
            config=config,
            flags=flags,
            overrides=overrides,
        )
        return self._merge_candidates(candidates, mode=resolved.postprocessing.mode)

    def postprocess(
        self,
        *,
        original_text: str,
        preprocessed_text: str,
        candidates: Sequence[Any],
        preprocessing_result: Any | None = None,
        config: PIIAnonymizerConfig | None = None,
        flags: Mapping[str, object] | None = None,
        enabled: bool | None = None,
        restore_non_pii: bool | None = None,
        masking_strategy: str | None = None,
        post_processing_mode: str | None = None,
        replacement_by_type: Mapping[str, str] | None = None,
    ) -> Any | None:
        overrides: dict[str, object] = _drop_none(
            {
                "use_postprocessing": enabled,
                "restore_non_pii": restore_non_pii,
                "masking_strategy": masking_strategy,
                "post_processing_mode": post_processing_mode,
                "replacement_by_type": replacement_by_type,
            }
        )
        resolved: ResolvedPIIAnonymizerConfig = self._resolve_config(
            config=config,
            flags=flags,
            overrides=overrides,
        )
        if not resolved.postprocessing.enabled:
            return None
        return self._postprocess(
            original_text=original_text,
            preprocessed_text=preprocessed_text,
            candidates=candidates,
            preprocessing_result=preprocessing_result,
            config=resolved.postprocessing,
        )

    def _run(
        self,
        text: str,
        *,
        config: PIIAnonymizerConfig | None,
        flags: Mapping[str, object] | None,
        overrides: Mapping[str, object],
    ) -> PIIAnonymizationResult:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        resolved: ResolvedPIIAnonymizerConfig = self._resolve_config(
            config=config,
            flags=flags,
            overrides=overrides,
        )
        preprocessing_result: Any = self._run_preprocessing(text, resolved.preprocessing)
        preprocessed_text: str = preprocessing_result.normalized_text

        rule_candidates: tuple[Any, ...] = self._detect_by_rules(
            preprocessed_text,
            resolved.rules,
        )
        ml_candidates: tuple[Any, ...] = self._detect_by_ml(preprocessed_text, resolved.ml)
        merged_candidates: tuple[Any, ...] = self._merge_candidates(
            rule_candidates + ml_candidates,
            mode=resolved.postprocessing.mode,
        )

        warnings: list[str] = []
        postprocessing_result: Any | None = None
        postprocessed_mentions: tuple[Any, ...] = ()
        entity_groups: tuple[Any, ...] = ()

        if resolved.postprocessing.enabled:
            postprocessing_result = self._postprocess(
                original_text=text,
                preprocessed_text=preprocessed_text,
                candidates=merged_candidates,
                preprocessing_result=preprocessing_result,
                config=resolved.postprocessing,
            )
            masked_preprocessed_text = postprocessing_result.masked_normalized_text
            masked_original_text = postprocessing_result.masked_original_text
            postprocessed_mentions = postprocessing_result.mentions
            entity_groups = postprocessing_result.entity_groups
        else:
            masked_preprocessed_text = _mask_candidates(
                preprocessed_text,
                merged_candidates,
                replacement_by_type=resolved.postprocessing.replacement_by_type,
                masking_strategy=resolved.postprocessing.masking_strategy,
            )
            masked_original_text = (
                masked_preprocessed_text if preprocessed_text == text else text
            )
            if preprocessed_text != text:
                warnings.append(
                    "postprocessing is disabled, so masked_original_text cannot be restored"
                )

        masked_text: str = (
            masked_original_text
            if resolved.postprocessing.restore_non_pii and resolved.postprocessing.enabled
            else masked_preprocessed_text
        )
        return PIIAnonymizationResult(
            original_text=text,
            preprocessed_text=preprocessed_text,
            masked_text=masked_text,
            masked_preprocessed_text=masked_preprocessed_text,
            masked_original_text=masked_original_text,
            candidates=merged_candidates,
            rule_candidates=rule_candidates,
            ml_candidates=ml_candidates,
            postprocessed_mentions=postprocessed_mentions,
            entity_groups=entity_groups,
            preprocessing_result=preprocessing_result,
            postprocessing_result=postprocessing_result,
            config=resolved,
            warnings=tuple(warnings),
        )

    def _resolve_config(
        self,
        *,
        config: PIIAnonymizerConfig | None,
        flags: Mapping[str, object] | None,
        overrides: Mapping[str, object],
    ) -> ResolvedPIIAnonymizerConfig:
        resolved: ResolvedPIIAnonymizerConfig = _builtin_defaults(
            ml_enabled=self._ml_model is not None
        )
        if self._default_config is not None:
            resolved = _apply_config(resolved, self._default_config)
        if self._default_overrides:
            resolved = _apply_overrides(resolved, self._default_overrides)
        if config is not None:
            resolved = _apply_config(resolved, config)
        merged_call_overrides: dict[str, object] = _merge_override_mappings(flags, overrides)
        if merged_call_overrides:
            resolved = _apply_overrides(resolved, merged_call_overrides)
        return resolved

    def _run_preprocessing(self, text: str, config: ResolvedPreprocessingConfig) -> Any:
        if not config.enabled:
            return _identity_preprocessing_result(text)

        pipeline: object | None = self._preprocessing_pipelines.get(config)
        if pipeline is None:
            from anonmed.preprocessing import ASRNormalizationPipeline

            pipeline = ASRNormalizationPipeline(
                remove_disfluencies=config.remove_disfluency,
                remove_punctuation=config.remove_punctuation,
                normalize_numbers=config.normalize_numbers,
                normalize_document_numbers=config.normalize_document_numbers,
                normalize_date_birth=config.normalize_dates,
                normalize_contacts=config.normalize_contacts,
                deduplicate_repetitions=config.deduplicate_repetitions,
            )
            self._preprocessing_pipelines[config] = pipeline
        return pipeline.run(text)

    def _detect_by_rules(
        self,
        text: str,
        config: ResolvedRuleDetectionConfig,
        *,
        rules: Sequence[object] | None = None,
    ) -> tuple[Any, ...]:
        if not config.enabled:
            return ()

        from anonmed.anonymization import candidate_from_numeric_match, collect_numeric_pii_candidates

        matches: tuple[Any, ...] = collect_numeric_pii_candidates(text, rules=rules)
        if config.pii_types is not None:
            allowed_types: frozenset[str] = frozenset(config.pii_types)
            matches = tuple(match for match in matches if match.pii_type in allowed_types)
        return tuple(candidate_from_numeric_match(match, source="regex") for match in matches)

    def _detect_by_ml(self, text: str, config: ResolvedMLDetectionConfig) -> tuple[Any, ...]:
        if not config.enabled:
            return ()
        runner: Any = self._get_ml_runner(config)
        result: Any = runner.run(text)
        labels: frozenset[str] | None = None
        if config.labels is not None:
            labels = frozenset(config.labels)
        return tuple(
            _candidate_from_ml_span(text=text, span=span)
            for span in result.spans
            if labels is None or span.label in labels
        )

    def _merge_candidates(self, candidates: Sequence[Any], *, mode: str) -> tuple[Any, ...]:
        from anonmed.anonymization import resolve_pii_candidates

        return resolve_pii_candidates(candidates, mode=mode)

    def _postprocess(
        self,
        *,
        original_text: str,
        preprocessed_text: str,
        candidates: Sequence[Any],
        preprocessing_result: Any | None,
        config: ResolvedPostProcessingConfig,
    ) -> Any:
        from anonmed.anonymization import run_pii_post_processing

        alignment: Any = getattr(preprocessing_result, "normalized_to_original_alignment", None)
        if alignment is None:
            from anonmed.preprocessing.asr.alignment import align_texts_by_diff

            alignment = align_texts_by_diff(original_text, preprocessed_text)

        return run_pii_post_processing(
            original_text=original_text,
            normalized_text=preprocessed_text,
            alignment=alignment,
            candidates=candidates,
            replacement_by_type=config.replacement_by_type,
            mode=config.mode,
            masking_strategy=config.masking_strategy,
        )

    def _get_ml_runner(self, config: ResolvedMLDetectionConfig) -> Any:
        if self._ml_model is None:
            raise ValueError("ml_model must be provided when ML detection is enabled")
        runner_key: tuple[tuple[str, str], ...] = tuple(
            sorted((str(key), repr(value)) for key, value in config.model_params.items())
        )
        if self._ml_runner is None or self._ml_runner_key != runner_key:
            from anonmed.ml.pipelines.runner import ModelRunner

            runner_kwargs: dict[str, object] = {}
            if self._device is not None:
                runner_kwargs["device"] = self._device
            if config.model_params:
                runner_kwargs["model_params"] = dict(config.model_params)
            self._ml_runner = ModelRunner(model=self._ml_model, **runner_kwargs)
            self._ml_runner_key = runner_key
        return self._ml_runner

    def _eager_import_lightweight_components(self) -> None:
        import anonmed.anonymization  # noqa: F401
        import anonmed.preprocessing  # noqa: F401


def anonymize_pii(
    text: str,
    *,
    ml_model: object | None = None,
    device: str | None = None,
    config: PIIAnonymizerConfig | None = None,
    flags: Mapping[str, object] | None = None,
    **overrides: object,
) -> PIIAnonymizationResult:
    anonymizer = PIIAnonymizer(ml_model=ml_model, device=device)
    return anonymizer.anonymize(text, config=config, flags=flags, **overrides)


def anonymize(
    text: str,
    *,
    ml_model: object | None = None,
    device: str | None = None,
    config: PIIAnonymizerConfig | None = None,
    flags: Mapping[str, object] | None = None,
    **overrides: object,
) -> PIIAnonymizationResult:
    return anonymize_pii(
        text,
        ml_model=ml_model,
        device=device,
        config=config,
        flags=flags,
        **overrides,
    )


def _builtin_defaults(*, ml_enabled: bool) -> ResolvedPIIAnonymizerConfig:
    return ResolvedPIIAnonymizerConfig(
        preprocessing=ResolvedPreprocessingConfig(
            enabled=True,
            remove_disfluency=True,
            remove_punctuation=True,
            normalize_numbers=True,
            normalize_document_numbers=False,
            normalize_contacts=True,
            normalize_dates=True,
            deduplicate_repetitions=False,
        ),
        rules=ResolvedRuleDetectionConfig(
            enabled=True,
            pii_types=None,
        ),
        ml=ResolvedMLDetectionConfig(
            enabled=ml_enabled,
            labels=None,
            model_params={},
        ),
        postprocessing=ResolvedPostProcessingConfig(
            enabled=True,
            restore_non_pii=True,
            mode="production_safe",
            masking_strategy="type",
            replacement_by_type={},
        ),
    )


def _apply_config(
    resolved: ResolvedPIIAnonymizerConfig,
    config: PIIAnonymizerConfig,
) -> ResolvedPIIAnonymizerConfig:
    return replace(
        resolved,
        preprocessing=_apply_preprocessing_config(resolved.preprocessing, config.preprocessing),
        rules=_apply_rules_config(resolved.rules, config.rules),
        ml=_apply_ml_config(resolved.ml, config.ml),
        postprocessing=_apply_postprocessing_config(
            resolved.postprocessing,
            config.postprocessing,
        ),
    )


def _apply_preprocessing_config(
    resolved: ResolvedPreprocessingConfig,
    config: PreprocessingConfig,
) -> ResolvedPreprocessingConfig:
    return replace(
        resolved,
        **_drop_none(
            {
                "enabled": config.enabled,
                "remove_disfluency": config.remove_disfluency,
                "remove_punctuation": config.remove_punctuation,
                "normalize_numbers": config.normalize_numbers,
                "normalize_document_numbers": config.normalize_document_numbers,
                "normalize_contacts": config.normalize_contacts,
                "normalize_dates": config.normalize_dates,
                "deduplicate_repetitions": config.deduplicate_repetitions,
            }
        ),
    )


def _apply_rules_config(
    resolved: ResolvedRuleDetectionConfig,
    config: RuleDetectionConfig,
) -> ResolvedRuleDetectionConfig:
    values: dict[str, object] = _drop_none(
        {
            "enabled": config.enabled,
            "pii_types": _string_tuple_or_none(config.pii_types, "rules.pii_types"),
        }
    )
    return replace(resolved, **values)


def _apply_ml_config(
    resolved: ResolvedMLDetectionConfig,
    config: MLDetectionConfig,
) -> ResolvedMLDetectionConfig:
    values: dict[str, object] = _drop_none(
        {
            "enabled": config.enabled,
            "labels": _string_tuple_or_none(config.labels, "ml.labels"),
            "model_params": _mapping_or_none(config.model_params, "ml.model_params"),
        }
    )
    return replace(resolved, **values)


def _apply_postprocessing_config(
    resolved: ResolvedPostProcessingConfig,
    config: PostProcessingConfig,
) -> ResolvedPostProcessingConfig:
    values: dict[str, object] = _drop_none(
        {
            "enabled": config.enabled,
            "restore_non_pii": config.restore_non_pii,
            "mode": _post_processing_mode_or_none(config.mode),
            "masking_strategy": _masking_strategy_or_none(config.masking_strategy),
            "replacement_by_type": _replacement_mapping_or_none(config.replacement_by_type),
        }
    )
    return replace(resolved, **values)


def _apply_overrides(
    resolved: ResolvedPIIAnonymizerConfig,
    overrides: Mapping[str, object],
) -> ResolvedPIIAnonymizerConfig:
    _validate_overrides(overrides)
    preprocessing_values: dict[str, object] = {}
    rules_values: dict[str, object] = {}
    ml_values: dict[str, object] = {}
    postprocessing_values: dict[str, object] = {}

    for key, value in overrides.items():
        if value is None:
            continue
        if key == "use_preprocessing":
            preprocessing_values["enabled"] = _bool(value, key)
        elif key in {"use_rules", "use_regulars"}:
            rules_values["enabled"] = _bool(value, key)
        elif key == "use_ml":
            ml_values["enabled"] = _bool(value, key)
        elif key == "use_postprocessing":
            postprocessing_values["enabled"] = _bool(value, key)
        elif key in {"remove_disfluency", "remove_disfluencies"}:
            preprocessing_values["remove_disfluency"] = _bool(value, key)
        elif key == "remove_punctuation":
            preprocessing_values["remove_punctuation"] = _bool(value, key)
        elif key == "normalize_numbers":
            preprocessing_values["normalize_numbers"] = _bool(value, key)
        elif key == "normalize_document_numbers":
            preprocessing_values["normalize_document_numbers"] = _bool(value, key)
        elif key == "normalize_contacts":
            preprocessing_values["normalize_contacts"] = _bool(value, key)
        elif key in {"normalize_dates", "normalize_date_birth"}:
            preprocessing_values["normalize_dates"] = _bool(value, key)
        elif key == "deduplicate_repetitions":
            preprocessing_values["deduplicate_repetitions"] = _bool(value, key)
        elif key == "pii_types":
            rules_values["pii_types"] = _string_tuple_or_none(value, key)
        elif key in {"ml_labels", "labels"}:
            ml_values["labels"] = _string_tuple_or_none(value, key)
        elif key == "model_params":
            ml_values["model_params"] = _mapping_or_none(value, key)
        elif key == "restore_non_pii":
            postprocessing_values["restore_non_pii"] = _bool(value, key)
        elif key in {"post_processing_mode", "mode"}:
            postprocessing_values["mode"] = _post_processing_mode(value, key)
        elif key == "masking_strategy":
            postprocessing_values["masking_strategy"] = _masking_strategy(value, key)
        elif key == "replacement_by_type":
            postprocessing_values["replacement_by_type"] = _replacement_mapping(value)

    return replace(
        resolved,
        preprocessing=replace(resolved.preprocessing, **preprocessing_values),
        rules=replace(resolved.rules, **rules_values),
        ml=replace(resolved.ml, **ml_values),
        postprocessing=replace(resolved.postprocessing, **postprocessing_values),
    )


def _identity_preprocessing_result(text: str) -> Any:
    from anonmed.preprocessing import ASRNormalizationResult
    from anonmed.preprocessing.asr.alignment import align_texts_by_diff

    return ASRNormalizationResult(
        original_text=text,
        disfluency_cleaned_text=text,
        punctuation_cleaned_text=text,
        repetition_cleaned_text=text,
        cleaned_text=text,
        normalized_text=text,
        numeric_normalized_text=text,
        document_number_normalized_text=text,
        date_birth_normalized_text=text,
        contact_normalized_text=text,
        normalized_to_original_alignment=align_texts_by_diff(text, text),
    )


def _candidate_from_ml_span(*, text: str, span: Any) -> Any:
    from anonmed.anonymization import PIICandidate

    label: str = str(span.label)
    start: int = int(span.begin)
    end: int = int(span.end)
    value: str = text[start:end]
    return PIICandidate(
        entity_type=label,
        source="ml",
        source_score=_score_from_span(span),
        start=start,
        end=end,
        value=value,
        normalized_value=value,
        rule_id=f"ml:{label}",
        context=_context_window(text, start, end),
        metadata={
            "label": label,
            "line_idx": int(span.line_idx),
            "data": str(span.data),
        },
        validators={},
        negative_context_hits=(),
        sensitivity_rank=_ml_sensitivity_rank(label),
    )


def _score_from_span(span: Any) -> float:
    raw_score: object | None = getattr(span, "score", None)
    if isinstance(raw_score, (int, float)):
        return min(0.99, max(0.01, float(raw_score)))
    return 0.86


def _ml_sensitivity_rank(label: str) -> int:
    return {
        "PHONE": 100,
        "SNILS": 99,
        "PASSPORT": 98,
        "OMS": 97,
        "INN": 96,
        "EMAIL": 93,
        "DATE_BIRTH": 90,
        "PER": 84,
        "PERSON": 84,
        "NAME": 84,
        "ADDRESS": 72,
        "AGE": 30,
    }.get(label.upper(), 50)


def _context_window(text: str, start: int, end: int, *, window: int = 72) -> str:
    return text[max(0, start - window) : min(len(text), end + window)]


def _mask_candidates(
    text: str,
    candidates: Sequence[Any],
    *,
    replacement_by_type: Mapping[str, str],
    masking_strategy: str,
) -> str:
    parts: list[str] = []
    cursor: int = 0
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
        if candidate.start < cursor or candidate.start >= candidate.end:
            continue
        parts.append(text[cursor : candidate.start])
        replacement: str | None = replacement_by_type.get(candidate.entity_type)
        if replacement is None:
            if masking_strategy == "same_length":
                replacement = "*" * max(1, candidate.end - candidate.start)
            else:
                replacement = f"[{candidate.entity_type}]"
        parts.append(replacement)
        cursor = candidate.end
    parts.append(text[cursor:])
    return "".join(parts)


def _merge_override_mappings(
    first: Mapping[str, object] | None,
    second: Mapping[str, object] | None,
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if first is not None:
        merged.update(dict(first))
    if second is not None:
        merged.update(dict(second))
    return {key: value for key, value in merged.items() if value is not None}


def _drop_none(values: Mapping[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}


def _validate_overrides(overrides: Mapping[str, object]) -> None:
    allowed_keys: frozenset[str] = frozenset(
        {
            "use_preprocessing",
            "use_rules",
            "use_regulars",
            "use_ml",
            "use_postprocessing",
            "remove_disfluency",
            "remove_disfluencies",
            "remove_punctuation",
            "normalize_numbers",
            "normalize_document_numbers",
            "normalize_contacts",
            "normalize_dates",
            "normalize_date_birth",
            "deduplicate_repetitions",
            "pii_types",
            "ml_labels",
            "labels",
            "model_params",
            "restore_non_pii",
            "post_processing_mode",
            "mode",
            "masking_strategy",
            "replacement_by_type",
        }
    )
    unknown_keys: set[str] = set(overrides) - allowed_keys
    if unknown_keys:
        formatted_keys: str = ", ".join(sorted(unknown_keys))
        raise TypeError(f"Unknown PIIAnonymizer flag(s): {formatted_keys}")


def _bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool, got {type(value).__name__}")


def _string_tuple_or_none(value: object, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items: tuple[str, ...] = tuple(item.strip() for item in value.split(",") if item.strip())
        return items or None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    raise TypeError(f"{name} must be a string or sequence of strings")


def _mapping_or_none(value: object, name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{name} must be a mapping")


def _replacement_mapping_or_none(value: object) -> Mapping[str, str] | None:
    if value is None:
        return None
    return _replacement_mapping(value)


def _replacement_mapping(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("replacement_by_type must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def _post_processing_mode_or_none(value: object) -> str | None:
    if value is None:
        return None
    return _post_processing_mode(value, "postprocessing.mode")


def _post_processing_mode(value: object, name: str) -> str:
    mode: str = str(value)
    if mode not in _POST_PROCESSING_MODES:
        choices: str = ", ".join(sorted(_POST_PROCESSING_MODES))
        raise ValueError(f"{name} must be one of: {choices}")
    return mode


def _masking_strategy_or_none(value: object) -> str | None:
    if value is None:
        return None
    return _masking_strategy(value, "postprocessing.masking_strategy")


def _masking_strategy(value: object, name: str) -> str:
    strategy: str = str(value)
    if strategy not in _MASKING_STRATEGIES:
        choices: str = ", ".join(sorted(_MASKING_STRATEGIES))
        raise ValueError(f"{name} must be one of: {choices}")
    return strategy


__all__: list[str] = [
    "MLDetectionConfig",
    "PIIAnonymizationResult",
    "PIIAnonymizer",
    "PIIAnonymizerConfig",
    "PostProcessingConfig",
    "PreprocessingConfig",
    "ResolvedMLDetectionConfig",
    "ResolvedPIIAnonymizerConfig",
    "ResolvedPostProcessingConfig",
    "ResolvedPreprocessingConfig",
    "ResolvedRuleDetectionConfig",
    "RuleDetectionConfig",
    "anonymize",
    "anonymize_pii",
]
