from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

from anonmed.preprocessing.asr.disfluency import (
    CleanedText,
    DisfluencyFilter,
    DisfluencyFilterConfig,
    RemovedSpan,
)
from anonmed.preprocessing.asr.number_extractor import IntegerExtractor
from anonmed.preprocessing.asr.punctuation import (
    ProtectedPunctuationSpan,
    PunctuationCleanedText,
    PunctuationRemovalConfig,
    PunctuationRemover,
    RemovedPunctuationSpan,
)
from anonmed.preprocessing.asr.types import ExtractorConfig, IntegerSpan


@dataclass(frozen=True, slots=True)
class ASRNormalizationResult:
    original_text: str
    disfluency_cleaned_text: str
    punctuation_cleaned_text: str
    cleaned_text: str
    normalized_text: str
    removed_spans: tuple[RemovedSpan, ...] = field(default_factory=tuple)
    punctuation_removed_spans: tuple[RemovedPunctuationSpan, ...] = field(default_factory=tuple)
    punctuation_protected_spans: tuple[ProtectedPunctuationSpan, ...] = field(default_factory=tuple)
    integer_spans: tuple[IntegerSpan, ...] = field(default_factory=tuple)


class ASRNormalizationPipeline:
    def __init__(
        self,
        *,
        disfluency_config: DisfluencyFilterConfig | None = None,
        punctuation_config: PunctuationRemovalConfig | None = None,
        extractor_config: ExtractorConfig | None = None,
        remove_disfluencies: bool = True,
        remove_punctuation: bool = True,
    ) -> None:
        self.remove_disfluencies: bool = remove_disfluencies
        self.remove_punctuation: bool = remove_punctuation
        self.disfluency_filter = DisfluencyFilter(config=disfluency_config)
        self.punctuation_remover = PunctuationRemover(config=punctuation_config)
        self.integer_extractor = IntegerExtractor(config=extractor_config)

    def run(self, text: str) -> ASRNormalizationResult:
        disfluency_cleaned: CleanedText
        if self.remove_disfluencies:
            disfluency_cleaned = self.disfluency_filter.clean(text)
        else:
            disfluency_cleaned = CleanedText(original_text=text, text=text)

        punctuation_cleaned: PunctuationCleanedText
        if self.remove_punctuation:
            punctuation_cleaned = self.punctuation_remover.clean(disfluency_cleaned.text)
        else:
            punctuation_cleaned = PunctuationCleanedText(
                original_text=disfluency_cleaned.text,
                text=disfluency_cleaned.text,
            )

        cleaned_text: str = punctuation_cleaned.text
        integer_spans: list[IntegerSpan] = self.integer_extractor.extract(cleaned_text)
        normalized_text: str = self.integer_extractor.replace(cleaned_text)
        result = ASRNormalizationResult(
            original_text=text,
            disfluency_cleaned_text=disfluency_cleaned.text,
            punctuation_cleaned_text=punctuation_cleaned.text,
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            removed_spans=disfluency_cleaned.removed_spans,
            punctuation_removed_spans=punctuation_cleaned.removed_spans,
            punctuation_protected_spans=punctuation_cleaned.protected_spans,
            integer_spans=tuple(integer_spans),
        )
        return result

    def to_json(self, text: str, *, ensure_ascii: bool = False) -> str:
        result: ASRNormalizationResult = self.run(text)
        payload: dict[str, object] = asdict(result)
        serialized: str = json.dumps(payload, ensure_ascii=ensure_ascii, indent=2)
        return serialized


def run_asr_normalization(
    text: str,
    *,
    disfluency_config: DisfluencyFilterConfig | None = None,
    punctuation_config: PunctuationRemovalConfig | None = None,
    extractor_config: ExtractorConfig | None = None,
    remove_disfluencies: bool = True,
    remove_punctuation: bool = True,
) -> ASRNormalizationResult:
    pipeline = ASRNormalizationPipeline(
        disfluency_config=disfluency_config,
        punctuation_config=punctuation_config,
        extractor_config=extractor_config,
        remove_disfluencies=remove_disfluencies,
        remove_punctuation=remove_punctuation,
    )
    result: ASRNormalizationResult = pipeline.run(text)
    return result


__all__: list[str] = [
    "ASRNormalizationPipeline",
    "ASRNormalizationResult",
    "run_asr_normalization",
]
