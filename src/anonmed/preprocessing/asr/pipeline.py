from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

from anonmed.preprocessing.asr.alignment import (
    TextAlignment,
    align_texts_by_diff,
    build_replacement_alignment,
    compose_alignments,
)
from anonmed.preprocessing.asr.contacts import (
    ContactNormalizedText,
    ContactNormalizer,
    ContactNormalizerConfig,
    ContactSpan,
)
from anonmed.preprocessing.asr.date_birth import (
    DateBirthNormalizedText,
    DateBirthNormalizer,
    DateBirthNormalizerConfig,
    DateBirthSpan,
)
from anonmed.preprocessing.asr.document_numbers import (
    DocumentNumberNormalizedText,
    DocumentNumberNormalizer,
    DocumentNumberNormalizerConfig,
    DocumentNumberSpan,
)
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
from anonmed.preprocessing.asr.repetition import (
    ASRRepeatDeduplicationConfig,
    ASRRepeatDeduplicationResult,
    deduplicate_asr_utterances,
)
from anonmed.preprocessing.asr.types import ExtractorConfig, IntegerSpan


@dataclass(frozen=True, slots=True)
class ASRNormalizationResult:
    original_text: str
    disfluency_cleaned_text: str
    punctuation_cleaned_text: str
    repetition_cleaned_text: str
    cleaned_text: str
    normalized_text: str
    numeric_normalized_text: str = ""
    document_number_normalized_text: str = ""
    date_birth_normalized_text: str = ""
    contact_normalized_text: str = ""
    removed_spans: tuple[RemovedSpan, ...] = field(default_factory=tuple)
    punctuation_removed_spans: tuple[RemovedPunctuationSpan, ...] = field(default_factory=tuple)
    punctuation_protected_spans: tuple[ProtectedPunctuationSpan, ...] = field(default_factory=tuple)
    repetition_suppressed_indexes: tuple[int, ...] = field(default_factory=tuple)
    integer_spans: tuple[IntegerSpan, ...] = field(default_factory=tuple)
    document_number_spans: tuple[DocumentNumberSpan, ...] = field(default_factory=tuple)
    date_birth_spans: tuple[DateBirthSpan, ...] = field(default_factory=tuple)
    contact_spans: tuple[ContactSpan, ...] = field(default_factory=tuple)
    normalized_to_original_alignment: TextAlignment | None = None


class ASRNormalizationPipeline:
    def __init__(
        self,
        *,
        disfluency_config: DisfluencyFilterConfig | None = None,
        punctuation_config: PunctuationRemovalConfig | None = None,
        extractor_config: ExtractorConfig | None = None,
        repetition_config: ASRRepeatDeduplicationConfig | None = None,
        document_number_config: DocumentNumberNormalizerConfig | None = None,
        date_birth_config: DateBirthNormalizerConfig | None = None,
        contact_config: ContactNormalizerConfig | None = None,
        remove_disfluencies: bool = True,
        remove_punctuation: bool = True,
        deduplicate_repetitions: bool = False,
        normalize_document_numbers: bool = False,
        normalize_date_birth: bool = True,
        normalize_contacts: bool = True,
    ) -> None:
        self.remove_disfluencies: bool = remove_disfluencies
        self.remove_punctuation: bool = remove_punctuation
        self.deduplicate_repetitions: bool = deduplicate_repetitions
        self.normalize_document_numbers: bool = normalize_document_numbers
        self.normalize_date_birth: bool = normalize_date_birth
        self.normalize_contacts: bool = normalize_contacts
        self.repetition_config: ASRRepeatDeduplicationConfig | None = repetition_config
        self.disfluency_filter = DisfluencyFilter(config=disfluency_config)
        self.punctuation_remover = PunctuationRemover(config=punctuation_config)
        self.integer_extractor = IntegerExtractor(config=extractor_config)
        self.document_number_normalizer = DocumentNumberNormalizer(config=document_number_config)
        self.date_birth_normalizer = DateBirthNormalizer(config=date_birth_config)
        self.contact_normalizer = ContactNormalizer(config=contact_config)

    def run(self, text: str) -> ASRNormalizationResult:
        disfluency_cleaned: CleanedText
        if self.remove_disfluencies:
            disfluency_cleaned = self.disfluency_filter.clean(text)
        else:
            disfluency_cleaned = CleanedText(original_text=text, text=text)
        original_to_disfluency_alignment: TextAlignment = align_texts_by_diff(
            text,
            disfluency_cleaned.text,
        )

        punctuation_cleaned: PunctuationCleanedText
        if self.remove_punctuation:
            punctuation_cleaned = self.punctuation_remover.clean(disfluency_cleaned.text)
        else:
            punctuation_cleaned = PunctuationCleanedText(
                original_text=disfluency_cleaned.text,
                text=disfluency_cleaned.text,
            )
        disfluency_to_punctuation_alignment: TextAlignment = align_texts_by_diff(
            disfluency_cleaned.text,
            punctuation_cleaned.text,
        )
        original_to_punctuation_alignment: TextAlignment = compose_alignments(
            original_to_disfluency_alignment,
            disfluency_to_punctuation_alignment,
        )

        repetition_cleaned_text: str = punctuation_cleaned.text
        repetition_suppressed_indexes: tuple[int, ...] = ()
        if self.deduplicate_repetitions:
            repetition_source_utterances: tuple[str, ...] = self._clean_utterance_lines(text)
            if not repetition_source_utterances:
                repetition_source_utterances = (punctuation_cleaned.text,)
            repetition_result: ASRRepeatDeduplicationResult = deduplicate_asr_utterances(
                repetition_source_utterances,
                config=self.repetition_config,
            )
            repetition_cleaned_text = repetition_result.clean_transcript
            repetition_suppressed_indexes = repetition_result.suppressed_indexes
        punctuation_to_repetition_alignment: TextAlignment = align_texts_by_diff(
            punctuation_cleaned.text,
            repetition_cleaned_text,
        )
        original_to_repetition_alignment: TextAlignment = compose_alignments(
            original_to_punctuation_alignment,
            punctuation_to_repetition_alignment,
        )

        cleaned_text: str = repetition_cleaned_text
        integer_spans: list[IntegerSpan] = self.integer_extractor.extract(cleaned_text)
        cleaned_to_numeric_alignment: TextAlignment = build_replacement_alignment(
            cleaned_text,
            integer_spans,
        )
        numeric_normalized_text: str = cleaned_to_numeric_alignment.target_text
        original_to_numeric_alignment: TextAlignment = compose_alignments(
            original_to_repetition_alignment,
            cleaned_to_numeric_alignment,
        )

        document_number_normalized: DocumentNumberNormalizedText
        if self.normalize_document_numbers:
            document_number_normalized = self.document_number_normalizer.normalize(
                numeric_normalized_text
            )
        else:
            document_number_normalized = DocumentNumberNormalizedText(
                original_text=numeric_normalized_text,
                text=numeric_normalized_text,
            )
        numeric_to_document_alignment: TextAlignment = build_replacement_alignment(
            numeric_normalized_text,
            document_number_normalized.spans,
        )
        original_to_document_alignment: TextAlignment = compose_alignments(
            original_to_numeric_alignment,
            numeric_to_document_alignment,
        )

        date_birth_normalized: DateBirthNormalizedText
        if self.normalize_date_birth:
            date_birth_normalized = self.date_birth_normalizer.normalize(
                document_number_normalized.text
            )
        else:
            date_birth_normalized = DateBirthNormalizedText(
                original_text=document_number_normalized.text,
                text=document_number_normalized.text,
            )
        document_to_date_birth_alignment: TextAlignment = build_replacement_alignment(
            document_number_normalized.text,
            date_birth_normalized.spans,
        )
        original_to_date_birth_alignment: TextAlignment = compose_alignments(
            original_to_document_alignment,
            document_to_date_birth_alignment,
        )

        contact_normalized: ContactNormalizedText
        if self.normalize_contacts:
            contact_normalized = self.contact_normalizer.normalize(date_birth_normalized.text)
        else:
            contact_normalized = ContactNormalizedText(
                original_text=date_birth_normalized.text,
                text=date_birth_normalized.text,
            )
        date_birth_to_contact_alignment: TextAlignment = build_replacement_alignment(
            date_birth_normalized.text,
            contact_normalized.spans,
        )
        normalized_to_original_alignment: TextAlignment = compose_alignments(
            original_to_date_birth_alignment,
            date_birth_to_contact_alignment,
        )

        normalized_text: str = contact_normalized.text
        result = ASRNormalizationResult(
            original_text=text,
            disfluency_cleaned_text=disfluency_cleaned.text,
            punctuation_cleaned_text=punctuation_cleaned.text,
            repetition_cleaned_text=repetition_cleaned_text,
            cleaned_text=cleaned_text,
            normalized_text=normalized_text,
            numeric_normalized_text=numeric_normalized_text,
            document_number_normalized_text=document_number_normalized.text,
            date_birth_normalized_text=date_birth_normalized.text,
            contact_normalized_text=contact_normalized.text,
            removed_spans=disfluency_cleaned.removed_spans,
            punctuation_removed_spans=punctuation_cleaned.removed_spans,
            punctuation_protected_spans=punctuation_cleaned.protected_spans,
            repetition_suppressed_indexes=repetition_suppressed_indexes,
            integer_spans=tuple(integer_spans),
            document_number_spans=document_number_normalized.spans,
            date_birth_spans=date_birth_normalized.spans,
            contact_spans=contact_normalized.spans,
            normalized_to_original_alignment=normalized_to_original_alignment,
        )
        return result

    def to_json(self, text: str, *, ensure_ascii: bool = False) -> str:
        result: ASRNormalizationResult = self.run(text)
        payload: dict[str, object] = asdict(result)
        serialized: str = json.dumps(payload, ensure_ascii=ensure_ascii, indent=2)
        return serialized

    def _clean_utterance_lines(self, text: str) -> tuple[str, ...]:
        raw_lines: tuple[str, ...] = tuple(line.strip() for line in text.splitlines() if line.strip())
        if len(raw_lines) <= 1:
            return ()

        cleaned_lines: list[str] = []
        for raw_line in raw_lines:
            disfluency_text: str
            if self.remove_disfluencies:
                disfluency_text = self.disfluency_filter.clean(raw_line).text
            else:
                disfluency_text = raw_line

            punctuation_text: str
            if self.remove_punctuation:
                punctuation_text = self.punctuation_remover.clean(disfluency_text).text
            else:
                punctuation_text = disfluency_text

            if punctuation_text:
                cleaned_lines.append(punctuation_text)

        return tuple(cleaned_lines)


def run_asr_normalization(
    text: str,
    *,
    disfluency_config: DisfluencyFilterConfig | None = None,
    punctuation_config: PunctuationRemovalConfig | None = None,
    extractor_config: ExtractorConfig | None = None,
    repetition_config: ASRRepeatDeduplicationConfig | None = None,
    document_number_config: DocumentNumberNormalizerConfig | None = None,
    date_birth_config: DateBirthNormalizerConfig | None = None,
    contact_config: ContactNormalizerConfig | None = None,
    remove_disfluencies: bool = True,
    remove_punctuation: bool = True,
    deduplicate_repetitions: bool = False,
    normalize_document_numbers: bool = False,
    normalize_date_birth: bool = True,
    normalize_contacts: bool = True,
) -> ASRNormalizationResult:
    pipeline = ASRNormalizationPipeline(
        disfluency_config=disfluency_config,
        punctuation_config=punctuation_config,
        extractor_config=extractor_config,
        repetition_config=repetition_config,
        document_number_config=document_number_config,
        date_birth_config=date_birth_config,
        contact_config=contact_config,
        remove_disfluencies=remove_disfluencies,
        remove_punctuation=remove_punctuation,
        deduplicate_repetitions=deduplicate_repetitions,
        normalize_document_numbers=normalize_document_numbers,
        normalize_date_birth=normalize_date_birth,
        normalize_contacts=normalize_contacts,
    )
    result: ASRNormalizationResult = pipeline.run(text)
    return result


__all__: list[str] = [
    "ASRNormalizationPipeline",
    "ASRNormalizationResult",
    "run_asr_normalization",
]
