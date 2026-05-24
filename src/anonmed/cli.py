from __future__ import annotations

import argparse
import json
import sys

from anonmed.preprocessing.asr import (
    ASRNormalizationPipeline,
    IntegerExtractor,
    remove_disfluencies,
    remove_punctuation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AnonMed preprocessing CLI for noisy Russian ASR text."
    )
    parser.add_argument("text", nargs="*", help="Input text. If empty, stdin is used.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--replace", action="store_true", help="Replace numeric spans only.")
    mode_group.add_argument(
        "--clean",
        action="store_true",
        help="Remove ASR fillers and interjections only.",
    )
    mode_group.add_argument(
        "--punctuation-clean",
        action="store_true",
        help="Remove punctuation only, preserving protected numeric/domain/email/URL punctuation.",
    )
    mode_group.add_argument(
        "--run",
        action="store_true",
        help="Remove ASR fillers/interjections and punctuation, then replace numeric spans.",
    )
    mode_group.add_argument(
        "--run-json",
        action="store_true",
        help="Run the complete cleanup pipeline and print structured JSON.",
    )
    mode_group.add_argument(
        "--anonymize",
        action="store_true",
        help="Run the PII anonymization pipeline and print masked text.",
    )
    mode_group.add_argument(
        "--anonymize-json",
        action="store_true",
        help="Run the PII anonymization pipeline and print structured JSON.",
    )
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="Disable punctuation removal in --run, --run-json, and anonymization modes.",
    )
    parser.add_argument(
        "--deduplicate-repetitions",
        action="store_true",
        help="Deduplicate repeated ASR utterance lines in --run and --run-json modes.",
    )
    parser.add_argument(
        "--normalize-document-numbers",
        action="store_true",
        help=(
            "Split exact repeated document/phone digit runs in strong PII contexts "
            "in --run, --run-json, and anonymization modes."
        ),
    )
    parser.add_argument(
        "--no-normalize-numbers",
        action="store_true",
        help="Disable spoken number normalization in anonymization modes.",
    )
    parser.add_argument(
        "--no-normalize-contacts",
        action="store_true",
        help="Disable spoken contact normalization in anonymization modes.",
    )
    parser.add_argument(
        "--no-normalize-dates",
        action="store_true",
        help="Disable spoken birth date normalization in anonymization modes.",
    )
    parser.add_argument(
        "--ml-model",
        help="ML model id for anonymization, for example 'natasha_per' or 'GLiNER2'.",
    )
    parser.add_argument(
        "--device",
        help="Optional ML device passed to the model runner.",
    )
    parser.add_argument(
        "--use-ml",
        dest="use_ml",
        action="store_true",
        default=None,
        help="Enable ML detection in anonymization modes.",
    )
    parser.add_argument(
        "--no-ml",
        dest="use_ml",
        action="store_false",
        help="Disable ML detection in anonymization modes.",
    )
    parser.add_argument(
        "--use-rules",
        dest="use_rules",
        action="store_true",
        default=None,
        help="Enable rule-based detection in anonymization modes.",
    )
    parser.add_argument(
        "--no-rules",
        dest="use_rules",
        action="store_false",
        help="Disable rule-based detection in anonymization modes.",
    )
    parser.add_argument(
        "--no-preprocessing",
        dest="use_preprocessing",
        action="store_false",
        default=None,
        help="Disable preprocessing before anonymization.",
    )
    parser.add_argument(
        "--no-postprocessing",
        dest="use_postprocessing",
        action="store_false",
        default=None,
        help="Disable postprocessing/restoration after detection.",
    )
    parser.add_argument(
        "--normalized-output",
        dest="restore_non_pii",
        action="store_false",
        default=None,
        help="Print masked preprocessed text instead of restored original-layer text.",
    )
    parser.add_argument(
        "--pii-types",
        help="Comma-separated rule-based PII types to keep, for example PHONE,SNILS.",
    )
    parser.add_argument(
        "--ml-labels",
        help="Comma-separated ML labels to keep, for example PER,ADDRESS,EMAIL.",
    )
    parser.add_argument(
        "--masking-strategy",
        choices=("type", "same_length"),
        default=None,
        help="Masking strategy for anonymization modes.",
    )
    parser.add_argument(
        "--post-processing-mode",
        choices=("balanced", "conservative", "production_safe"),
        default=None,
        help="Candidate overlap resolution mode for anonymization modes.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Escape non-ASCII characters in JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    joined_text: str = " ".join(args.text).strip()
    input_text: str = joined_text if joined_text else sys.stdin.read()

    if args.clean:
        cleaned_text: str = remove_disfluencies(input_text)
        print(cleaned_text)
        return 0

    if args.punctuation_clean:
        punctuation_cleaned_text: str = remove_punctuation(input_text)
        print(punctuation_cleaned_text)
        return 0

    if args.run or args.run_json:
        pipeline = ASRNormalizationPipeline(
            remove_punctuation=not args.keep_punctuation,
            deduplicate_repetitions=args.deduplicate_repetitions,
            normalize_document_numbers=args.normalize_document_numbers,
        )
        if args.run_json:
            serialized: str = pipeline.to_json(input_text, ensure_ascii=args.ascii)
            print(serialized)
        else:
            normalized_text: str = pipeline.run(input_text).normalized_text
            print(normalized_text)
        return 0

    if args.anonymize or args.anonymize_json:
        from anonmed import PIIAnonymizer

        anonymizer = PIIAnonymizer(ml_model=args.ml_model, device=args.device)
        result = anonymizer(
            input_text,
            use_preprocessing=args.use_preprocessing,
            use_rules=args.use_rules,
            use_ml=args.use_ml,
            use_postprocessing=args.use_postprocessing,
            remove_punctuation=False if args.keep_punctuation else None,
            normalize_numbers=False if args.no_normalize_numbers else None,
            normalize_document_numbers=args.normalize_document_numbers or None,
            normalize_contacts=False if args.no_normalize_contacts else None,
            normalize_dates=False if args.no_normalize_dates else None,
            deduplicate_repetitions=args.deduplicate_repetitions or None,
            restore_non_pii=args.restore_non_pii,
            pii_types=_split_csv(args.pii_types),
            ml_labels=_split_csv(args.ml_labels),
            masking_strategy=args.masking_strategy,
            post_processing_mode=args.post_processing_mode,
        )
        if args.anonymize_json:
            serialized = json.dumps(
                result.to_dict(include_debug=True),
                ensure_ascii=args.ascii,
                indent=2,
            )
            print(serialized)
        else:
            print(result.masked_text)
        return 0

    extractor = IntegerExtractor()
    if args.replace:
        replaced: str = extractor.replace(input_text)
        print(replaced)
    else:
        extractor_serialized: str = extractor.to_json(input_text, ensure_ascii=args.ascii)
        print(extractor_serialized)
    return 0


def _split_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items: tuple[str, ...] = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = ["build_parser", "main"]
