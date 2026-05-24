from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="Disable punctuation removal in --run and --run-json modes.",
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
        pipeline = ASRNormalizationPipeline(remove_punctuation=not args.keep_punctuation)
        if args.run_json:
            serialized: str = pipeline.to_json(input_text, ensure_ascii=args.ascii)
            print(serialized)
        else:
            normalized_text: str = pipeline.run(input_text).normalized_text
            print(normalized_text)
        return 0

    extractor = IntegerExtractor()
    if args.replace:
        replaced: str = extractor.replace(input_text)
        print(replaced)
    else:
        serialized: str = extractor.to_json(input_text, ensure_ascii=args.ascii)
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = ["build_parser", "main"]
