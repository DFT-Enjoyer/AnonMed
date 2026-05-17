from __future__ import annotations

import argparse
import sys

from asr_integer_extractor import IntegerExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract integer spans from noisy Russian ASR text.")
    parser.add_argument("text", nargs="*", help="Input text. If empty, stdin is used.")
    parser.add_argument("--replace", action="store_true", help="Replace numeric spans in text and print text.")
    parser.add_argument("--ascii", action="store_true", help="Escape non-ASCII characters in JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    joined_text: str = " ".join(args.text).strip()
    input_text: str = joined_text if joined_text else sys.stdin.read()
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
