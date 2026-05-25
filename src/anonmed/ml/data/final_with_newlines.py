from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Case, Role, Span, TextDocument, TextLine
from anonmed.ml.data.base import Dataset


DEFAULT_FINAL_WITH_NEWLINES_PATH = (
    Path(__file__).resolve().parents[4] / "GenerateDialogs" / "final_with_newlines.jsonl"
)
DEFAULT_SPAN_LABELS = ("name",)


@dataclass(frozen=True)
class FinalWithNewlinesDataset(Dataset):
    path: str | Path = DEFAULT_FINAL_WITH_NEWLINES_PATH
    sample_size: int | None = None
    random_seed: int = 42
    split: str | None = None
    span_labels: tuple[str, ...] | str = DEFAULT_SPAN_LABELS
    label: str = "PER"

    def _load(self) -> None:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"final_with_newlines dataset file not found: {path}")

        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL row in {path}:{line_number}: expected object")
            if self.split is not None and row.get("split") != self.split:
                continue
            rows.append(row)

        if self.sample_size is not None and self.sample_size < len(rows):
            rng = random.Random(self.random_seed)
            rows = rng.sample(rows, self.sample_size)

        object.__setattr__(self, "_row_data", rows)

    def _convert(self) -> None:
        cases: list[Case] = []
        role = Role(name="text")
        allowed_labels = _allowed_labels(self.span_labels)
        for index, row in enumerate(self._row_data):
            text = str(row.get("text", ""))
            if not text:
                continue
            sample_id = str(row.get("id", index))
            line = TextLine(idx=0, role=role, text=text)
            document = TextDocument(lines=(line,), sample_id=sample_id)
            spans = [
                self._source_span_to_target_span(source_span, text=text)
                for source_span in _source_spans(row)
                if str(source_span.get("label", "")) in allowed_labels
            ]
            spans = [span for span in spans if span is not None]
            spans.sort(key=lambda span: (span.begin, span.end, span.label))
            target_line = AnnotationSetLine(idx=0, role=role, spans=spans)
            target = AnnotationSet(lines=(target_line,), idx=sample_id)
            cases.append(Case(document=document, target=target))

        object.__setattr__(self, "cases", tuple(cases))

    def _source_span_to_target_span(self, source_span: Mapping[str, Any], *, text: str) -> Span | None:
        begin = _optional_int(source_span.get("begin"))
        end = _optional_int(source_span.get("end"))
        if begin is None or end is None:
            return None
        if not (0 <= begin < end <= len(text)):
            raise ValueError(
                f"Invalid final_with_newlines span [{begin},{end}) for text length {len(text)}"
            )
        span_text = text[begin:end]
        data = str(source_span.get("data") or span_text)
        return Span(line_idx=0, begin=begin, end=end, label=self.label, data=data)


def _allowed_labels(value: tuple[str, ...] | str) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(label) for label in value}


def _source_spans(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    spans = row.get("spans", ())
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes, bytearray)):
        return ()
    return tuple(span for span in spans if isinstance(span, Mapping))


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["DEFAULT_FINAL_WITH_NEWLINES_PATH", "FinalWithNewlinesDataset"]
