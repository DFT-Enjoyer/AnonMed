from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Case, Role, Span, TextDocument, TextLine
from anonmed.ml.data.base import Dataset


DEFAULT_GT_ASR_PATH = Path(__file__).resolve().parents[4] / "gt_asr.jsonl"
DEFAULT_ANNOTATION_TYPES = ("ФИО",)


@dataclass(frozen=True)
class GTASRDataset(Dataset):
    path: str | Path = DEFAULT_GT_ASR_PATH
    sample_size: int | None = None
    random_seed: int = 42
    split: str | None = None
    annotation_types: tuple[str, ...] = DEFAULT_ANNOTATION_TYPES
    label: str = "PER"

    def _load(self) -> None:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"GT ASR dataset file not found: {path}")

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
        allowed_types = set(self.annotation_types)
        for index, row in enumerate(self._row_data):
            text = str(row.get("value", ""))
            if not text:
                continue
            sample_id = str(row.get("id", index))
            line = TextLine(idx=0, role=role, text=text)
            document = TextDocument(lines=(line,), sample_id=sample_id)
            spans = [
                self._annotation_to_span(annotation, text=text)
                for annotation in _annotations(row)
                if str(annotation.get("type", "")) in allowed_types
            ]
            spans = [span for span in spans if span is not None]
            spans.sort(key=lambda span: (span.begin, span.end, span.label))
            target_line = AnnotationSetLine(idx=0, role=role, spans=spans)
            target = AnnotationSet(lines=(target_line,), idx=sample_id)
            cases.append(Case(document=document, target=target))

        object.__setattr__(self, "cases", tuple(cases))

    def _annotation_to_span(self, annotation: Mapping[str, Any], *, text: str) -> Span | None:
        start = _optional_int(annotation.get("start"))
        end = _optional_int(annotation.get("end"))
        if start is None or end is None:
            return None
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"Invalid GT ASR span [{start},{end}) for text length {len(text)}")
        span_text = text[start:end]
        data = str(annotation.get("text") or span_text)
        return Span(line_idx=0, begin=start, end=end, label=self.label, data=data)


def _annotations(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    annotations = row.get("annotations", ())
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes, bytearray)):
        return ()
    return tuple(annotation for annotation in annotations if isinstance(annotation, Mapping))


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["DEFAULT_GT_ASR_PATH", "GTASRDataset"]
