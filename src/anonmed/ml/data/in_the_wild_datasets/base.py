from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, ClassVar, Literal

from anonmed.ml.core.types import (
    AnnotationSet,
    AnnotationSetLine,
    Case,
    ParticipantKind,
    Role,
    Span,
    TextDocument,
    TextLine,
)
from anonmed.ml.data.base import Dataset

PathLike = str | Path
Representation = Literal["digits", "letters"]

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[5]
DEFAULT_IN_THE_WILD_DATASET_ROOT: Path = _REPOSITORY_ROOT / "data" / "in_the_wild_datasets"
_BRAT_ENTITY_PATTERN: re.Pattern[str] = re.compile(
    r"^T\d+\t(?P<label>\S+)\s+(?P<offsets>[0-9; ]+)\t(?P<data>.*)$"
)


@dataclass(frozen=True, slots=True)
class _RawSpan:
    begin: int
    end: int
    label: str
    data: str


@dataclass(frozen=True, slots=True)
class _RawSample:
    sample_id: str
    text: str
    spans: tuple[_RawSpan, ...]


class InTheWildDataset(Dataset):
    dataset_directory: ClassVar[str]
    name: ClassVar[str]

    def __init__(
        self,
        root: PathLike | None = None,
        *,
        representation: Representation | None = None,
        filename: str | None = None,
        split: str | None = None,
        sample_size: int | None = None,
        strict_spans: bool = False,
    ) -> None:
        _validate_sample_size(sample_size)
        object.__setattr__(self, "root", _resolve_root(root))
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "strict_spans", strict_spans)
        super().__init__()

    def _load(self) -> None:
        files: tuple[Path, ...] = _select_jsonl_files(
            root=self.root,
            dataset_directory=self.dataset_directory,
            representation=self.representation,
            filename=self.filename,
            split=self.split,
        )
        samples: list[_RawSample] = []
        for path in files:
            for line_number, row in _iter_jsonl_mappings(path):
                text: str = _text_from_row(row)
                if text == "":
                    continue
                spans: tuple[_RawSpan, ...] = _parse_spans(
                    row=row,
                    text=text,
                    path=path,
                    line_number=line_number,
                    strict_spans=self.strict_spans,
                )
                sample_id: str = _sample_id(
                    root=self.root,
                    path=path,
                    line_number=line_number,
                    explicit_id=row.get("id"),
                )
                samples.append(_RawSample(sample_id=sample_id, text=text, spans=spans))
                if self.sample_size is not None and len(samples) >= self.sample_size:
                    object.__setattr__(self, "_row_data", tuple(samples))
                    return
        object.__setattr__(self, "_row_data", tuple(samples))

    def _convert(self) -> None:
        cases: tuple[Case, ...] = _cases_from_samples(self._row_data)
        object.__setattr__(self, "cases", cases)


def _resolve_root(root: PathLike | None) -> Path:
    resolved_root: Path = DEFAULT_IN_THE_WILD_DATASET_ROOT if root is None else Path(root)
    return resolved_root.expanduser()


def _validate_sample_size(sample_size: int | None) -> None:
    if sample_size is not None and sample_size <= 0:
        raise ValueError(f"sample_size must be positive or None, got {sample_size}")


def _select_jsonl_files(
    root: Path,
    *,
    dataset_directory: str,
    representation: Representation | None,
    filename: str | None,
    split: str | None,
) -> tuple[Path, ...]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    candidates: Iterable[Path] = (root,) if root.is_file() else root.rglob("*.jsonl")
    files: list[Path] = []
    for path in sorted(candidates):
        if filename is not None and path.name != filename:
            continue
        if dataset_directory not in path.parts:
            continue
        if representation is not None and representation not in path.parts:
            continue
        if split is not None and split not in path.stem:
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No JSONL files matched dataset filters under {root}")
    return tuple(files)


def _iter_jsonl_mappings(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload: Any = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"JSONL row must be an object: {path}:{line_number}")
            yield line_number, payload


def _text_from_row(row: Mapping[str, Any]) -> str:
    for key in ("source_text", "text"):
        value: Any = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def _parse_spans(
    *,
    row: Mapping[str, Any],
    text: str,
    path: Path,
    line_number: int,
    strict_spans: bool,
) -> tuple[_RawSpan, ...]:
    raw_entities: Any = row.get("entities", ())
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        _handle_invalid_span(path, line_number, "entities must be a sequence", strict_spans)
        return ()

    spans: list[_RawSpan] = []
    for raw_entity in raw_entities:
        raw_span: _RawSpan | None
        if isinstance(raw_entity, Mapping):
            raw_span = _raw_span_from_mapping(raw_entity, text=text)
        elif isinstance(raw_entity, str):
            raw_span = _raw_span_from_brat(raw_entity)
        else:
            raw_span = None
        if raw_span is None:
            _handle_invalid_span(path, line_number, f"invalid entity: {raw_entity!r}", strict_spans)
            continue
        if not _is_valid_raw_span(raw_span, text):
            _handle_invalid_span(path, line_number, f"invalid span: {raw_span!r}", strict_spans)
            continue
        spans.append(raw_span)
    return tuple(spans)


def _raw_span_from_mapping(raw_span: Mapping[str, Any], *, text: str) -> _RawSpan | None:
    begin: int | None = _optional_int(raw_span.get("start", raw_span.get("begin")))
    end: int | None = _optional_int(raw_span.get("end"))
    if begin is None or end is None:
        return None
    label_value: Any = raw_span.get("type", raw_span.get("label"))
    data_value: Any = raw_span.get("text", raw_span.get("data"))
    label: str = str(label_value) if label_value is not None else "unknown"
    data: str = str(data_value) if data_value not in (None, "") else text[begin:end]
    return _RawSpan(begin=begin, end=end, label=label, data=data)


def _raw_span_from_brat(raw_entity: str) -> _RawSpan | None:
    match: re.Match[str] | None = _BRAT_ENTITY_PATTERN.match(raw_entity)
    if match is None:
        return None
    offsets: list[int] = [int(value) for value in re.findall(r"\d+", match.group("offsets"))]
    if len(offsets) < 2:
        return None
    begin: int = min(offsets[::2])
    end: int = max(offsets[1::2])
    return _RawSpan(begin=begin, end=end, label=match.group("label"), data=match.group("data"))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_valid_raw_span(raw_span: _RawSpan, text: str) -> bool:
    return 0 <= raw_span.begin < raw_span.end <= len(text) and raw_span.label != ""


def _handle_invalid_span(path: Path, line_number: int, reason: str, strict_spans: bool) -> None:
    if strict_spans:
        raise ValueError(f"{path}:{line_number}: {reason}")


def _sample_id(root: Path, path: Path, line_number: int, explicit_id: Any) -> str:
    if explicit_id is not None:
        explicit_id_text: str = str(explicit_id)
        if explicit_id_text:
            return explicit_id_text
    base_path: Path = root.parent if root.is_file() else root
    try:
        relative_path: Path = path.relative_to(base_path)
    except ValueError:
        relative_path = path
    return f"{relative_path.as_posix()}:{line_number}"


def _cases_from_samples(samples: Sequence[_RawSample]) -> tuple[Case, ...]:
    role = Role(name="text", kind=ParticipantKind.UNKNOWN)
    cases: list[Case] = []
    for sample in samples:
        line = TextLine(idx=0, role=role, text=sample.text)
        document = TextDocument(lines=(line,), sample_id=sample.sample_id)
        spans: list[Span] = [
            Span(
                line_idx=0,
                begin=raw_span.begin,
                end=raw_span.end,
                label=raw_span.label,
                data=raw_span.data,
            )
            for raw_span in sample.spans
        ]
        target_line = AnnotationSetLine(idx=0, role=role, spans=spans)
        target = AnnotationSet(lines=(target_line,), idx=sample.sample_id)
        cases.append(Case(document=document, target=target))
    return tuple(cases)


__all__: list[str] = [
    "DEFAULT_IN_THE_WILD_DATASET_ROOT",
    "InTheWildDataset",
    "PathLike",
    "Representation",
]
