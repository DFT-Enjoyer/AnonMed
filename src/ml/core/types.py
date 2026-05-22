from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias


MetricValue: TypeAlias = (
    int
    | float
    | str
    | bool
    | None
    | list["MetricValue"]
    | dict[str, "MetricValue"]
)
MetricResult: TypeAlias = dict[str, MetricValue]


class ParticipantKind(str, Enum):
    CLIENT = "client"
    SPECIALIST = "specialist"
    SYSTEM = "system"
    UNKNOWN = "unknown"


def _ensure_non_negative(value: int, field_name: str) -> None:
    if value < 0: raise ValueError(f"{field_name} must be non-negative, got {value}")


def _ensure_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
    if value == "":
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class Role:
    name: str
    kind: ParticipantKind = ParticipantKind.UNKNOWN

    def __post_init__(self) -> None:
        _ensure_non_empty_string(self.name, "Role.name")


@dataclass(frozen=True, slots=True)
class Span:
    line_idx: int
    begin: int
    end: int
    label: str
    data: str

    def __post_init__(self) -> None:
        _ensure_non_negative(self.line_idx, "Span.line_idx")
        _ensure_non_negative(self.begin, "Span.begin")
        if self.end <= self.begin:
            raise ValueError(f"Span.end must be greater than Span.begin, got [{self.begin}, {self.end})")
        _ensure_non_empty_string(self.label, "Span.label")
        _ensure_non_empty_string(self.data, "Span.data")


@dataclass(frozen=True, slots=True)
class TextLine:
    idx: int
    role: Role
    text: str

    def __post_init__(self) -> None:
        _ensure_non_negative(self.idx, "TextLine.idx")
        if not isinstance(self.role, Role):
            raise TypeError(f"TextLine.role must be Role, got {type(self.role).__name__}")
        if not isinstance(self.text, str):
            raise TypeError(f"TextLine.text must be str, got {type(self.text).__name__}")


@dataclass(frozen=True, slots=True)
class TextDocument:
    lines: tuple[TextLine, ...]
    sample_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lines, tuple):
            raise TypeError(f"TextDocument.lines must be tuple, got {type(self.lines).__name__}")
        if len(self.lines) == 0:
            raise ValueError("TextDocument.lines must not be empty")

        indices: set[int] = set()
        for line in self.lines:
            if not isinstance(line, TextLine):
                raise TypeError(f"TextDocument.lines must contain TextLine, got {type(line).__name__}")
            _ensure_non_negative(line.idx, "TextDocument.lines[idx]")
            if line.idx in indices:
                raise ValueError(f"TextDocument.lines contain duplicate idx={line.idx}")
            indices.add(line.idx)

        if self.sample_id is not None and not isinstance(self.sample_id, str):
            raise TypeError(f"TextDocument.sample_id must be str|None, got {type(self.sample_id).__name__}")


@dataclass(frozen=True, slots=True)
class AnnotationSetLine:
    idx: int
    role: Role
    spans: list[Span]

    def __post_init__(self) -> None:
        _ensure_non_negative(self.idx, "AnnotationSetLine.idx")
        if not isinstance(self.role, Role):
            raise TypeError(f"AnnotationSetLine.role must be Role, got {type(self.role).__name__}")
        if not isinstance(self.spans, list):
            raise TypeError(f"AnnotationSetLine.spans must be list, got {type(self.spans).__name__}")
        for span in self.spans:
            if not isinstance(span, Span):
                raise TypeError(f"AnnotationSetLine.spans must contain Span, got {type(span).__name__}")
            if span.line_idx != self.idx:
                raise ValueError(
                    "AnnotationSetLine span line mismatch: "
                    f"line idx={self.idx}, span.line_idx={span.line_idx}"
                )


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    lines: tuple[AnnotationSetLine, ...]
    idx: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lines, tuple):
            raise TypeError(f"AnnotationSet.lines must be tuple, got {type(self.lines).__name__}")
        for line in self.lines:
            if not isinstance(line, AnnotationSetLine):
                raise TypeError(f"AnnotationSet.lines must contain AnnotationSetLine, got {type(line).__name__}")
        if self.idx is not None and not isinstance(self.idx, str):
            raise TypeError(f"AnnotationSet.idx must be str|None, got {type(self.idx).__name__}")


@dataclass(frozen=True, slots=True)
class Case:
    document: TextDocument
    target: AnnotationSet

    def __post_init__(self) -> None:
        if not isinstance(self.document, TextDocument):
            raise TypeError(f"Case.document must be TextDocument, got {type(self.document).__name__}")
        if not isinstance(self.target, AnnotationSet):
            raise TypeError(f"Case.target must be AnnotationSet, got {type(self.target).__name__}")

        document_line_indices = {line.idx for line in self.document.lines}
        target_line_indices = {line.idx for line in self.target.lines}
        if not target_line_indices.issubset(document_line_indices):
            raise ValueError(
                "Case target lines must be subset of document lines: "
                f"target={sorted(target_line_indices)}, document={sorted(document_line_indices)}"
            )

        if self.document.sample_id is not None and self.target.idx is not None:
            if self.document.sample_id != self.target.idx:
                raise ValueError(
                    "Case id mismatch between document.sample_id and target.idx: "
                    f"{self.document.sample_id} != {self.target.idx}"
                )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    metrics: dict[str, MetricResult]
    samples_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, dict):
            raise TypeError(f"EvaluationReport.metrics must be dict, got {type(self.metrics).__name__}")
        _ensure_non_negative(self.samples_count, "EvaluationReport.samples_count")
