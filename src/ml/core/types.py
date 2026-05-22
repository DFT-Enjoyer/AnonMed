from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


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


@dataclass(frozen=True, slots=True)
class Role:
    name: str
    kind: ParticipantKind = ParticipantKind.UNKNOWN


@dataclass(frozen=True, slots=True)
class Span:
    line_idx: int
    begin: int
    end: int
    label: str
    data: str


@dataclass(frozen=True, slots=True)
class TextLine:
    idx: int
    role: Role
    text: str


@dataclass(frozen=True, slots=True)
class TextDocument:
    lines: tuple[TextLine, ...]
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnnotationSetLine:
    idx: int
    role: Role
    spans: list[Span]


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    lines: tuple[AnnotationSetLine]
    idx: str | None = None


@dataclass(frozen=True, slots=True)
class Case:
    document: TextDocument
    target: AnnotationSet


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    metrics: dict[str, MetricResult]
    samples_count: int
