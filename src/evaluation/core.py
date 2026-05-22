from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple, TypeAlias


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
    CLIENT = 'client'
    SPECIALIST = 'specialist'
    SYSTEM = 'system'
    UNKNOWN = 'unknown'


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
    lines: Tuple[TextLine, ...]
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnnotationSetLine:
    idx: int
    role: Role
    spans: List[Span]


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    lines: Tuple[AnnotationSetLine]
    idx: str | None = None


@dataclass(frozen=True, slots=True)
class Case:
    document: TextDocument
    target: AnnotationSet


@dataclass(frozen=True)
class Dataset(ABC):
    cases: Tuple[Case, ...]
    _row_data: Any = field(init=False)

    @property
    def documents(self) -> Tuple[TextDocument, ...]:
        return tuple([case.document for case in self.cases])
    
    def __post_init__(self):
        self._load()
        self._convert()
    
    @abstractmethod
    def _load(self):
        ...
    
    @abstractmethod
    def _convert(self):
        ...


class PIIModel(ABC):
    def process(self, documents: Tuple[TextDocument, ...]) -> tuple[AnnotationSet, ...]:
        predictions: List[AnnotationSet] = []
        for document in documents:
            prediction: AnnotationSet = self.predict(document)
            predictions.append(prediction)
        return tuple(predictions)
    
    @abstractmethod
    def predict(self, document) -> AnnotationSet:
        ...


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    metrics: Dict[str, MetricResult]
    samples_count: int


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def compute(self, dataset: Dataset, predictions: Tuple[AnnotationSet, ...]) -> Dict[str, Any]:
        ...


class Evaluator:
    def __init__(self, dataset: Dataset):
        self._dataset: Dataset = dataset

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def eval_model(self, model: PIIModel, metrics: Sequence[Metric]) -> EvaluationReport:
        for metric in metrics:
            metric.reset()

        predictions: Tuple[AnnotationSet, ...] = model.process(self.dataset.documents)

        results: Dict[str, MetricResult] = {}
        for metric in metrics:
            metric_values: MetricResult = metric.compute(dataset=self._dataset, predictions=predictions);
            results[metric.name] = metric_values

        report = EvaluationReport(metrics=results, samples_count=len(self._dataset.cases));
        return report
