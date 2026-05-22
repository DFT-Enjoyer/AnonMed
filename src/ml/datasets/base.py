from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ml.core.types import Case, TextDocument


@dataclass(frozen=True)
class Dataset(ABC):
    cases: tuple[Case, ...]
    _row_data: Any = field(init=False)

    @property
    def documents(self) -> tuple[TextDocument, ...]:
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
