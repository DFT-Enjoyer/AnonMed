from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from anonmed.ml.core.types import Case, TextDocument


@dataclass(frozen=True)
class Dataset(ABC):
    cases: tuple[Case, ...]
    _row_data: Any = field(init=False)

    @property
    def documents(self) -> tuple[TextDocument, ...]:
        return tuple([case.document for case in self.cases])

    def __post_init__(self):
        self._load()
        if not hasattr(self, "_row_data"):
            raise ValueError("Dataset._load() must initialize _row_data")
        self._convert()
        self._validate_cases()

    def _validate_cases(self) -> None:
        if not isinstance(self.cases, tuple):
            raise TypeError(f"Dataset.cases must be tuple[Case, ...], got {type(self.cases).__name__}")
        if len(self.cases) == 0:
            raise ValueError("Dataset.cases must not be empty after _convert()")
        for case in self.cases:
            if not isinstance(case, Case):
                raise TypeError(f"Dataset.cases must contain Case, got {type(case).__name__}")

    @abstractmethod
    def _load(self):
        ...

    @abstractmethod
    def _convert(self):
        ...
