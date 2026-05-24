from dataclasses import dataclass
from typing import Any

from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Case, Role, TextDocument, TextLine
from anonmed.ml.datasets.base import Dataset


@dataclass(frozen=True)
class ExampleDataset(Dataset):
    def _load(self):
        object.__setattr__(self, "_row_data", [{"sample_id": "example-1", "text": "example text"}])

    def _convert(self):
        if self.cases:
            return

        line = TextLine(idx=0, role=Role(name="client"), text="example text")
        document = TextDocument(lines=(line,), sample_id="example-1")
        target_line = AnnotationSetLine(idx=0, role=Role(name="client"), spans=[])
        target = AnnotationSet(lines=(target_line,), idx="example-1")
        object.__setattr__(self, "cases", (Case(document=document, target=target),))


def build_example_dataset() -> ExampleDataset:
    return ExampleDataset(cases=())
