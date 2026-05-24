from dataclasses import dataclass
from typing import Any

from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Case, Role, TextDocument, TextLine
from anonmed.ml.data.base import Dataset


@dataclass(frozen=True)
class ExampleDataset(Dataset):
    def _load(self):
        object.__setattr__(self, "_row_data", [{"sample_id": "example-1", "text": "example text"}])

    def _convert(self):
        row: dict[str, Any] = self._row_data[0]
        sample_id = str(row["sample_id"])
        text = str(row["text"])
        role = Role(name="client")
        line = TextLine(idx=0, role=role, text=text)
        document = TextDocument(lines=(line,), sample_id=sample_id)
        target_line = AnnotationSetLine(idx=0, role=role, spans=[])
        target = AnnotationSet(lines=(target_line,), idx=sample_id)
        object.__setattr__(self, "cases", (Case(document=document, target=target),))


def build_example_dataset() -> ExampleDataset:
    return ExampleDataset()
