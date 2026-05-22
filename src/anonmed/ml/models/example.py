from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, TextDocument
from anonmed.ml.models.base import PIIModel


class ExamplePIIModel(PIIModel):
    def predict(self, document: TextDocument) -> AnnotationSet:
        lines = tuple(
            AnnotationSetLine(idx=line.idx, role=line.role, spans=[])
            for line in document.lines
        )
        return AnnotationSet(lines=lines, idx=document.sample_id)
