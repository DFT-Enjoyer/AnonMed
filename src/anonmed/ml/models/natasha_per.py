from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc, PER
from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Span, TextDocument
from anonmed.ml.models.base import PIIModel

segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

def predict_natasha_per(text: str):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    spans = []
    for span in doc.spans: # type: ignore
        if span.type == PER:
            spans.append(Span(line_idx=0, begin=span.start, end=span.stop, label="PER", data=span.text))
    return spans

class NatashaPERModel(PIIModel):
    def predict(self, document: TextDocument) -> AnnotationSet:
        if not document.lines:
            return AnnotationSet(lines=(), idx=None)
        text = document.lines[0].text
        spans = predict_natasha_per(text)
        line = AnnotationSetLine(idx=0, role=document.lines[0].role, spans=spans)
        return AnnotationSet(lines=(line,), idx=document.sample_id)
