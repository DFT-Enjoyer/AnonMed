from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Span, TextDocument
from anonmed.ml.models.base import PIIModel

_NATASHA_COMPONENTS = None


def _load_natasha_components():
    global _NATASHA_COMPONENTS

    if _NATASHA_COMPONENTS is not None:
        return _NATASHA_COMPONENTS

    try:
        from natasha import Doc, NewsEmbedding, NewsNERTagger, PER, Segmenter
    except ImportError as error:
        message = (
            "NatashaPERModel requires the 'natasha' package. "
            "Install the ML extras or add 'natasha>=1.6' to the environment."
        )
        raise ImportError(message) from error

    segmenter = Segmenter()
    embedding = NewsEmbedding()
    ner_tagger = NewsNERTagger(embedding)
    _NATASHA_COMPONENTS = (Doc, PER, segmenter, ner_tagger)
    return _NATASHA_COMPONENTS

def predict_natasha_per(text: str):
    Doc, PER, segmenter, ner_tagger = _load_natasha_components()
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
