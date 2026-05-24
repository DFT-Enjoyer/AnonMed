from __future__ import annotations

from typing import Any, Mapping

from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Span, TextDocument
from anonmed.ml.models.base import PIIModel


DEFAULT_MODEL_NAME = "fastino/gliner2-multi-v1"
DEFAULT_ENTITY_NAME = "person"
DEFAULT_ENTITY_DESCRIPTION = (
    "Full personal name of an individual in Russian text, also known as Russian ФИО. "
    "Extract the complete contiguous mention of a person's surname, given name, and patronymic "
    "when present, for example Иванов Иван Иванович. Return the whole person-name span as one "
    "entity; do not split it into surname, given name, patronymic, or initials. Do not extract "
    "organizations, job titles, medical roles, addresses, document numbers, phone numbers, dates, "
    "or other non-person PII."
)


def _load_gliner2_model(model_name: str, load_kwargs: Mapping[str, Any]) -> Any:
    try:
        from gliner2 import GLiNER2
    except ImportError as error:
        message = (
            "GLiNER2Model requires the 'gliner2' package. "
            "Install the ML extras or add 'gliner2' to the environment."
        )
        raise ImportError(message) from error

    return GLiNER2.from_pretrained(model_name, **dict(load_kwargs))


class GLiNER2Model(PIIModel):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        entity_name: str = DEFAULT_ENTITY_NAME,
        entity_description: str = DEFAULT_ENTITY_DESCRIPTION,
        label: str = "PER",
        threshold: float | None = 0.5,
        extractor: Any | None = None,
        load_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.entity_name = entity_name
        self.entity_description = entity_description
        self.label = label
        self.threshold = threshold
        self._extractor = extractor or _load_gliner2_model(model_name, load_kwargs or {})
        self._schema = self._build_schema()

    def predict(self, document: TextDocument) -> AnnotationSet:
        annotation_lines: list[AnnotationSetLine] = []
        for line in document.lines:
            spans = self._predict_line(line.text, line.idx)
            annotation_lines.append(AnnotationSetLine(idx=line.idx, role=line.role, spans=spans))
        return AnnotationSet(lines=tuple(annotation_lines), idx=document.sample_id)

    def _predict_line(self, text: str, line_idx: int) -> list[Span]:
        kwargs: dict[str, Any] = {
            "include_spans": True,
        }
        if self.threshold is not None:
            kwargs["threshold"] = self.threshold

        result = self._extractor.extract(text, self._schema, **kwargs)
        return [
            Span(
                line_idx=line_idx,
                begin=begin,
                end=end,
                label=self.label,
                data=entity_text,
            )
            for entity_text, begin, end in self._iter_entity_spans(text, result)
        ]

    def _build_schema(self) -> Any:
        entity_config: dict[str, object] = {
            "description": self.entity_description,
            "dtype": "list",
        }
        if self.threshold is not None:
            entity_config["threshold"] = self.threshold
        return self._extractor.create_schema().entities({self.entity_name: entity_config})

    def _iter_entity_spans(self, text: str, result: object) -> list[tuple[str, int, int]]:
        entities = _entities_for_name(result, self.entity_name)
        spans: list[tuple[str, int, int]] = []
        cursor = 0
        for entity in entities:
            entity_text, begin, end = _entity_span(text, entity, cursor)
            if begin < 0 or end > len(text) or begin >= end:
                continue
            spans.append((entity_text, begin, end))
            cursor = end
        return spans


def _entities_for_name(result: object, entity_name: str) -> list[object]:
    if isinstance(result, Mapping):
        entities = result.get("entities", result)
        if isinstance(entities, Mapping):
            raw_items = entities.get(entity_name, ())
            if isinstance(raw_items, list):
                return raw_items
            if raw_items:
                return [raw_items]
    return []


def _entity_span(text: str, entity: object, cursor: int) -> tuple[str, int, int]:
    if isinstance(entity, Mapping):
        entity_text = str(entity.get("text", ""))
        if "start" in entity and "end" in entity:
            begin = int(entity["start"])
            end = int(entity["end"])
            return entity_text or text[begin:end], begin, end
        return _find_entity_text(text, entity_text, cursor)
    return _find_entity_text(text, str(entity), cursor)


def _find_entity_text(text: str, entity_text: str, cursor: int) -> tuple[str, int, int]:
    if not entity_text:
        return "", -1, -1
    begin = text.find(entity_text, cursor)
    if begin < 0:
        begin = text.find(entity_text)
    if begin < 0:
        return entity_text, -1, -1
    return entity_text, begin, begin + len(entity_text)


__all__ = [
    "DEFAULT_ENTITY_DESCRIPTION",
    "DEFAULT_ENTITY_NAME",
    "DEFAULT_MODEL_NAME",
    "GLiNER2Model",
]
