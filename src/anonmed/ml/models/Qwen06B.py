from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import json
from pathlib import Path
import re
from typing import Any, TypeAlias

from anonmed.ml.core.types import AnnotationSet, AnnotationSetLine, Span, TextDocument
from anonmed.ml.models.base import PIIModel


DEFAULT_MODEL_PATH: str = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS: int = 256
# Обновлённый промпт: убраны все типы кроме PER, примеры адаптированы
DEFAULT_SYSTEM_PROMPT: str = """/no_think
Ты — эксперт по поиску персональных данных (ФИО) в тексте.
Возвращай ТОЛЬКО валидный JSON вида:
{"entities": [{"text": "...", "type": "PER"}]}

Если ФИО нет — {"entities": []}
Никаких пояснений, markdown-блоков или дополнительного текста.
Игнорируй телефоны, email, адреса, организации и паспорта.

Примеры:

Вход: "Встретился с Петром Сергеевичем в кафе на Арбате"
Выход: {"entities": [{"text": "Петром Сергеевичем", "type": "PER"}]}

Вход: "Звонила Марина из ООО Ромашка, оставила номер 89165551234"
Выход: {"entities": [{"text": "Марина", "type": "PER"}]}

Вход: "Документы лежат на столе"
Выход: {"entities": []}

Вход: "Отправь на ivanov@mail.ru и petrov.s@gmail.com"
Выход: {"entities": []}

Вход: "Анна Владимировна и Иван Петрович пришли на встречу"
Выход: {"entities": [{"text": "Анна Владимировна", "type": "PER"}, {"text": "Иван Петрович", "type": "PER"}]}"""

# Ограничиваем разрешённые типы только PER
DEFAULT_ALLOWED_ENTITY_TYPES: frozenset[str] = frozenset({"PER"})
DEFAULT_LABEL_MAPPING: dict[str, str] = {}

JsonObject: TypeAlias = Mapping[str, Any]
RawEntity: TypeAlias = Mapping[str, Any]


class Qwen06BModel(PIIModel):
    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        tokenizer: Any | None = None,
        model: Any | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        allowed_entity_types: Sequence[str] | None = tuple(DEFAULT_ALLOWED_ENTITY_TYPES),
        label_mapping: Mapping[str, str] | None = None,
        enable_thinking: bool | None = False,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be provided together or omitted together.")

        self.model_path: str = str(model_path)
        self.system_prompt: str = system_prompt
        self.max_new_tokens: int = max_new_tokens
        self.generation_kwargs: dict[str, Any] = dict(generation_kwargs or {})
        self.allowed_entity_types: frozenset[str] | None = (
            frozenset(allowed_entity_types) if allowed_entity_types is not None else None
        )
        self.label_mapping: dict[str, str] = dict(
            DEFAULT_LABEL_MAPPING if label_mapping is None else label_mapping
        )
        self.enable_thinking: bool | None = enable_thinking

        if tokenizer is not None and model is not None:
            self.tokenizer: Any = tokenizer
            self.model: Any = model
        else:
            loaded_model, loaded_tokenizer = _load_qwen06b_model(
                self.model_path,
                tokenizer_kwargs=tokenizer_kwargs or {},
                model_kwargs=model_kwargs or {},
            )
            self.model = loaded_model
            self.tokenizer = loaded_tokenizer

    def predict(self, document: TextDocument) -> AnnotationSet:
        annotation_lines: list[AnnotationSetLine] = []
        for line in document.lines:
            spans: list[Span] = self._predict_line(line.text, line.idx)
            annotation_line: AnnotationSetLine = AnnotationSetLine(
                idx=line.idx,
                role=line.role,
                spans=spans,
            )
            annotation_lines.append(annotation_line)
        return AnnotationSet(lines=tuple(annotation_lines), idx=document.sample_id)

    def _predict_line(self, text: str, line_idx: int) -> list[Span]:
        response: str = self._generate_response(text)
        raw_entities: list[RawEntity] = parse_qwen06b_entities(
            response,
            allowed_entity_types=self.allowed_entity_types,
        )
        spans: list[Span] = []
        cursor: int = 0
        for raw_entity in raw_entities:
            entity_text: str = str(raw_entity.get("text", ""))
            entity_type: str = str(raw_entity.get("type", ""))
            label: str = self.label_mapping.get(entity_type, entity_type)
            begin, end = _entity_offsets(text, raw_entity, entity_text, cursor)
            if begin < 0 or end > len(text) or begin >= end:
                continue
            span: Span = Span(
                line_idx=line_idx,
                begin=begin,
                end=end,
                label=label,
                data=text[begin:end],
            )
            spans.append(span)
            cursor = end
        return spans

    def _generate_response(self, text: str) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        formatted_prompt: str = _format_chat_prompt(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )
        inputs: Any = self.tokenizer([formatted_prompt], return_tensors="pt")
        device: Any = _infer_model_device(self.model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)

        model_inputs: Mapping[str, Any] = _as_mapping(inputs)
        input_length: int = _input_length(model_inputs)
        generation_kwargs: dict[str, Any] = self._generation_kwargs()

        with _no_grad_context():
            outputs: Any = self.model.generate(**model_inputs, **generation_kwargs)

        new_tokens: Any = _generated_tokens(outputs, input_length)
        response: str = str(
            self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        ).strip()
        return response

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        kwargs.update(self.generation_kwargs)
        return kwargs


def parse_qwen06b_entities(
    response: str,
    *,
    allowed_entity_types: frozenset[str] | None = DEFAULT_ALLOWED_ENTITY_TYPES,
) -> list[RawEntity]:
    data: JsonObject | None = _loads_json_object(response)
    if data is None:
        return []

    raw_entities: Any = data.get("entities", [])
    if not isinstance(raw_entities, list):
        return []

    entities: list[RawEntity] = []
    for raw_entity in raw_entities:
        if not isinstance(raw_entity, Mapping):
            continue
        entity_text_value: Any = raw_entity.get("text")
        entity_type_value: Any = raw_entity.get("type")
        if not isinstance(entity_text_value, str) or not entity_text_value:
            continue
        if not isinstance(entity_type_value, str) or not entity_type_value:
            continue
        if allowed_entity_types is not None and entity_type_value not in allowed_entity_types:
            continue
        entities.append(raw_entity)
    return entities


def _load_qwen06b_model(
    model_path: str,
    *,
    tokenizer_kwargs: Mapping[str, Any],
    model_kwargs: Mapping[str, Any],
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        message: str = (
            "Qwen06BModel requires the 'transformers' package. "
            "Install transformers and torch, or pass preloaded tokenizer and model."
        )
        raise ImportError(message) from error

    # Изменено: больше не форсируем local_files_only=True
    # Пользователь может передать его явно через kwargs, если нужно
    resolved_tokenizer_kwargs: dict[str, Any] = dict(tokenizer_kwargs)
    resolved_model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    resolved_model_kwargs.update(dict(model_kwargs))

    tokenizer: Any = AutoTokenizer.from_pretrained(model_path, **resolved_tokenizer_kwargs)
    model: Any = AutoModelForCausalLM.from_pretrained(model_path, **resolved_model_kwargs)
    return model, tokenizer


def _format_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    enable_thinking: bool | None,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        prompt: Any = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise
        kwargs.pop("enable_thinking")
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    return str(prompt)


def _loads_json_object(response: str) -> JsonObject | None:
    try:
        data: Any = json.loads(response)
    except json.JSONDecodeError:
        data = _find_first_json_object(response)
    if isinstance(data, Mapping):
        return data
    return None


def _find_first_json_object(response: str) -> JsonObject | None:
    decoder: json.JSONDecoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        start: int = match.start()
        try:
            data, _ = decoder.raw_decode(response[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, Mapping):
            return data
    return None


def _entity_offsets(
    text: str,
    raw_entity: RawEntity,
    entity_text: str,
    cursor: int,
) -> tuple[int, int]:
    span_value: Any = raw_entity.get("span")
    if isinstance(span_value, Sequence) and not isinstance(span_value, (str, bytes, bytearray)):
        if len(span_value) == 2:
            begin_candidate: Any = span_value[0]
            end_candidate: Any = span_value[1]
            if isinstance(begin_candidate, int) and isinstance(end_candidate, int):
                if _valid_offsets(text, begin_candidate, end_candidate, entity_text):
                    return begin_candidate, end_candidate

    start_value: Any = raw_entity.get("start", raw_entity.get("begin"))
    end_value: Any = raw_entity.get("end")
    if isinstance(start_value, int) and isinstance(end_value, int):
        if _valid_offsets(text, start_value, end_value, entity_text):
            return start_value, end_value

    return _find_entity_text(text, entity_text, cursor)


def _valid_offsets(text: str, begin: int, end: int, entity_text: str) -> bool:
    if not 0 <= begin < end <= len(text):
        return False
    if not entity_text:
        return True
    return text[begin:end] == entity_text


def _find_entity_text(text: str, entity_text: str, cursor: int) -> tuple[int, int]:
    if not entity_text:
        return -1, -1
    begin: int = text.find(entity_text, cursor)
    if begin < 0:
        begin = text.find(entity_text)
    if begin < 0:
        return -1, -1
    return begin, begin + len(entity_text)


def _infer_model_device(model: Any) -> Any:
    try:
        parameter_iterator: Any = iter(model.parameters())
        parameter: Any = next(parameter_iterator)
    except (AttributeError, StopIteration, TypeError):
        return getattr(model, "device", "cpu")
    return getattr(parameter, "device", "cpu")


def _as_mapping(inputs: Any) -> Mapping[str, Any]:
    if not isinstance(inputs, Mapping):
        raise TypeError(f"Tokenizer output must be mapping-like, got {type(inputs).__name__}")
    return inputs


def _input_length(inputs: Mapping[str, Any]) -> int:
    input_ids: Any = inputs.get("input_ids")
    shape: Any = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    if isinstance(input_ids, Sequence) and input_ids:
        first_row: Any = input_ids[0]
        if isinstance(first_row, Sequence):
            return len(first_row)
    raise TypeError("Unable to determine prompt token length from tokenizer output.")


def _generated_tokens(outputs: Any, input_length: int) -> Any:
    first_sequence: Any = outputs[0]
    return first_sequence[input_length:]


def _no_grad_context() -> Any:
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


__all__: list[str] = [
    "DEFAULT_ALLOWED_ENTITY_TYPES",
    "DEFAULT_LABEL_MAPPING",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SYSTEM_PROMPT",
    "Qwen06BModel",
    "parse_qwen06b_entities",
]
