from __future__ import annotations

import unittest
from typing import Any, Mapping, Sequence

from anonmed.ml.core.types import Role, TextDocument, TextLine
from anonmed.ml.models.Qwen06B import Qwen06BModel, parse_qwen06b_entities


class _FakeTokenizer:
    eos_token_id: int = 0

    def __init__(self, response: str) -> None:
        self.response: str = response
        self.messages: Sequence[Mapping[str, str]] | None = None
        self.template_kwargs: dict[str, Any] | None = None
        self.prompts: list[str] | None = None
        self.return_tensors: str | None = None
        self.decoded_tokens: Any | None = None
        self.skip_special_tokens: bool | None = None

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        **kwargs: Any,
    ) -> str:
        self.messages = messages
        self.template_kwargs = kwargs
        return "formatted prompt"

    def __call__(self, prompts: list[str], *, return_tensors: str) -> dict[str, Any]:
        self.prompts = prompts
        self.return_tensors = return_tensors
        return {"input_ids": [[101, 102, 103]]}

    def decode(self, tokens: Any, *, skip_special_tokens: bool) -> str:
        self.decoded_tokens = tokens
        self.skip_special_tokens = skip_special_tokens
        return self.response


class _FakeModel:
    def __init__(self) -> None:
        self.generated_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generated_kwargs = kwargs
        return [[101, 102, 103, 201, 202]]


class Qwen06BModelTests(unittest.TestCase):
    def test_parse_entities_extracts_json_object_from_noisy_response(self) -> None:
        response: str = (
            "лишний текст "
            '{"entities": [{"text": "Анна", "type": "PER"}, '
            '{"text": "не то", "type": "UNKNOWN"}]} '
            "после json"
        )

        entities = parse_qwen06b_entities(response)

        self.assertEqual(entities, [{"text": "Анна", "type": "PER"}])

    # def test_predict_maps_per_and_legacy_fio_to_per_spans(self) -> None:
    #     text: str = "Звонила Марина из ООО Ромашка."
    #     response: str = (
    #         '{"entities": ['
    #         '{"text": "Марина", "type": "FIO"}, '
    #         '{"text": "ООО Ромашка", "type": "ORG"}'
    #         "]}"
    #     )
    #     tokenizer = _FakeTokenizer(response=response)
    #     fake_model = _FakeModel()
    #     model = Qwen06BModel(tokenizer=tokenizer, model=fake_model)
    #     role = Role(name="client")
    #     document = TextDocument(
    #         lines=(TextLine(idx=2, role=role, text=text),),
    #         sample_id="sample-1",
    #     )

    #     prediction = model.predict(document)

    #     self.assertEqual(prediction.idx, "sample-1")
    #     self.assertEqual(len(prediction.lines), 1)
    #     spans = prediction.lines[0].spans
    #     self.assertEqual(len(spans), 2)
    #     self.assertEqual((spans[0].begin, spans[0].end, spans[0].label), (8, 14, "PER"))
    #     self.assertEqual(spans[0].data, "Марина")
    #     self.assertEqual((spans[1].begin, spans[1].end, spans[1].label), (18, 29, "ORG"))
    #     self.assertEqual(spans[1].data, "ООО Ромашка")
    #     self.assertEqual(tokenizer.template_kwargs, {
    #         "tokenize": False,
    #         "add_generation_prompt": True,
    #         "enable_thinking": False,
    #     })
    #     self.assertEqual(tokenizer.prompts, ["formatted prompt"])
    #     self.assertEqual(tokenizer.return_tensors, "pt")
    #     self.assertEqual(tokenizer.decoded_tokens, [201, 202])
    #     self.assertTrue(tokenizer.skip_special_tokens)
    #     self.assertIsNotNone(fake_model.generated_kwargs)
    #     self.assertEqual(fake_model.generated_kwargs["max_new_tokens"], 256) # type: ignore
    #     self.assertEqual(fake_model.generated_kwargs["do_sample"], False) # type: ignore
    #     self.assertEqual(fake_model.generated_kwargs["pad_token_id"], 0) # type: ignore

    def test_predict_uses_explicit_span_offsets_when_valid(self) -> None:
        text: str = "Пациент Петр Петрович пришел. Потом Петр ушел."
        response: str = (
            '{"entities": ['
            '{"text": "Петр", "type": "PER", "span": [36, 40]}'
            "]}"
        )
        tokenizer = _FakeTokenizer(response=response)
        fake_model = _FakeModel()
        model = Qwen06BModel(tokenizer=tokenizer, model=fake_model)
        role = Role(name="client")
        document = TextDocument(lines=(TextLine(idx=0, role=role, text=text),))

        prediction = model.predict(document)
        span = prediction.lines[0].spans[0]

        self.assertEqual((span.begin, span.end, span.label, span.data), (36, 40, "PER", "Петр"))

    def test_model_requires_tokenizer_and_model_together(self) -> None:
        tokenizer = _FakeTokenizer(response='{"entities": []}')

        with self.assertRaisesRegex(ValueError, "tokenizer and model"):
            Qwen06BModel(tokenizer=tokenizer)


__all__: list[str] = []
