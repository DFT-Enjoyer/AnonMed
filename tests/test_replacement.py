from __future__ import annotations

import unittest

from asr_integer_extractor import replace_integer_spans


class ReplacementTests(unittest.TestCase):
    def test_replace_keeps_other_text(self) -> None:
        text: str = "до девять три два после 432 конец"
        replaced: str = replace_integer_spans(text)
        self.assertEqual(replaced, "до 932 после 432 конец")

    def test_replace_keeps_adjacent_text(self) -> None:
        text: str = "abc5435453def"
        replaced: str = replace_integer_spans(text)
        self.assertEqual(replaced, "abc5435453def")

    def test_replace_fraction_span(self) -> None:
        text: str = "пример двенадцать с половиной дальше"
        replaced: str = replace_integer_spans(text)
        self.assertEqual(replaced, "пример 12 дальше")


__all__: list[str] = []
