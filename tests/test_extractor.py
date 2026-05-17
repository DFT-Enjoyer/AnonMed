from __future__ import annotations

import unittest

from asr_integer_extractor import IntegerExtractor


class ExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = IntegerExtractor()

    def test_extract_many_numbers_from_text(self) -> None:
        text: str = "до 93243243243 потом 43249329 и текст5435453..."
        spans = self.extractor.extract(text)
        values: list[str] = [span.value for span in spans]
        raws: list[str] = [span.raw for span in spans]
        self.assertEqual(values, ["93243243243", "43249329", "5435453"])
        self.assertEqual(raws, ["93243243243", "43249329", "5435453"])

    def test_extract_words_and_digits_in_order(self) -> None:
        text: str = "номер девять три два четыре, потом 43249329 и сорок два"
        spans = self.extractor.extract(text)
        values: list[str] = [span.value for span in spans]
        kinds: list[str] = [span.kind for span in spans]
        self.assertEqual(values, ["9324", "43249329", "42"])
        self.assertEqual(kinds, ["digit_sequence", "digits", "cardinal"])

    def test_span_offsets_match_raw_text(self) -> None:
        text: str = "abc девять три два xyz"
        spans = self.extractor.extract(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(text[spans[0].start : spans[0].end], "девять три два")

    def test_extractor_does_not_eat_surrounding_words(self) -> None:
        text: str = "до двенадцать с половиной после"
        spans = self.extractor.extract(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].raw, "двенадцать с половиной")
        self.assertEqual(text[: spans[0].start], "до ")
        self.assertEqual(text[spans[0].end :], " после")


__all__: list[str] = []
