from __future__ import annotations

import unittest

from asr_integer_extractor import IntegerExtractor


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = IntegerExtractor()

    def values(self, text: str) -> list[str]:
        values: list[str] = [span.value for span in self.extractor.extract(text)]
        return values

    def test_cardinal_numbers(self) -> None:
        self.assertEqual(self.values("сорок два"), ["42"])
        self.assertEqual(self.values("одна тысяча пять"), ["1005"])
        self.assertEqual(self.values("девятьсот тридцать два"), ["932"])

    def test_digit_sequence_preferred(self) -> None:
        self.assertEqual(self.values("девять три два четыре"), ["9324"])
        self.assertEqual(self.values("один два три"), ["123"])
        self.assertEqual(self.values("ноль пять шесть"), ["056"])

    def test_mixed_asr_output(self) -> None:
        self.assertEqual(self.values("двадцать 5"), ["25"])
        self.assertEqual(self.values("сто 5"), ["105"])
        self.assertEqual(self.values("минус семь"), ["-7"])

    def test_fraction_tail_truncated_to_integer(self) -> None:
        self.assertEqual(self.values("двенадцать с половиной"), ["12"])
        self.assertEqual(self.values("12 с половиной"), ["12"])

    def test_fuzzy_asr_noise(self) -> None:
        spans = self.extractor.extract("двадцат пять")
        self.assertEqual([span.value for span in spans], ["25"])
        self.assertEqual(spans[0].status, "fuzzy_ok")

    def test_non_numeric_words_are_not_overcorrected(self) -> None:
        self.assertEqual(self.values("стоп здесь"), [])


__all__: list[str] = []
