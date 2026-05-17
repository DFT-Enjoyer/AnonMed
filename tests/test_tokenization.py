from __future__ import annotations

import unittest

from asr_integer_extractor.normalization import tokenize_preserving_spans


class TokenizationTests(unittest.TestCase):
    def test_digits_adjacent_to_text_keep_positions(self) -> None:
        text: str = "текст5435453..."
        tokens = tokenize_preserving_spans(text)
        digit_tokens = [token for token in tokens if token.kind == "digits"]
        self.assertEqual(len(digit_tokens), 1)
        self.assertEqual(digit_tokens[0].text, "5435453")
        self.assertEqual(text[digit_tokens[0].start : digit_tokens[0].end], "5435453")

    def test_tokenizer_does_not_delete_text(self) -> None:
        text: str = "до девять три два после"
        tokens = tokenize_preserving_spans(text)
        reconstructed: str = "".join(text[token.start : token.end] for token in tokens)
        self.assertEqual(reconstructed, "додевятьтридвапосле")


__all__: list[str] = []
