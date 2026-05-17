from __future__ import annotations

import unittest

from asr_integer_extractor.scoring import build_feature_vector, score_confidence


class ScoringTests(unittest.TestCase):
    def test_feature_vector_shape(self) -> None:
        features = build_feature_vector(
            token_count=3,
            fuzzy_count=1,
            mean_lexical_score=0.9,
            has_fraction_tail=False,
            kind="digit_sequence",
        )
        self.assertEqual(features.shape, (5,))

    def test_confidence_in_range(self) -> None:
        features = build_feature_vector(
            token_count=3,
            fuzzy_count=1,
            mean_lexical_score=0.9,
            has_fraction_tail=True,
            kind="cardinal",
        )
        confidence: float = score_confidence(features)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


__all__: list[str] = []
