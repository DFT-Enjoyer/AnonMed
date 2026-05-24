from __future__ import annotations

import unittest

from anonmed.anonymization import PostProcessedPIIMention, restore_safe_original_text


def _mention(
    *,
    original_start: int,
    original_end: int,
    normalized_start: int,
    normalized_end: int,
    projection_status: str = "ok",
) -> PostProcessedPIIMention:
    return PostProcessedPIIMention(
        entity_type="PHONE",
        original_start=original_start,
        original_end=original_end,
        normalized_start=normalized_start,
        normalized_end=normalized_end,
        original_text="восемь девять один три один два три четыре пять шесть семь",
        normalized_text="89131234567",
        normalized_value="+79131234567",
        replacement="[PHONE]",
        confidence=0.99,
        rule_id="phone_mobile_11_digits",
        source="regex",
        entity_id="PHONE:0001",
        mention_id="PHONE:0001:m01",
        projection_status=projection_status,
        metadata={},
    )


class OriginalTextRestorationTests(unittest.TestCase):
    def test_restore_masks_original_layer_but_preserves_non_pii_original_text(self) -> None:
        original_text: str = (
            "ну, телефон: восемь девять один три один два три четыре пять шесть семь!"
        )
        normalized_text: str = "телефон 89131234567"
        result = restore_safe_original_text(
            original_text=original_text,
            normalized_text=normalized_text,
            mentions=(
                _mention(
                    original_start=13,
                    original_end=71,
                    normalized_start=8,
                    normalized_end=19,
                ),
            ),
        )

        self.assertTrue(result.is_safe)
        self.assertEqual(result.safe_text, "ну, телефон: [PHONE]!")
        self.assertEqual(result.masked_normalized_text, "телефон [PHONE]")
        self.assertEqual(result.audit["used_mention_count"], 1)

    def test_restore_marks_result_unsafe_when_projection_is_not_restorable(self) -> None:
        result = restore_safe_original_text(
            original_text="телефон восемь девять один три",
            normalized_text="телефон 8913",
            mentions=(
                _mention(
                    original_start=8,
                    original_end=31,
                    normalized_start=8,
                    normalized_end=12,
                    projection_status="failed",
                ),
            ),
        )

        self.assertFalse(result.is_safe)
        self.assertEqual(result.safe_text, "телефон восемь девять один три")
        self.assertEqual(result.audit["skipped_mention_count"], 1)


__all__: list[str] = []
