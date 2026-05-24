from __future__ import annotations

import unittest

from anonmed.preprocessing import (
    ASRRepeatDeduplicationConfig,
    ASRUtterance,
    deduplicate_asr_utterances,
)


class ASRRepetitionDeduplicationTests(unittest.TestCase):
    def test_exact_duplicate_without_speaker_labels_is_suppressed_in_clean_layer(self) -> None:
        result = deduplicate_asr_utterances(["алло", "алло", "меня слышно"])

        self.assertEqual(result.raw_transcript, "алло алло меня слышно")
        self.assertEqual(result.clean_transcript, "алло меня слышно")
        self.assertEqual(result.suppressed_indexes, (1,))
        self.assertEqual(result.utterances[1].reason, "exact_duplicate")

    def test_overlap_boundary_duplicate_is_suppressed(self) -> None:
        utterances: list[ASRUtterance] = [
            ASRUtterance("номер полиса 123", start=0.0, end=4.0),
            ASRUtterance("номер полиса 123", start=3.9, end=6.0),
        ]

        result = deduplicate_asr_utterances(utterances)

        self.assertEqual(result.clean_transcript, "номер полиса 123")
        self.assertEqual(result.utterances[1].reason, "chunk_overlap")

    def test_more_complete_partial_repeat_supersedes_previous_short_turn(self) -> None:
        result = deduplicate_asr_utterances(
            ["адрес семь", "адрес семь квартира двенадцать"]
        )

        self.assertEqual(
            result.clean_transcript,
            "адрес семь квартира двенадцать",
        )
        self.assertEqual(result.suppressed_indexes, (0,))
        self.assertEqual(result.utterances[0].reason, "superseded_by_more_complete_repeat")
        self.assertEqual(result.utterances[0].duplicate_of, 1)

    def test_numeric_delta_without_repair_marker_is_preserved(self) -> None:
        result = deduplicate_asr_utterances(["код полиса 123", "код полиса 124"])

        self.assertEqual(result.clean_transcript, "код полиса 123 код полиса 124")
        self.assertEqual(result.suppressed_indexes, ())

    def test_repair_marker_allows_newer_numeric_turn_to_supersede_previous(self) -> None:
        result = deduplicate_asr_utterances(
            ["код полиса 123", "нет код полиса 124"]
        )

        self.assertEqual(result.clean_transcript, "нет код полиса 124")
        self.assertEqual(result.suppressed_indexes, (0,))
        self.assertEqual(result.utterances[0].reason, "superseded_by_repair")

    def test_distant_duplicate_outside_time_window_is_kept(self) -> None:
        utterances: list[ASRUtterance] = [
            ASRUtterance("аллергия на пенициллин", start=0.0, end=2.0),
            ASRUtterance("аллергия на пенициллин", start=120.0, end=122.0),
        ]
        config = ASRRepeatDeduplicationConfig(max_time_gap_seconds=15.0)

        result = deduplicate_asr_utterances(utterances, config=config)

        self.assertEqual(
            result.clean_transcript,
            "аллергия на пенициллин аллергия на пенициллин",
        )
        self.assertEqual(result.suppressed_indexes, ())

    def test_cross_speaker_policy_ignores_same_speaker_repeats(self) -> None:
        utterances: list[ASRUtterance] = [
            ASRUtterance("повторите пожалуйста", speaker="doctor"),
            ASRUtterance("повторите пожалуйста", speaker="doctor"),
        ]
        config = ASRRepeatDeduplicationConfig(speaker_policy="cross")

        result = deduplicate_asr_utterances(utterances, config=config)

        self.assertEqual(
            result.clean_transcript,
            "повторите пожалуйста повторите пожалуйста",
        )
        self.assertEqual(result.suppressed_indexes, ())


__all__: list[str] = []
