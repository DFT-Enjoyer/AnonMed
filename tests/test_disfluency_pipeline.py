from __future__ import annotations

import json
import subprocess
import sys
import unittest

from anonmed.preprocessing import (
    ASRNormalizationPipeline,
    DisfluencyFilter,
    PunctuationFilterConfig,
    remove_punctuation,
    remove_disfluencies,
    run_asr_normalization,
)


class DisfluencyPipelineTests(unittest.TestCase):
    def test_remove_hesitations_and_phrase_fillers(self) -> None:
        cleaned_text: str = remove_disfluencies("Ну, эм, я как бы диктую номер один два три.")
        self.assertEqual(cleaned_text, "я диктую номер один два три.")

    def test_remove_punctuation_as_standalone_preprocessing_step(self) -> None:
        cleaned_text: str = remove_punctuation("Привет, мир! Номер: один-два.")
        self.assertEqual(cleaned_text, "Привет мир Номер один два")

    def test_remove_punctuation_keeps_number_formats(self) -> None:
        cleaned_text: str = remove_punctuation("цена 12.50 дата 17/05/2026 код 10-20")
        self.assertEqual(cleaned_text, "цена 12.50 дата 17/05/2026 код 10-20")

    def test_remove_punctuation_keeps_hardcoded_domain_suffixes(self) -> None:
        cleaned_text: str = remove_punctuation("сайт test.com, зеркало portal.ком!")
        self.assertEqual(cleaned_text, "сайт test.com зеркало portal.ком")

    def test_hyphenated_hesitation_is_removed(self) -> None:
        cleaned_text: str = remove_disfluencies("э-э номер девять ноль один")
        self.assertEqual(cleaned_text, "номер девять ноль один")

    def test_ambiguous_discourse_marker_is_context_limited(self) -> None:
        disfluency_filter = DisfluencyFilter()
        cleaned_leading: str = disfluency_filter.clean("а потом один").text
        cleaned_inner: str = disfluency_filter.clean("пять а шесть").text
        self.assertEqual(cleaned_leading, "потом один")
        self.assertEqual(cleaned_inner, "пять а шесть")

    def test_pipeline_runs_disfluency_filter_before_integer_replacement(self) -> None:
        result = run_asr_normalization("ну, эм, номер один два три")
        self.assertEqual(result.cleaned_text, "номер один два три")
        self.assertEqual(result.normalized_text, "номер 123")
        self.assertEqual([span.value for span in result.integer_spans], ["123"])
        self.assertGreaterEqual(len(result.removed_spans), 2)

    def test_pipeline_applies_punctuation_removal_after_disfluency_cleanup(self) -> None:
        result = run_asr_normalization("ну, эм, номер один, два, три.")
        self.assertEqual(result.cleaned_text, "номер один два три")
        self.assertEqual(result.normalized_text, "номер 123")

    def test_pipeline_can_disable_punctuation_removal(self) -> None:
        result = run_asr_normalization(
            "номер один, два.",
            punctuation_config=PunctuationFilterConfig(enabled=False),
        )
        self.assertEqual(result.cleaned_text, "номер один, два.")

    def test_pipeline_json_contains_all_layers(self) -> None:
        pipeline = ASRNormalizationPipeline()
        serialized: str = pipeline.to_json("ну один два", ensure_ascii=False)
        payload: dict[str, object] = json.loads(serialized)
        self.assertEqual(payload["cleaned_text"], "один два")
        self.assertEqual(payload["normalized_text"], "12")
        self.assertIn("removed_spans", payload)
        self.assertIn("integer_spans", payload)

    def test_cli_run_uses_complete_pipeline(self) -> None:
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-m", "anonmed.cli", "ну", "один", "два", "--run"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed_process.returncode, 0, msg=completed_process.stderr)
        self.assertEqual(completed_process.stdout.strip(), "12")


__all__: list[str] = []
