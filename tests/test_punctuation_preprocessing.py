from __future__ import annotations

import json
import subprocess
import sys
import unittest

from asr_integer_extractor import (
    ASRNormalizationPipeline,
    PunctuationRemovalConfig,
    PunctuationRemover,
    remove_punctuation,
    run_asr_normalization,
)


class PunctuationPreprocessingTests(unittest.TestCase):
    def test_basic_sentence_punctuation_is_removed(self) -> None:
        cleaned_text: str = remove_punctuation("Привет, мир! Это тест.")
        self.assertEqual(cleaned_text, "Привет мир Это тест")

    def test_colon_and_semicolon_are_removed_as_boundaries(self) -> None:
        cleaned_text: str = remove_punctuation("номер: один; код: два")
        self.assertEqual(cleaned_text, "номер один код два")

    def test_repeated_punctuation_collapses_to_single_space(self) -> None:
        cleaned_text: str = remove_punctuation("ну... номер!!! один???")
        self.assertEqual(cleaned_text, "ну номер один")

    def test_parentheses_and_quotes_are_removed(self) -> None:
        cleaned_text: str = remove_punctuation("он сказал: «номер (один два)»")
        self.assertEqual(cleaned_text, "он сказал номер один два")

    def test_number_decimal_dot_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("значение 12.5 дальше")
        self.assertEqual(cleaned_text, "значение 12.5 дальше")

    def test_number_decimal_comma_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("значение 12,5 дальше")
        self.assertEqual(cleaned_text, "значение 12,5 дальше")

    def test_date_dots_are_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("дата 01.02.2024 записана")
        self.assertEqual(cleaned_text, "дата 01.02.2024 записана")

    def test_time_colon_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("время 12:30 точное")
        self.assertEqual(cleaned_text, "время 12:30 точное")

    def test_phone_hyphens_between_digits_are_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("телефон 8-900-123-45-67 указан")
        self.assertEqual(cleaned_text, "телефон 8-900-123-45-67 указан")

    def test_fraction_slash_between_digits_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("доля 1/2 примерно")
        self.assertEqual(cleaned_text, "доля 1/2 примерно")

    def test_dash_between_words_is_removed(self) -> None:
        cleaned_text: str = remove_punctuation("тест-система готова")
        self.assertEqual(cleaned_text, "тест система готова")

    def test_latin_com_domain_dot_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("сайт example.com работает")
        self.assertEqual(cleaned_text, "сайт example.com работает")

    def test_cyrillic_com_domain_dot_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("сайт пример.ком работает")
        self.assertEqual(cleaned_text, "сайт пример.ком работает")

    def test_ru_domain_dot_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("сайт example.ru работает")
        self.assertEqual(cleaned_text, "сайт example.ru работает")

    def test_rf_domain_dot_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("сайт пример.рф работает")
        self.assertEqual(cleaned_text, "сайт пример.рф работает")

    def test_www_domain_dots_are_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("адрес www.example.com указан")
        self.assertEqual(cleaned_text, "адрес www.example.com указан")

    def test_domain_trailing_comma_is_removed(self) -> None:
        cleaned_text: str = remove_punctuation("сайт example.com, дальше текст")
        self.assertEqual(cleaned_text, "сайт example.com дальше текст")

    def test_domain_trailing_period_is_removed(self) -> None:
        cleaned_text: str = remove_punctuation("сайт example.com. дальше")
        self.assertEqual(cleaned_text, "сайт example.com дальше")

    def test_email_punctuation_is_preserved(self) -> None:
        cleaned_text: str = remove_punctuation("почта test.user+one@example.com указана")
        self.assertEqual(cleaned_text, "почта test.user+one@example.com указана")

    def test_email_trailing_comma_is_removed(self) -> None:
        cleaned_text: str = remove_punctuation("почта test@example.com, далее")
        self.assertEqual(cleaned_text, "почта test@example.com далее")

    def test_url_punctuation_is_preserved_inside_url(self) -> None:
        cleaned_text: str = remove_punctuation("ссылка https://example.com/a-b?q=1, далее")
        self.assertEqual(cleaned_text, "ссылка https://example.com/a-b?q=1 далее")

    def test_url_trailing_period_is_removed(self) -> None:
        cleaned_text: str = remove_punctuation("ссылка https://example.com/path.")
        self.assertEqual(cleaned_text, "ссылка https://example.com/path")

    def test_number_separator_can_be_disabled(self) -> None:
        config = PunctuationRemovalConfig(preserve_numeric_separators=False)
        cleaned_text: str = remove_punctuation("значение 12.5", config=config)
        self.assertEqual(cleaned_text, "значение 12 5")

    def test_domain_preservation_can_be_disabled(self) -> None:
        config = PunctuationRemovalConfig(preserve_domains=False)
        cleaned_text: str = remove_punctuation("сайт example.com", config=config)
        self.assertEqual(cleaned_text, "сайт example com")

    def test_email_preservation_can_be_disabled(self) -> None:
        config = PunctuationRemovalConfig(preserve_emails=False, preserve_domains=False)
        cleaned_text: str = remove_punctuation("почта test@example.com", config=config)
        self.assertEqual(cleaned_text, "почта test example com")

    def test_sentence_final_punctuation_can_be_preserved(self) -> None:
        config = PunctuationRemovalConfig(preserve_sentence_final_punctuation=True)
        cleaned_text: str = remove_punctuation("номер: один.", config=config)
        self.assertEqual(cleaned_text, "номер один.")

    def test_removed_span_audit_trail_keeps_original_boundaries(self) -> None:
        remover = PunctuationRemover()
        result = remover.clean("тест, пример")
        self.assertEqual(result.text, "тест пример")
        self.assertEqual(len(result.removed_spans), 1)
        self.assertEqual(result.removed_spans[0].start, 4)
        self.assertEqual(result.removed_spans[0].end, 5)
        self.assertEqual(result.removed_spans[0].raw, ",")

    def test_protected_span_audit_trail_marks_domain(self) -> None:
        remover = PunctuationRemover()
        result = remover.clean("сайт example.com")
        reasons: set[str] = {span.reason for span in result.protected_spans}
        self.assertIn("domain", reasons)

    def test_pipeline_removes_punctuation_before_integer_replacement(self) -> None:
        result = run_asr_normalization("ну, номер: один два три.")
        self.assertEqual(result.cleaned_text, "номер один два три")
        self.assertEqual(result.normalized_text, "номер 123")
        self.assertGreaterEqual(len(result.punctuation_removed_spans), 1)

    def test_pipeline_preserves_decimal_separator(self) -> None:
        result = run_asr_normalization("значение 12.5, потом один два")
        self.assertEqual(result.cleaned_text, "значение 12.5 потом один два")
        self.assertEqual(result.normalized_text, "значение 12.5 потом 12")

    def test_pipeline_preserves_domain_punctuation(self) -> None:
        result = run_asr_normalization("ну, сайт example.com, код один два")
        self.assertEqual(result.cleaned_text, "сайт example.com код один два")
        self.assertEqual(result.normalized_text, "сайт example.com код 12")

    def test_pipeline_can_disable_punctuation_removal(self) -> None:
        pipeline = ASRNormalizationPipeline(remove_punctuation=False)
        result = pipeline.run("номер: один два")
        self.assertEqual(result.cleaned_text, "номер: один два")
        self.assertEqual(result.normalized_text, "номер: 12")

    def test_pipeline_json_contains_punctuation_layers(self) -> None:
        pipeline = ASRNormalizationPipeline()
        serialized: str = pipeline.to_json("номер: один два", ensure_ascii=False)
        payload: dict[str, object] = json.loads(serialized)
        self.assertEqual(payload["cleaned_text"], "номер один два")
        self.assertIn("punctuation_removed_spans", payload)
        self.assertIn("punctuation_protected_spans", payload)

    def test_cli_punctuation_clean(self) -> None:
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-m",
                "asr_integer_extractor.cli",
                "сайт",
                "example.com,",
                "номер:",
                "один",
                "--punctuation-clean",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed_process.returncode, 0, msg=completed_process.stderr)
        self.assertEqual(completed_process.stdout.strip(), "сайт example.com номер один")

    def test_cli_run_removes_punctuation_and_preserves_domain(self) -> None:
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-m",
                "asr_integer_extractor.cli",
                "ну,",
                "сайт",
                "example.com,",
                "код:",
                "один",
                "два",
                "--run",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed_process.returncode, 0, msg=completed_process.stderr)
        self.assertEqual(completed_process.stdout.strip(), "сайт example.com код 12")

    def test_cli_run_can_keep_punctuation(self) -> None:
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            [
                sys.executable,
                "-m",
                "asr_integer_extractor.cli",
                "номер:",
                "один",
                "два",
                "--run",
                "--keep-punctuation",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed_process.returncode, 0, msg=completed_process.stderr)
        self.assertEqual(completed_process.stdout.strip(), "номер: 12")


__all__: list[str] = []
