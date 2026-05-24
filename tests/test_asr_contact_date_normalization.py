from __future__ import annotations

import json
import unittest

from anonmed.anonymization import run_numeric_pii_pipeline
from anonmed.preprocessing import (
    ASRNormalizationPipeline,
    ContactNormalizer,
    DateBirthNormalizer,
    run_asr_normalization,
)


class ASRContactDateNormalizationTests(unittest.TestCase):
    def test_spoken_birth_date_month_is_normalized_after_integer_replacement(self) -> None:
        result = run_asr_normalization(
            "дату рождения назовите двенадцатое марта "
            "одна тысяча девятьсот восемьдесят четвертого года"
        )

        self.assertEqual(
            result.numeric_normalized_text,
            "дату рождения назовите 12 марта 1984 года",
        )
        self.assertEqual(
            result.normalized_text,
            "дату рождения назовите 12.03.1984 года",
        )
        self.assertEqual(len(result.date_birth_spans), 1)
        self.assertEqual(result.date_birth_spans[0].normalized, "12.03.1984")

    def test_spoken_birth_date_feeds_numeric_pii_pipeline(self) -> None:
        result = run_numeric_pii_pipeline(
            "родился пятое мая одна тысяча девятьсот восемьдесят пятого года"
        )

        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in result.matches],
            [("DATE_BIRTH", "05.05.1985")],
        )

    def test_spoken_birth_date_trims_repeated_year_tail(self) -> None:
        result = run_numeric_pii_pipeline(
            "дата рождения двенадцатое августа "
            "тысяча девятьсот семьдесят девятого семьдесят девятого года"
        )

        self.assertEqual(
            result.preprocessing_result.numeric_normalized_text,
            "дата рождения 12 августа 197979 года",
        )
        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in result.matches],
            [("DATE_BIRTH", "12.08.1979")],
        )

    def test_non_birth_month_date_is_not_rewritten_without_birth_context(self) -> None:
        result = run_asr_normalization(
            "дата приема двенадцатое марта две тысячи двадцать четвертого года"
        )

        self.assertEqual(
            result.normalized_text,
            "дата приема 12 марта 2024 года",
        )
        self.assertEqual(result.date_birth_spans, ())

    def test_spoken_email_is_normalized(self) -> None:
        result = run_asr_normalization(
            "электронная почта ковалев восемьдесят четыре собака мэйл точка ру"
        )

        self.assertEqual(
            result.normalized_text,
            "электронная почта kovalev84@mail.ru",
        )
        self.assertEqual(len(result.contact_spans), 1)
        self.assertEqual(result.contact_spans[0].kind, "email")

    def test_spoken_telegram_handle_is_normalized(self) -> None:
        result = run_asr_normalization(
            "телеграм собака ковалев нижнее подчеркивание а ан"
        )

        self.assertEqual(result.normalized_text, "телеграм @kovalev_aan")
        self.assertEqual(len(result.contact_spans), 1)
        self.assertEqual(result.contact_spans[0].kind, "telegram")

    def test_spoken_phone_plus_seven_is_normalized_to_plus_prefix(self) -> None:
        result = run_asr_normalization(
            "телефон плюс семь девять один три один два три четыре пять шесть семь"
        )

        self.assertEqual(result.numeric_normalized_text, "телефон +79131234567")
        self.assertEqual(result.normalized_text, "телефон +79131234567")

    def test_spoken_phone_without_plus_keeps_semantic_mobile_prefix(self) -> None:
        result = run_numeric_pii_pipeline(
            "номер телефона семь девять один три один два три четыре пять шесть семь"
        )

        self.assertEqual(
            result.preprocessing_result.normalized_text,
            "номер телефона 79131234567",
        )
        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in result.matches],
            [("PHONE", "+79131234567")],
        )

    def test_standalone_normalizers_return_audit_spans(self) -> None:
        date_normalizer = DateBirthNormalizer()
        contact_normalizer = ContactNormalizer()

        date_result = date_normalizer.normalize("дата рождения 5 мая 1985")
        contact_result = contact_normalizer.normalize("почта user собака яндекс точка ру")

        self.assertEqual(date_result.text, "дата рождения 05.05.1985")
        self.assertEqual(date_result.spans[0].raw, "5 мая 1985")
        self.assertEqual(contact_result.text, "почта user@yandex.ru")
        self.assertEqual(contact_result.spans[0].raw, "user собака яндекс точка ру")

    def test_pipeline_json_contains_contact_and_date_layers(self) -> None:
        pipeline = ASRNormalizationPipeline()
        payload: dict[str, object] = json.loads(
            pipeline.to_json(
                "дата рождения первое января две тысячи первого года "
                "почта тест собака мэйл точка ру",
                ensure_ascii=False,
            )
        )

        self.assertEqual(
            payload["normalized_text"],
            "дата рождения 01.01.2001 года почта test@mail.ru",
        )
        self.assertIn("date_birth_spans", payload)
        self.assertIn("contact_spans", payload)


__all__: list[str] = []
