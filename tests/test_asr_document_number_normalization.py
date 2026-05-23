from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from anonmed.anonymization import run_numeric_pii_pipeline
from anonmed.preprocessing import (
    ASRNormalizationPipeline,
    DocumentNumberNormalizer,
    DocumentNumberNormalizerConfig,
    run_asr_normalization,
)


class ASRDocumentNumberNormalizationTests(unittest.TestCase):
    def test_repeated_spoken_phone_digits_are_split_before_pii_matching(self) -> None:
        result = run_numeric_pii_pipeline(
            "телефон для связи восемь девять один семь четыре четыре четыре пять пять два два "
            "восемь девять один семь четыре четыре четыре пять пять два два",
            normalize_document_numbers=True,
        )

        self.assertEqual(
            result.preprocessing_result.numeric_normalized_text,
            "телефон для связи 8917444552289174445522",
        )
        self.assertEqual(
            result.preprocessing_result.normalized_text,
            "телефон для связи 89174445522, 89174445522",
        )
        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in result.matches],
            [("PHONE", "+79174445522"), ("PHONE", "+79174445522")],
        )

    def test_repeated_inn_digits_are_split_in_inn_context(self) -> None:
        result = run_asr_normalization(
            "инн пять ноль ноль один ноль ноль семь три два два пять девять "
            "пять ноль ноль один ноль ноль семь три два два пять девять",
            normalize_document_numbers=True,
        )

        self.assertEqual(result.normalized_text, "инн 500100732259, 500100732259")
        self.assertEqual(len(result.document_number_spans), 1)
        self.assertEqual(result.document_number_spans[0].pii_type, "INN")
        self.assertEqual(result.document_number_spans[0].reason, "split_repeated_document_number")

    def test_repeated_passport_number_component_is_split(self) -> None:
        result = run_numeric_pii_pipeline(
            "паспорт серия четыре пять один два номер шесть семь восемь девять один ноль "
            "шесть семь восемь девять один ноль",
            normalize_document_numbers=True,
        )

        self.assertEqual(
            result.preprocessing_result.normalized_text,
            "паспорт серия 4512 номер 678910, 678910",
        )
        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in result.matches],
            [("PASSPORT", "4512678910")],
        )

    def test_short_echo_tail_after_driver_license_is_trimmed(self) -> None:
        pipeline = ASRNormalizationPipeline(
            document_number_config=DocumentNumberNormalizerConfig(trim_echo_tail=True),
            normalize_document_numbers=True,
        )
        normalized_text: str = pipeline.run(
            "водительское удостоверение девять девять ноль восемь один два три четыре пять шесть "
            "пять шесть"
        ).normalized_text
        result = run_numeric_pii_pipeline(
            normalized_text,
            normalize_document_numbers=False,
        )

        self.assertEqual(normalized_text, "водительское удостоверение 9908123456")
        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in result.matches],
            [("DRIVER_LICENSE", "9908123456")],
        )
        self.assertEqual(
            pipeline.run(
                "водительское удостоверение девять девять ноль восемь один два три четыре пять шесть "
                "пять шесть"
            ).document_number_spans[0].reason,
            "echo_tail_document_number",
        )

    def test_repeated_digits_without_document_context_are_left_unchanged(self) -> None:
        result = run_asr_normalization("давление сто двадцать сто двадцать")

        self.assertEqual(result.normalized_text, "давление 120120")
        self.assertEqual(result.document_number_spans, ())

    def test_document_number_normalization_can_be_disabled(self) -> None:
        result = run_asr_normalization(
            "инн пять ноль ноль один ноль ноль семь три два два пять девять "
            "пять ноль ноль один ноль ноль семь три два два пять девять",
            normalize_document_numbers=False,
        )

        self.assertEqual(result.normalized_text, "инн 500100732259500100732259")
        self.assertEqual(result.document_number_spans, ())

    def test_standalone_normalizer_returns_audit_span(self) -> None:
        normalizer = DocumentNumberNormalizer()
        result = normalizer.normalize("снилс 1234567890012345678900")

        self.assertEqual(result.text, "снилс 12345678900, 12345678900")
        self.assertEqual(result.spans[0].raw, "1234567890012345678900")
        self.assertEqual(result.spans[0].normalized, "12345678900, 12345678900")

    def test_pipeline_json_contains_document_number_layer(self) -> None:
        pipeline = ASRNormalizationPipeline()
        payload: dict[str, object] = json.loads(
            pipeline.to_json(
                "инн пять ноль ноль один ноль ноль семь три два два пять девять "
                "пять ноль ноль один ноль ноль семь три два два пять девять",
                ensure_ascii=False,
            )
        )

        self.assertEqual(
            payload["document_number_normalized_text"],
            "инн 500100732259500100732259",
        )
        self.assertEqual(payload["document_number_spans"], [])

    def test_pipeline_json_contains_enabled_document_number_layer(self) -> None:
        pipeline = ASRNormalizationPipeline(normalize_document_numbers=True)
        payload: dict[str, object] = json.loads(
            pipeline.to_json(
                "инн пять ноль ноль один ноль ноль семь три два два пять девять "
                "пять ноль ноль один ноль ноль семь три два два пять девять",
                ensure_ascii=False,
            )
        )

        self.assertEqual(
            payload["document_number_normalized_text"],
            "инн 500100732259, 500100732259",
        )
        self.assertIn("document_number_spans", payload)

    def test_cli_can_enable_document_number_normalization(self) -> None:
        repository_root: Path = Path(__file__).resolve().parents[1]
        command: list[str] = [
            sys.executable,
            "-m",
            "anonmed.cli",
            "инн пять ноль ноль один ноль ноль семь три два два пять девять "
            "пять ноль ноль один ноль ноль семь три два два пять девять",
            "--run",
            "--normalize-document-numbers",
        ]
        completed_process: subprocess.CompletedProcess[str] = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed_process.returncode, 0, completed_process.stderr)
        self.assertEqual(completed_process.stdout.strip(), "инн 500100732259, 500100732259")


__all__: list[str] = []
