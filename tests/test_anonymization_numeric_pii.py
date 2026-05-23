from __future__ import annotations

import unittest

from anonmed.anonymization import run_numeric_pii_pipeline


class NumericPIICompositionTests(unittest.TestCase):
    def test_numeric_pii_pipeline_runs_end_to_end_for_phone(self) -> None:
        pipeline_result = run_numeric_pii_pipeline(
            "телефон восемь девять один три один два три четыре пять шесть семь"
        )
        self.assertEqual(pipeline_result.preprocessing_result.normalized_text, "телефон 89131234567")
        self.assertEqual(len(pipeline_result.matches), 1)
        self.assertEqual(pipeline_result.matches[0].pii_type, "PHONE")
        self.assertEqual(pipeline_result.matches[0].normalized_value, "+79131234567")
        self.assertEqual(pipeline_result.masked_text, "телефон [PHONE]")

    def test_numeric_pii_pipeline_can_deduplicate_repeated_lines_before_matching(self) -> None:
        pipeline_result = run_numeric_pii_pipeline(
            (
                "телефон восемь девять один три один два три четыре пять шесть семь\n"
                "телефон восемь девять один три один два три четыре пять шесть семь"
            ),
            deduplicate_repetitions=True,
        )

        self.assertEqual(
            pipeline_result.preprocessing_result.normalized_text,
            "телефон 89131234567",
        )
        self.assertEqual(
            pipeline_result.preprocessing_result.repetition_suppressed_indexes,
            (1,),
        )
        self.assertEqual(len(pipeline_result.matches), 1)

    def test_numeric_pii_pipeline_runs_end_to_end_for_mse(self) -> None:
        pipeline_result = run_numeric_pii_pipeline(
            "справка мсэ номер ноль восемь семь четыре два три дробь две тысячи двадцать один"
        )
        self.assertEqual(
            pipeline_result.preprocessing_result.normalized_text,
            "справка мсэ номер 087423 дробь 2021",
        )
        self.assertEqual([match.normalized_value for match in pipeline_result.matches], ["0874232021"])
        self.assertEqual(pipeline_result.masked_text, "справка мсэ номер [MSE]")

    def test_numeric_pii_pipeline_concatenates_document_number_chunks(self) -> None:
        pipeline_result = run_numeric_pii_pipeline(
            "паспорт серия сорок пять одиннадцать семьсот восемьдесят девять триста двадцать четыре "
            "инн свой помните да да семьсот семьдесят два девятьсот восемнадцать "
            "четыреста пятьдесят шесть тридцать два"
        )
        self.assertEqual(
            pipeline_result.preprocessing_result.normalized_text,
            "паспорт серия 4511789324 инн свой помните да да 77291845632",
        )
        self.assertEqual([match.pii_type for match in pipeline_result.matches], ["PASSPORT"])
        self.assertEqual([match.normalized_value for match in pipeline_result.matches], ["4511789324"])
        self.assertEqual(
            pipeline_result.masked_text,
            "паспорт серия [PASSPORT] инн свой помните да да 77291845632",
        )


__all__: list[str] = []
