from __future__ import annotations

import unittest

from anonmed.anonymization import (
    PIICandidate,
    resolve_pii_candidates,
    run_numeric_pii_pipeline,
)


class NumericPIIPostProcessingTests(unittest.TestCase):
    def test_pipeline_groups_repeated_mentions_into_one_entity(self) -> None:
        pipeline_result = run_numeric_pii_pipeline(
            "телефон для связи восемь девять один семь четыре четыре четыре пять пять два два "
            "восемь девять один семь четыре четыре четыре пять пять два два",
            normalize_document_numbers=True,
        )

        self.assertEqual(len(pipeline_result.matches), 2)
        self.assertEqual(len(pipeline_result.post_processing_result.entity_groups), 1)
        entity_group = pipeline_result.post_processing_result.entity_groups[0]
        self.assertEqual(entity_group.entity_type, "PHONE")
        self.assertEqual(entity_group.mention_count, 2)
        self.assertEqual(
            {match.metadata["entity_id"] for match in pipeline_result.matches},
            {entity_group.entity_id},
        )
        self.assertEqual(pipeline_result.masked_normalized_text.count("[PHONE]"), 2)
        self.assertEqual(pipeline_result.masked_original_text.count("[PHONE]"), 1)

    def test_pipeline_supports_same_length_masking_on_original_layer(self) -> None:
        pipeline_result = run_numeric_pii_pipeline(
            "телефон 89131234567",
            masking_strategy="same_length",
        )

        self.assertEqual(pipeline_result.masked_original_text, "телефон ***********")
        self.assertEqual(pipeline_result.masked_normalized_text, "телефон ***********")

    def test_production_safe_candidate_resolution_unions_overlapping_direct_ids(self) -> None:
        first_candidate = PIICandidate(
            entity_type="PHONE",
            source="regex",
            source_score=0.80,
            start=10,
            end=21,
            value="89131234567",
            normalized_value="+79131234567",
            rule_id="phone",
            context="телефон 89131234567",
            metadata={"positive_context_hit": True},
            validators={"format_ok": True},
            negative_context_hits=(),
            sensitivity_rank=100,
        )
        second_candidate = PIICandidate(
            entity_type="SNILS",
            source="regex",
            source_score=0.82,
            start=15,
            end=26,
            value="23456789012",
            normalized_value="23456789012",
            rule_id="snils",
            context="снилс 23456789012",
            metadata={"positive_context_hit": True},
            validators={"format_ok": True},
            negative_context_hits=(),
            sensitivity_rank=99,
        )

        resolved_candidates = resolve_pii_candidates(
            (first_candidate, second_candidate),
            mode="production_safe",
        )

        self.assertEqual(len(resolved_candidates), 1)
        self.assertEqual((resolved_candidates[0].start, resolved_candidates[0].end), (10, 26))
        self.assertEqual(resolved_candidates[0].metadata["post_processing"], "union_overlap")


__all__: list[str] = []
