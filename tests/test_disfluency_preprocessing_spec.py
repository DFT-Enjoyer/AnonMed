from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import unittest

from asr_integer_extractor import DisfluencyFilter, DisfluencyFilterConfig, remove_disfluencies


@dataclass(frozen=True, slots=True)
class PreprocessingCase:
    name: str
    text: str
    expected: str
    config: DisfluencyFilterConfig | None = None


_DEFAULT_CASES: tuple[PreprocessingCase, ...] = (
    # Canonical hesitation tokens are always removable because they do not carry lexical content.
    PreprocessingCase("hesitation_em_leading", "эм номер один", "номер один"),
    PreprocessingCase("hesitation_em_inner", "номер эм один", "номер один"),
    PreprocessingCase("hesitation_em_trailing", "номер один эм", "номер один"),
    PreprocessingCase("hesitation_em_punctuated_leading", "эм, номер один", "номер один"),
    PreprocessingCase("hesitation_em_punctuated_inner", "номер, эм, один", "номер один"),
    PreprocessingCase("hesitation_em_punctuated_trailing", "номер один, эм", "номер один"),
    PreprocessingCase("hesitation_emm_leading", "эмм номер один", "номер один"),
    PreprocessingCase("hesitation_emmm_leading", "эммм номер один", "номер один"),
    PreprocessingCase("hesitation_e_leading", "э номер один", "номер один"),
    PreprocessingCase("hesitation_ee_leading", "ээ номер один", "номер один"),
    PreprocessingCase("hesitation_eee_leading", "эээ номер один", "номер один"),
    PreprocessingCase("hesitation_eeee_leading", "ээээ номер один", "номер один"),
    PreprocessingCase("hesitation_mm_leading", "мм номер один", "номер один"),
    PreprocessingCase("hesitation_mmm_leading", "ммм номер один", "номер один"),
    PreprocessingCase("hesitation_m_leading", "м номер один", "номер один"),
    PreprocessingCase("hesitation_hm_leading", "хм номер один", "номер один"),
    PreprocessingCase("hesitation_uppercase", "ЭМ номер один", "номер один"),
    PreprocessingCase("hesitation_titlecase", "Эм номер один", "номер один"),
    PreprocessingCase("hesitation_multiple_tokens", "эм эээ мм номер один", "номер один"),
    PreprocessingCase("hesitation_between_number_words", "один эм два", "один два"),
    PreprocessingCase("hesitation_before_digits", "эм 123", "123"),
    PreprocessingCase("hesitation_after_digits", "123 эм", "123"),
    PreprocessingCase("hesitation_keeps_final_period", "номер один эм.", "номер один."),
    PreprocessingCase("hesitation_keeps_final_question", "эм номер один?", "номер один?"),
    PreprocessingCase("hesitation_keeps_final_exclamation", "эм номер один!", "номер один!"),
    # Hyphenated hesitation variants should be treated as one removable acoustic artifact.
    PreprocessingCase("hyphenated_e_e_plain", "э-э номер один", "номер один"),
    PreprocessingCase("hyphenated_e_e_spaced", "э - э номер один", "номер один"),
    PreprocessingCase("hyphenated_e_e_en_dash", "э–э номер один", "номер один"),
    PreprocessingCase("hyphenated_e_e_em_dash", "э—э номер один", "номер один"),
    PreprocessingCase("hyphenated_e_e_minus", "э−э номер один", "номер один"),
    PreprocessingCase("hyphenated_m_m_plain", "м-м номер один", "номер один"),
    PreprocessingCase("hyphenated_m_m_spaced", "м - м номер один", "номер один"),
    PreprocessingCase("hyphenated_e_m_plain", "э-м номер один", "номер один"),
    PreprocessingCase("hyphenated_m_e_plain", "м-э номер один", "номер один"),
    PreprocessingCase("hyphenated_inner", "номер э-э один", "номер один"),
    PreprocessingCase("hyphenated_trailing", "номер один э-э", "номер один"),
    PreprocessingCase("hyphenated_punctuated_inner", "номер, э-э, один", "номер один"),
    PreprocessingCase("hyphenated_only_period", "э-э.", ""),
    PreprocessingCase("hyphenated_after_discourse", "ну э-э номер один", "номер один"),
    PreprocessingCase("hyphenated_before_phrase_filler", "э-э как бы номер один", "номер один"),
    # Multi-token fillers are removable regardless of their local syntactic position.
    PreprocessingCase("phrase_kak_by_leading", "как бы номер один", "номер один"),
    PreprocessingCase("phrase_kak_by_inner", "номер как бы один", "номер один"),
    PreprocessingCase("phrase_kak_by_trailing", "номер один как бы", "номер один"),
    PreprocessingCase("phrase_kak_by_punctuated", "номер, как бы, один", "номер один"),
    PreprocessingCase("phrase_kak_by_uppercase", "КАК БЫ номер один", "номер один"),
    PreprocessingCase("phrase_eto_samoe_leading", "это самое номер один", "номер один"),
    PreprocessingCase("phrase_eto_samoe_inner", "номер это самое один", "номер один"),
    PreprocessingCase("phrase_eto_samoe_trailing", "номер один это самое", "номер один"),
    PreprocessingCase("phrase_eto_samoe_punctuated", "номер, это самое, один", "номер один"),
    PreprocessingCase("phrase_tak_skazat_leading", "так сказать номер один", "номер один"),
    PreprocessingCase("phrase_tak_skazat_inner", "номер так сказать один", "номер один"),
    PreprocessingCase("phrase_tak_skazat_trailing", "номер один так сказать", "номер один"),
    PreprocessingCase("phrase_tak_skazat_punctuated", "номер, так сказать, один", "номер один"),
    PreprocessingCase("phrase_v_obshchem_leading", "в общем номер один", "номер один"),
    PreprocessingCase("phrase_v_obshchem_inner", "номер в общем один", "номер один"),
    PreprocessingCase("phrase_v_obshchem_trailing", "номер один в общем", "номер один"),
    PreprocessingCase("phrase_v_obshchem_punctuated", "номер, в общем, один", "номер один"),
    PreprocessingCase("phrase_multiple_same", "как бы номер как бы один", "номер один"),
    PreprocessingCase("phrase_multiple_different", "как бы номер это самое один", "номер один"),
    PreprocessingCase("phrase_with_hesitation_between", "как бы эм номер один", "номер один"),
    # Conservative discourse markers: remove at utterance or pause boundaries, not inside content.
    PreprocessingCase("marker_nu_leading", "ну номер один", "номер один"),
    PreprocessingCase("marker_nu_leading_comma", "ну, номер один", "номер один"),
    PreprocessingCase("marker_nu_inner_comma", "номер, ну, один", "номер один"),
    PreprocessingCase("marker_nu_after_period", "текст. ну номер один", "текст. номер один"),
    PreprocessingCase("marker_nu_trailing_comma", "номер один, ну", "номер один"),
    PreprocessingCase("marker_nu_content_position_kept", "номер ну один", "номер ну один"),
    PreprocessingCase("marker_a_leading", "а номер один", "номер один"),
    PreprocessingCase("marker_a_leading_comma", "а, номер один", "номер один"),
    PreprocessingCase("marker_a_inner_comma", "номер, а, один", "номер один"),
    PreprocessingCase("marker_a_content_conjunction_kept", "номер а один", "номер а один"),
    PreprocessingCase("marker_vot_leading", "вот номер один", "номер один"),
    PreprocessingCase("marker_vot_leading_comma", "вот, номер один", "номер один"),
    PreprocessingCase("marker_vot_inner_comma", "номер, вот, один", "номер один"),
    PreprocessingCase("marker_vot_content_position_kept", "номер вот один", "номер вот один"),
    PreprocessingCase("marker_znachit_leading", "значит номер один", "номер один"),
    PreprocessingCase("marker_znachit_leading_comma", "значит, номер один", "номер один"),
    PreprocessingCase("marker_znachit_inner_comma", "номер, значит, один", "номер один"),
    PreprocessingCase("marker_znachit_content_position_kept", "номер значит один", "номер значит один"),
    PreprocessingCase("marker_tipa_leading", "типа номер один", "номер один"),
    PreprocessingCase("marker_tipa_leading_comma", "типа, номер один", "номер один"),
    PreprocessingCase("marker_tipa_inner_comma", "номер, типа, один", "номер один"),
    PreprocessingCase("marker_tipa_content_position_kept", "номер типа один", "номер типа один"),
    PreprocessingCase("marker_chain_leading", "ну а вот номер один", "номер один"),
    PreprocessingCase("marker_chain_punctuated", "ну, а, вот, номер один", "номер один"),
    PreprocessingCase("marker_chain_mixed", "ну, эм, вот, номер один", "номер один"),
    # Interjections are removed only when isolated by a boundary; embedded lexical uses are preserved.
    PreprocessingCase("interjection_oy_leading", "ой, номер один", "номер один"),
    PreprocessingCase("interjection_oy_trailing", "номер один, ой", "номер один"),
    PreprocessingCase("interjection_oy_isolated_inner", "номер, ой, один", "номер один"),
    PreprocessingCase("interjection_oy_content_position_kept", "номер ой один", "номер ой один"),
    PreprocessingCase("interjection_ah_leading", "ах, номер один", "номер один"),
    PreprocessingCase("interjection_ah_trailing", "номер один, ах", "номер один"),
    PreprocessingCase("interjection_ah_content_position_kept", "номер ах один", "номер ах один"),
    PreprocessingCase("interjection_oh_leading", "ох, номер один", "номер один"),
    PreprocessingCase("interjection_oh_trailing", "номер один, ох", "номер один"),
    PreprocessingCase("interjection_oh_content_position_kept", "номер ох один", "номер ох один"),
    PreprocessingCase("interjection_uh_leading", "ух, номер один", "номер один"),
    PreprocessingCase("interjection_uh_trailing", "номер один, ух", "номер один"),
    PreprocessingCase("interjection_uh_content_position_kept", "номер ух один", "номер ух один"),
    PreprocessingCase("interjection_eh_leading", "эх, номер один", "номер один"),
    PreprocessingCase("interjection_eh_trailing", "номер один, эх", "номер один"),
    PreprocessingCase("interjection_eh_content_position_kept", "номер эх один", "номер эх один"),
    PreprocessingCase("interjection_uppercase", "ОЙ, номер один", "номер один"),
    PreprocessingCase("interjection_with_hesitation", "ой, эм, номер один", "номер один"),
    PreprocessingCase("interjection_before_phrase", "ой, как бы номер один", "номер один"),
    PreprocessingCase("interjection_between_sentences", "первый. ой, второй", "первый. второй"),
    # Lexical false positives must not be damaged by substring matching.
    PreprocessingCase("false_positive_echo", "эхо номер один", "эхо номер один"),
    PreprocessingCase("false_positive_type_word", "типаж номер один", "типаж номер один"),
    PreprocessingCase("false_positive_votchina", "вотчина номер один", "вотчина номер один"),
    PreprocessingCase("false_positive_nuka", "нука номер один", "нука номер один"),
    PreprocessingCase("false_positive_common", "общение номер один", "общение номер один"),
    PreprocessingCase("false_positive_takoy", "такой сказать нельзя", "такой сказать нельзя"),
    PreprocessingCase("false_positive_kakoy", "какой бы номер", "какой бы номер"),
    PreprocessingCase("false_positive_samoe_without_eto", "самое важное число", "самое важное число"),
    PreprocessingCase("false_positive_eto_without_samoe", "это важное число", "это важное число"),
    PreprocessingCase("false_positive_a_inside_word", "адрес номер один", "адрес номер один"),
    PreprocessingCase("false_positive_m_inside_word", "метка номер один", "метка номер один"),
    PreprocessingCase("false_positive_hm_inside_word", "хмурый день", "хмурый день"),
    # Space and punctuation normalization after deletion.
    PreprocessingCase("spacing_multiple_spaces", "эм   номер    один", "номер один"),
    PreprocessingCase("spacing_tabs", "эм\tномер\tодин", "номер один"),
    PreprocessingCase("spacing_newline", "эм\nномер\nодин", "номер один"),
    PreprocessingCase("punct_leading_comma_removed", ", эм, номер один", "номер один"),
    PreprocessingCase("punct_leading_semicolon_removed", "; эм; номер один", "номер один"),
    PreprocessingCase("punct_leading_colon_removed", ": эм: номер один", "номер один"),
    PreprocessingCase("punct_trailing_comma_removed", "номер один, эм,", "номер один"),
    PreprocessingCase("punct_trailing_semicolon_removed", "номер один; эм;", "номер один"),
    PreprocessingCase("punct_trailing_colon_removed", "номер один: эм:", "номер один"),
    PreprocessingCase("punct_repeated_soft_boundaries", "номер, эм, , один", "номер один"),
    PreprocessingCase("punct_soft_before_period", "номер, эм.", "номер."),
    PreprocessingCase("punct_soft_before_question", "номер, эм?", "номер?"),
    PreprocessingCase("punct_soft_before_exclamation", "номер, эм!", "номер!"),
    PreprocessingCase("punct_sentence_boundary_preserved", "эм. номер один", "номер один"),
    PreprocessingCase("punct_two_sentences", "первый. эм, второй", "первый. второй"),
)

_CONFIG_CASES: tuple[PreprocessingCase, ...] = (
    PreprocessingCase(
        "config_keep_hesitations",
        "эм номер один",
        "эм номер один",
        DisfluencyFilterConfig(remove_hesitations=False),
    ),
    PreprocessingCase(
        "config_keep_hyphenated_hesitations",
        "э-э номер один",
        "э-э номер один",
        DisfluencyFilterConfig(remove_hesitations=False),
    ),
    PreprocessingCase(
        "config_keep_phrase_fillers",
        "как бы номер один",
        "как бы номер один",
        DisfluencyFilterConfig(remove_phrase_fillers=False),
    ),
    PreprocessingCase(
        "config_keep_interjections",
        "ой, номер один",
        "ой, номер один",
        DisfluencyFilterConfig(remove_interjections=False),
    ),
    PreprocessingCase(
        "config_keep_discourse_markers",
        "ну, номер один",
        "ну, номер один",
        DisfluencyFilterConfig(remove_discourse_markers=False),
    ),
    PreprocessingCase(
        "config_aggressive_discourse_nu",
        "номер ну один",
        "номер один",
        DisfluencyFilterConfig(aggressive_discourse_markers=True),
    ),
    PreprocessingCase(
        "config_aggressive_discourse_a",
        "номер а один",
        "номер один",
        DisfluencyFilterConfig(aggressive_discourse_markers=True),
    ),
    PreprocessingCase(
        "config_aggressive_discourse_vot",
        "номер вот один",
        "номер один",
        DisfluencyFilterConfig(aggressive_discourse_markers=True),
    ),
    PreprocessingCase(
        "config_aggressive_discourse_znachit",
        "номер значит один",
        "номер один",
        DisfluencyFilterConfig(aggressive_discourse_markers=True),
    ),
    PreprocessingCase(
        "config_aggressive_discourse_tipa",
        "номер типа один",
        "номер один",
        DisfluencyFilterConfig(aggressive_discourse_markers=True),
    ),
    PreprocessingCase(
        "config_only_hesitations_disabled_still_removes_phrase",
        "эм как бы номер один",
        "эм номер один",
        DisfluencyFilterConfig(remove_hesitations=False),
    ),
    PreprocessingCase(
        "config_only_phrases_disabled_still_removes_hesitation",
        "эм как бы номер один",
        "как бы номер один",
        DisfluencyFilterConfig(remove_phrase_fillers=False),
    ),
    PreprocessingCase(
        "config_all_removal_disabled",
        "ну, эм, как бы ой, номер один",
        "ну, эм, как бы ой, номер один",
        DisfluencyFilterConfig(
            remove_hesitations=False,
            remove_interjections=False,
            remove_phrase_fillers=False,
            remove_discourse_markers=False,
        ),
    ),
)

_PREPROCESSING_CASES: tuple[PreprocessingCase, ...] = _DEFAULT_CASES + _CONFIG_CASES


class DisfluencyPreprocessingSpecificationTests(unittest.TestCase):
    def test_preprocessing_spec_contains_at_least_100_cases(self) -> None:
        case_count: int = len(_PREPROCESSING_CASES)
        self.assertGreaterEqual(case_count, 100)


class DisfluencyPreprocessingMetadataSpecificationTests(unittest.TestCase):
    def test_removed_spans_are_reported_with_original_text_boundaries(self) -> None:
        disfluency_filter = DisfluencyFilter()
        cleaned = disfluency_filter.clean("ну, эм, номер один")
        removed_raw_values: tuple[str, ...] = tuple(span.raw for span in cleaned.removed_spans)
        self.assertIn("ну", removed_raw_values)
        self.assertIn("эм", removed_raw_values)
        for span in cleaned.removed_spans:
            recovered_raw: str = cleaned.original_text[span.start : span.end]
            self.assertEqual(recovered_raw, span.raw)

    def test_removed_span_reasons_are_part_of_the_public_audit_trail(self) -> None:
        disfluency_filter = DisfluencyFilter()
        cleaned = disfluency_filter.clean("ну, эм, как бы ой, номер один")
        reasons: set[str] = {span.reason for span in cleaned.removed_spans}
        self.assertIn("discourse_marker", reasons)
        self.assertIn("hesitation", reasons)
        self.assertIn("phrase_filler", reasons)
        self.assertIn("interjection", reasons)


def _make_preprocessing_case_test(case: PreprocessingCase) -> Callable[[unittest.TestCase], None]:
    def test_case(self: unittest.TestCase) -> None:
        cleaned_text: str = remove_disfluencies(case.text, config=case.config)
        self.assertEqual(cleaned_text, case.expected)

    test_case.__name__ = f"test_preprocessing_case_{case.name}"
    test_case.__doc__ = f"{case.text!r} -> {case.expected!r}"
    return test_case


for _case_index, _case in enumerate(_PREPROCESSING_CASES, start=1):
    _test_name: str = f"test_preprocessing_case_{_case_index:03d}_{_case.name}"
    setattr(
        DisfluencyPreprocessingSpecificationTests,
        _test_name,
        _make_preprocessing_case_test(_case),
    )


__all__: list[str] = []
