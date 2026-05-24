from __future__ import annotations

import unittest
from typing import Final

from anonmed.anonymization import NumericPIIMatch, find_numeric_pii, mask_numeric_pii


__all__: Final[tuple[str, ...]] = ("NumericPIIRulesTests",)


class NumericPIIRulesTests(unittest.TestCase):
    def assert_single_match(
        self,
        text: str,
        expected_type: str,
        expected_normalized_value: str,
    ) -> NumericPIIMatch:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(text)
        self.assertEqual(len(matches), 1, matches)
        match: NumericPIIMatch = matches[0]
        self.assertEqual(match.pii_type, expected_type)
        self.assertEqual(match.normalized_value, expected_normalized_value)
        return match

    def test_phone_with_eleven_spaced_digits(self) -> None:
        self.assert_single_match("мой телефон 8 913 123 45 67", "PHONE", "+79131234567")

    def test_phone_with_ten_digits_after_preprocessing(self) -> None:
        self.assert_single_match("для связи 913 123 45 67", "PHONE", "+79131234567")

    def test_phone_with_compact_digits(self) -> None:
        self.assert_single_match("телефон 89131234567", "PHONE", "+79131234567")

    def test_phone_with_plus_symbol(self) -> None:
        self.assert_single_match("номер телефона +7 913 123 45 67", "PHONE", "+79131234567")

    def test_contextual_city_phone_with_country_code(self) -> None:
        self.assert_single_match(
            "контактный телефон +7 495 123 45 67",
            "PHONE",
            "+74951234567",
        )

    def test_contextual_city_phone_with_trunk_code(self) -> None:
        self.assert_single_match(
            "городской 8 (495) 123-45-67",
            "PHONE",
            "+74951234567",
        )

    def test_contextual_ten_digit_city_phone(self) -> None:
        self.assert_single_match("тел. 495 123 45 67", "PHONE", "+74951234567")

    def test_phone_context_can_follow_number(self) -> None:
        self.assert_single_match("495 123 45 67 контактный", "PHONE", "+74951234567")

    def test_messenger_context_matches_phone(self) -> None:
        self.assert_single_match("ватсап +7 (999) 123-45-67", "PHONE", "+79991234567")

    def test_landline_phone_with_context(self) -> None:
        self.assert_single_match("домашний телефон 123 456", "PHONE", "123456")

    def test_landline_phone_without_context_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("анализ 123 456")
        self.assertEqual(matches, ())

    def test_city_phone_without_context_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("анализ 495 123 45 67")
        self.assertEqual(matches, ())

    def test_phone_is_rejected_in_policy_context(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("полис 89131234567")
        self.assertEqual(matches, ())

    def test_city_phone_is_rejected_in_policy_context(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("полис 8 495 123 45 67")
        self.assertEqual(matches, ())

    def test_snils_with_context(self) -> None:
        self.assert_single_match("снилс 123 456 789 00", "SNILS", "12345678900")

    def test_snils_without_context_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("показатель 123 456 789 00")
        self.assertEqual(matches, ())

    def test_snils_wins_over_phone_like_digits(self) -> None:
        self.assert_single_match("снилс 891 312 345 67", "SNILS", "89131234567")

    def test_inn_with_context(self) -> None:
        self.assert_single_match("инн 500100732259", "INN", "500100732259")

    def test_inn_without_context_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("номер анализа 500100732259")
        self.assertEqual(matches, ())

    def test_oms_sixteen_digits(self) -> None:
        self.assert_single_match("полис омс 1234 5678 9012 3456", "OMS", "1234567890123456")

    def test_oms_short_contextual_digits_are_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("номер полиса 1234567890")
        self.assertEqual(matches, ())

    def test_passport_series_and_number_with_inner_number_word(self) -> None:
        self.assert_single_match("паспорт серия 12 34 номер 567890", "PASSPORT", "1234567890")

    def test_passport_compact_with_context(self) -> None:
        self.assert_single_match("паспорт 1234567890", "PASSPORT", "1234567890")

    def test_passport_without_context_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("анализ 1234567890")
        self.assertEqual(matches, ())

    def test_driver_license_with_context(self) -> None:
        self.assert_single_match("водительское удостоверение 99 99 123456", "DRIVER_LICENSE", "9999123456")

    def test_birth_date_separated(self) -> None:
        self.assert_single_match("дата рождения 12 03 1984", "DATE_BIRTH", "12.03.1984")

    def test_birth_date_single_digit_day_month(self) -> None:
        self.assert_single_match("родился 5 5 1985", "DATE_BIRTH", "05.05.1985")

    def test_birth_date_compact(self) -> None:
        self.assert_single_match("дата рождения 12031984", "DATE_BIRTH", "12.03.1984")

    def test_invalid_birth_date_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("дата рождения 31 02 1984")
        self.assertEqual(matches, ())

    def test_appointment_date_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("дата приема 12 03 2024")
        self.assertEqual(matches, ())

    def test_age_before_context(self) -> None:
        self.assert_single_match("возраст 41", "AGE", "41")

    def test_age_after_context(self) -> None:
        self.assert_single_match("пациенту 41 год", "AGE", "41")

    def test_age_phrase_from_asr(self) -> None:
        self.assert_single_match("вам сейчас 41 верно", "AGE", "41")

    def test_temperature_is_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("температура 38 и 5")
        self.assertEqual(matches, ())

    def test_pressure_is_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("давление 120 на 80")
        self.assertEqual(matches, ())

    def test_days_are_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("кашель 12 дней")
        self.assertEqual(matches, ())

    def test_past_year_count_is_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("в зал ходил 2 года назад")
        self.assertEqual(matches, ())

    def test_family_history_age_is_not_patient_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
            "у отца была аритмия лет в 60 началось"
        )
        self.assertEqual(matches, ())

    def test_pain_scale_numbers_are_not_age_near_patient_age_context(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
            "шкала боли сейчас 7 из 10 запишу 41 год"
        )
        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in matches],
            [("AGE", "41")],
        )

    def test_clock_time_is_not_age_near_age_context(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
            "пациенту 41 год повторный прием в 10 утра"
        )
        self.assertEqual(
            [(match.pii_type, match.normalized_value) for match in matches],
            [("AGE", "41")],
        )

    def test_duration_years_are_not_age(self) -> None:
        samples: tuple[str, ...] = (
            "стаж работы 7 лет",
            "курил лет 20",
            "жалобы появились лет 5 назад",
            "эрозию прижигали еще лет в 25",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(sample)
                self.assertEqual(matches, ())

    def test_childhood_year_marker_is_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
            "на амоксициллин была сыпь в 3 года"
        )
        self.assertEqual(matches, ())

    def test_date_month_parts_are_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
            "дата рождения 12 марта 1985 года"
        )
        self.assertEqual(matches, ())

    def test_document_fraction_parts_are_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
            "справка номер 12 дробь 693"
        )
        self.assertEqual(matches, ())

    def test_clinical_numeric_units_are_not_age(self) -> None:
        samples: tuple[str, ...] = (
            "диабет 2 типа",
            "гонартроз 3 стадии",
            "фракция выброса 38 процентов",
            "эутирокс 50 микрограмм",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(sample)
                self.assertEqual(matches, ())

    def test_periodic_count_is_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(
            "в целом здорова 1 в год проверяюсь"
        )
        self.assertEqual(matches, ())

    def test_age_with_patient_context_still_matches_years_unit(self) -> None:
        self.assert_single_match("мне 46 лет", "AGE", "46")
        self.assert_single_match("сколько вам полных лет 38", "AGE", "38")
        self.assert_single_match("сколько лет кириллу 6 лет", "AGE", "6")
        self.assert_single_match("ему 4 годика", "AGE", "4")
        self.assert_single_match("представьтесь соколов андрей николаевич 58 лет", "AGE", "58")

    def test_room_number_is_not_age(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("идите в 20 кабинет")
        self.assertEqual(matches, ())

    def test_mse_number(self) -> None:
        self.assert_single_match("справка мсэ номер 0126 0001234", "MSE", "01260001234")

    def test_mse_number_with_inner_number_word(self) -> None:
        self.assert_single_match("справка мсэ серия 0126 номер 0001234", "MSE", "01260001234")

    def test_short_mse_number_is_rejected(self) -> None:
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii("справка мсэ номер 087423 2021")
        self.assertEqual(matches, ())

    def test_birth_certificate_number(self) -> None:
        self.assert_single_match("свидетельство о рождении номер 123456", "BIRTH_CERTIFICATE", "123456")

    def test_multiple_matches_order(self) -> None:
        text: str = "дата рождения 12 03 1984 телефон 8 913 123 45 67 снилс 123 456 789 00"
        matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(text)
        match_types: tuple[str, ...] = tuple(match.pii_type for match in matches)
        self.assertEqual(match_types, ("DATE_BIRTH", "PHONE", "SNILS"))

    def test_mask_numeric_pii(self) -> None:
        text: str = "дата рождения 12 03 1984 телефон 8 913 123 45 67"
        masked_text: str = mask_numeric_pii(text)
        self.assertEqual(masked_text, "дата рождения [DATE_BIRTH] телефон [PHONE]")

    def test_custom_mask_numeric_pii(self) -> None:
        text: str = "инн 500100732259"
        masked_text: str = mask_numeric_pii(text, {"INN": "[ENC_INN]"})
        self.assertEqual(masked_text, "инн [ENC_INN]")

    def test_punctuation_separators_are_allowed(self) -> None:
        self.assert_single_match("телефон 8-913-123-45-67", "PHONE", "+79131234567")

    def test_date_with_dot_separators_is_allowed(self) -> None:
        self.assert_single_match("дата рождения 12.03.1984", "DATE_BIRTH", "12.03.1984")

    def test_passport_not_split_by_spaces(self) -> None:
        match: NumericPIIMatch = self.assert_single_match("паспорт 1 2 3 4 5 6 7 8 9 0", "PASSPORT", "1234567890")
        self.assertEqual(match.value, "1 2 3 4 5 6 7 8 9 0")


if __name__ == "__main__":
    unittest.main()
