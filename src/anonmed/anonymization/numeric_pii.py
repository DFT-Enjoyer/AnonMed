from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal, Mapping, Pattern, Sequence

NumericPIIType = Literal[
    "PHONE",
    "SNILS",
    "PASSPORT",
    "DATE_BIRTH",
    "OMS",
    "INN",
    "AGE",
    "MSE",
    "BIRTH_CERTIFICATE",
    "DRIVER_LICENSE",
]

__all__: tuple[str, ...] = (
    "NumericPIIType",
    "NumericPIIMatch",
    "NumericPIIRule",
    "build_default_numeric_rules",
    "find_numeric_pii",
    "mask_numeric_pii",
    "normalize_numeric_pii_value",
)

_SEPARATOR_PATTERN: Final[str] = r"[\s\u00A0\-–—().]*"
_REQUIRED_SEPARATOR_PATTERN: Final[str] = r"[\s\u00A0\-–—()./]+"
_WORD_SEPARATOR_PATTERN: Final[str] = r"[\s\u00A0]+"
_REGEX_FLAGS: Final[int] = re.IGNORECASE | re.UNICODE

_COMMON_NEGATIVE_CONTEXT_PATTERNS: Final[tuple[str, ...]] = (
    r"\b(?:температур\w*|давлен\w*|пульс\w*|сатураци\w*|сахар\w*|глюкоз\w*|холестерин\w*)\b",
    r"\b(?:гемоглобин\w*|лейкоцит\w*|тромбоцит\w*|креатинин\w*|билирубин\w*)\b",
    r"\b(?:дозировк\w*|таблет\w*|капсул\w*|капл\w*|ампул\w*|миллиграм\w*|мг|мл|единиц\w*)\b",
    r"\b(?:дн(?:я|ей)|недел\w*|месяц\w*|час\w*|минут\w*|секунд\w*)\b",
    r"\b(?:кабинет\w*|палат\w*|этаж\w*|корпус\w*|дом\w*|квартир\w*|рост\w*|вес\w*)\b",
    r"\b(?:мкб|код\s+мкб|диагноз\w*|анализ\w*)\b",
)


@dataclass(frozen=True, slots=True)
class NumericPIIMatch:
    pii_type: NumericPIIType
    start: int
    end: int
    value: str
    normalized_value: str
    confidence: float
    rule_id: str
    context: str
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NumericPIIRule:
    pii_type: NumericPIIType
    rule_id: str
    pattern: Pattern[str]
    positive_context: tuple[Pattern[str], ...] = ()
    negative_context: tuple[Pattern[str], ...] = ()
    require_context: bool = False
    context_window: int = 72
    context_after_window: int = 16
    priority: int = 50
    base_confidence: float = 0.70


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, flags=_REGEX_FLAGS)


def _compile_many(patterns: Sequence[str]) -> tuple[Pattern[str], ...]:
    return tuple(_compile(pattern) for pattern in patterns)


def _digit_count_pattern(
    min_count: int,
    max_count: int | None = None,
    *,
    separator_pattern: str = _SEPARATOR_PATTERN,
) -> str:
    quantifier: str
    if max_count is None:
        quantifier = f"{{{min_count}}}"
    else:
        quantifier = f"{{{min_count},{max_count}}}"
    return rf"(?:\d{separator_pattern}){quantifier}"


def _bounded_value_pattern(value_pattern: str) -> str:
    return rf"(?<!\d)(?P<value>{value_pattern})(?!{_SEPARATOR_PATTERN}\d)"


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    trimmed_start: int = start
    trimmed_end: int = end
    while trimmed_start < trimmed_end and text[trimmed_start].isspace():
        trimmed_start += 1
    while trimmed_end > trimmed_start and text[trimmed_end - 1].isspace():
        trimmed_end -= 1
    return trimmed_start, trimmed_end


def _digits_only(value: str) -> str:
    return "".join(re.findall(r"\d", value))


def _has_context(patterns: Sequence[Pattern[str]], scope: str) -> bool:
    return any(pattern.search(scope) is not None for pattern in patterns)


def _snils_checksum_is_valid(digits: str) -> bool:
    if len(digits) != 11:
        return False
    first_digits: list[int] = [int(char) for char in digits[:9]]
    checksum_digits: int = int(digits[9:])
    weighted_sum: int = sum(value * weight for value, weight in zip(first_digits, range(9, 0, -1)))
    if weighted_sum < 100:
        expected_checksum: int = weighted_sum
    elif weighted_sum in (100, 101):
        expected_checksum = 0
    else:
        expected_checksum = weighted_sum % 101
        if expected_checksum == 100:
            expected_checksum = 0
    return expected_checksum == checksum_digits


def _inn12_checksum_is_valid(digits: str) -> bool:
    if len(digits) != 12:
        return False
    values: list[int] = [int(char) for char in digits]
    coeffs_11: tuple[int, ...] = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    coeffs_12: tuple[int, ...] = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    checksum_11: int = sum(value * coeff for value, coeff in zip(values[:10], coeffs_11)) % 11 % 10
    checksum_12: int = sum(value * coeff for value, coeff in zip(values[:11], coeffs_12)) % 11 % 10
    return checksum_11 == values[10] and checksum_12 == values[11]


def _normalize_phone(raw_value: str) -> str | None:
    digits: str = _digits_only(raw_value)
    if len(digits) == 10 and digits.startswith("9"):
        return "+7" + digits
    if len(digits) == 11 and digits[0] in {"7", "8"} and digits[1] == "9":
        return "+7" + digits[1:]
    return None


def _normalize_date_birth(raw_value: str) -> str | None:
    digit_groups: list[str] = re.findall(r"\d+", raw_value)
    digits: str = _digits_only(raw_value)
    if len(digit_groups) >= 3:
        day_value: int = int(digit_groups[0])
        month_value: int = int(digit_groups[1])
        year_value: int = int(digit_groups[2])
    elif len(digits) == 8:
        day_value = int(digits[:2])
        month_value = int(digits[2:4])
        year_value = int(digits[4:])
    else:
        return None

    current_year: int = date.today().year
    if year_value < 1900 or year_value > current_year:
        return None

    try:
        date(year_value, month_value, day_value)
    except ValueError:
        return None

    return f"{day_value:02d}.{month_value:02d}.{year_value:04d}"


def _normalize_age(raw_value: str) -> str | None:
    digits: str = _digits_only(raw_value)
    if not digits:
        return None
    age_value: int = int(digits)
    if age_value < 0 or age_value > 120:
        return None
    return str(age_value)


def _normalize_digits(
    raw_value: str,
    expected_min_length: int,
    expected_max_length: int | None = None,
) -> str | None:
    digits: str = _digits_only(raw_value)
    max_length: int = expected_min_length if expected_max_length is None else expected_max_length
    if expected_min_length <= len(digits) <= max_length:
        return digits
    return None


def _normalize_value(pii_type: NumericPIIType, raw_value: str) -> str | None:
    if pii_type == "PHONE":
        return _normalize_phone(raw_value)
    if pii_type == "SNILS":
        return _normalize_digits(raw_value, 11)
    if pii_type == "PASSPORT":
        return _normalize_digits(raw_value, 10)
    if pii_type == "DATE_BIRTH":
        return _normalize_date_birth(raw_value)
    if pii_type == "OMS":
        return _normalize_digits(raw_value, 10, 16)
    if pii_type == "INN":
        return _normalize_digits(raw_value, 12)
    if pii_type == "AGE":
        return _normalize_age(raw_value)
    if pii_type == "MSE":
        return _normalize_digits(raw_value, 4, 12)
    if pii_type == "BIRTH_CERTIFICATE":
        return _normalize_digits(raw_value, 4, 12)
    if pii_type == "DRIVER_LICENSE":
        return _normalize_digits(raw_value, 10)
    return None


def normalize_numeric_pii_value(pii_type: NumericPIIType, raw_value: str) -> str | None:
    return _normalize_value(pii_type, raw_value)


def _build_metadata(pii_type: NumericPIIType, normalized_value: str) -> dict[str, object]:
    metadata: dict[str, object] = {"length": len(_digits_only(normalized_value))}
    if pii_type == "SNILS":
        metadata["checksum_valid"] = _snils_checksum_is_valid(normalized_value)
    elif pii_type == "INN":
        metadata["checksum_valid"] = _inn12_checksum_is_valid(normalized_value)
    return metadata


def _confidence(rule: NumericPIIRule, positive_context_hit: bool, metadata: Mapping[str, object]) -> float:
    confidence_value: float = rule.base_confidence
    if positive_context_hit:
        confidence_value += 0.12
    checksum_value: object | None = metadata.get("checksum_valid")
    if checksum_value is True:
        confidence_value += 0.08
    elif checksum_value is False and rule.pii_type in {"SNILS", "INN"}:
        confidence_value -= 0.03
    return min(0.99, max(0.01, confidence_value))


def _create_default_numeric_rules() -> tuple[NumericPIIRule, ...]:
    common_negative_context: tuple[Pattern[str], ...] = _compile_many(_COMMON_NEGATIVE_CONTEXT_PATTERNS)
    phone_negative_context: tuple[Pattern[str], ...] = common_negative_context + _compile_many(
        (
            r"\b(?:паспорт\w*|серия\w*|снилс|полис\w*|омс|инн|мсэ)\b",
            r"\b(?:свидетельств\w*|водительск\w*|удостоверени\w*)\b",
        )
    )
    document_negative_context: tuple[Pattern[str], ...] = common_negative_context
    date_negative_context: tuple[Pattern[str], ...] = _compile_many(
        (
            r"\b(?:при[её]м\w*|запис\w*|операци\w*|анализ\w*|осмотр\w*|выписк\w*)\b",
            r"\b(?:сегодня|завтра|вчера|недел\w*|месяц\w*)\b",
        )
    )
    age_negative_context: tuple[Pattern[str], ...] = common_negative_context + _compile_many(
        (
            r"\b(?:дата\w*|рождени\w*|родил\w*|паспорт\w*|снилс|полис\w*|омс|инн)\b",
            r"\b(?:дом\w*|квартир\w*|улиц\w*|строени\w*|корпус\w*)\b",
        )
    )
    phone_value_pattern: str = (
        rf"(?:\+{_SEPARATOR_PATTERN})?"
        rf"(?:(?:7|8){_SEPARATOR_PATTERN})?"
        rf"9{_SEPARATOR_PATTERN}(?:\d{_SEPARATOR_PATTERN}){{9}}"
    )
    passport_value_pattern: str = (
        rf"{_digit_count_pattern(4)}(?:номер{_WORD_SEPARATOR_PATTERN})?{_digit_count_pattern(6)}"
    )
    date_value_pattern: str = (
        rf"(?:0?[1-9]|[12]\d|3[01]){_REQUIRED_SEPARATOR_PATTERN}"
        rf"(?:0?[1-9]|1[0-2]){_REQUIRED_SEPARATOR_PATTERN}"
        rf"(?:19|20)\d{{2}}"
        rf"|"
        rf"(?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])(?:19|20)\d{{2}}"
    )
    age_value_pattern: str = r"(?:[1-9]\d?|1[01]\d|120)"
    oms_value_pattern: str = rf"(?:{_digit_count_pattern(16)}|{_digit_count_pattern(10)})"
    mse_compound_value_pattern: str = (
        rf"(?:\d{_SEPARATOR_PATTERN}){{4,8}}(?:/|\s*дробь\s*)(?:\d{_SEPARATOR_PATTERN}){{2,4}}"
    )
    mse_value_pattern: str = rf"(?:{mse_compound_value_pattern}|{_digit_count_pattern(4, 12)})"

    return (
        NumericPIIRule(
            pii_type="DATE_BIRTH",
            rule_id="date_birth_with_context",
            pattern=_compile(_bounded_value_pattern(date_value_pattern)),
            positive_context=_compile_many(
                (
                    r"\b(?:дат[ауы]?\s+рожд\w*|день\s+рожд\w*|год\s+рожд\w*)\b",
                    r"\b(?:родил[аи]?с[ья]?|когда\s+родил\w*)\b",
                )
            ),
            negative_context=date_negative_context,
            require_context=True,
            context_window=88,
            context_after_window=0,
            priority=100,
            base_confidence=0.82,
        ),
        NumericPIIRule(
            pii_type="PASSPORT",
            rule_id="passport_series_number_with_context",
            pattern=_compile(_bounded_value_pattern(passport_value_pattern)),
            positive_context=_compile_many(
                (
                    r"\b(?:паспорт\w*|паспортн\w*|серия\w*|выдан\w*|код\s+подразделени\w*)\b",
                    r"\b(?:серия\w*\s+и\s+номер\w*|номер\w*\s+паспорт\w*)\b",
                )
            ),
            negative_context=document_negative_context,
            require_context=True,
            context_window=84,
            context_after_window=0,
            priority=98,
            base_confidence=0.82,
        ),
        NumericPIIRule(
            pii_type="DRIVER_LICENSE",
            rule_id="driver_license_number_with_context",
            pattern=_compile(_bounded_value_pattern(_digit_count_pattern(10))),
            positive_context=_compile_many(
                (
                    r"\b(?:водительск\w*\s+удостоверени\w*|водительск\w*\s+прав\w*)\b",
                    r"\b(?:права\w*\s+номер\w*|номер\w*\s+прав\w*)\b",
                )
            ),
            negative_context=document_negative_context,
            require_context=True,
            context_window=84,
            context_after_window=0,
            priority=97,
            base_confidence=0.82,
        ),
        NumericPIIRule(
            pii_type="OMS",
            rule_id="oms_policy_number_with_context",
            pattern=_compile(_bounded_value_pattern(oms_value_pattern)),
            positive_context=_compile_many(
                (
                    r"\b(?:омс|полис\w*)\b",
                    r"\b(?:обязательн\w*\s+медицинск\w*\s+страхован\w*)\b",
                )
            ),
            negative_context=document_negative_context,
            require_context=True,
            context_window=84,
            context_after_window=0,
            priority=96,
            base_confidence=0.80,
        ),
        NumericPIIRule(
            pii_type="SNILS",
            rule_id="snils_number_with_context",
            pattern=_compile(_bounded_value_pattern(_digit_count_pattern(11))),
            positive_context=_compile_many(
                (
                    r"\bснилс\b",
                    r"\b(?:страхов\w*\s+номер\w*|индивидуальн\w*\s+лицев\w*\s+счет\w*)\b",
                )
            ),
            negative_context=document_negative_context,
            require_context=True,
            context_window=76,
            context_after_window=0,
            priority=95,
            base_confidence=0.80,
        ),
        NumericPIIRule(
            pii_type="INN",
            rule_id="inn_person_number_with_context",
            pattern=_compile(_bounded_value_pattern(_digit_count_pattern(12))),
            positive_context=_compile_many(
                (
                    r"\bинн\b",
                    r"\b(?:идентификацион\w*\s+номер\w*\s+налогоплательщик\w*)\b",
                )
            ),
            negative_context=document_negative_context,
            require_context=True,
            context_window=76,
            context_after_window=0,
            priority=95,
            base_confidence=0.80,
        ),
        NumericPIIRule(
            pii_type="PHONE",
            rule_id="russian_mobile_phone",
            pattern=_compile(_bounded_value_pattern(phone_value_pattern)),
            positive_context=_compile_many(
                (
                    r"\b(?:телефон\w*|мобильн\w*|сотов\w*|домашн\w*|номер\s+телефон\w*)\b",
                    r"\b(?:позвон\w*|перезвон\w*|для\s+связи)\b",
                )
            ),
            negative_context=phone_negative_context,
            require_context=False,
            context_window=68,
            context_after_window=0,
            priority=72,
            base_confidence=0.76,
        ),
        NumericPIIRule(
            pii_type="MSE",
            rule_id="mse_certificate_number_with_context",
            pattern=_compile(_bounded_value_pattern(mse_value_pattern)),
            positive_context=_compile_many(
                (
                    r"\b(?:мсэ|медико\s+социальн\w*\s+экспертиз\w*)\b",
                    r"\b(?:справк\w*\s+мсэ|акт\w*\s+мсэ|инвалидност\w*)\b",
                )
            ),
            negative_context=document_negative_context,
            require_context=True,
            context_window=84,
            context_after_window=0,
            priority=86,
            base_confidence=0.76,
        ),
        NumericPIIRule(
            pii_type="BIRTH_CERTIFICATE",
            rule_id="birth_certificate_number_with_context",
            pattern=_compile(_bounded_value_pattern(_digit_count_pattern(4, 12))),
            positive_context=_compile_many(
                (
                    r"\b(?:свидетельств\w*\s+о\s+рождени\w*)\b",
                    r"\b(?:номер\w*\s+свидетельств\w*)\b",
                )
            ),
            negative_context=document_negative_context,
            require_context=True,
            context_window=96,
            context_after_window=0,
            priority=86,
            base_confidence=0.76,
        ),
        NumericPIIRule(
            pii_type="AGE",
            rule_id="age_number_with_context",
            pattern=_compile(_bounded_value_pattern(age_value_pattern)),
            positive_context=_compile_many(
                (
                    r"\b(?:возраст\w*|сколько\s+лет|полных|исполн(?:илось|ится))\b",
                    r"\b(?:вам\s+сейчас|ему\s+сейчас|ей\s+сейчас|пациент[у]?\s+сейчас)\b",
                    r"\b(?:лет|год|года)\b",
                )
            ),
            negative_context=age_negative_context,
            require_context=True,
            context_window=28,
            context_after_window=28,
            priority=40,
            base_confidence=0.67,
        ),
    )


_DEFAULT_NUMERIC_RULES: Final[tuple[NumericPIIRule, ...]] = _create_default_numeric_rules()


def build_default_numeric_rules() -> tuple[NumericPIIRule, ...]:
    return _DEFAULT_NUMERIC_RULES


def _candidate_from_match(
    text: str,
    rule: NumericPIIRule,
    regex_match: re.Match[str],
) -> tuple[NumericPIIMatch, int] | None:
    start: int = regex_match.start("value")
    end: int = regex_match.end("value")
    start, end = _trim_span(text, start, end)
    if start >= end:
        return None

    raw_value: str = text[start:end]
    normalized_value: str | None = _normalize_value(rule.pii_type, raw_value)
    if normalized_value is None:
        return None

    before_start: int = max(0, start - rule.context_window)
    after_end: int = min(len(text), end + rule.context_after_window)
    scope: str = text[before_start:after_end]
    positive_context_hit: bool = _has_context(rule.positive_context, scope)
    if rule.require_context and not positive_context_hit:
        return None

    if _has_context(rule.negative_context, scope):
        return None

    metadata: dict[str, object] = _build_metadata(rule.pii_type, normalized_value)
    metadata["positive_context_hit"] = positive_context_hit
    metadata["priority"] = rule.priority
    confidence_value: float = _confidence(rule, positive_context_hit, metadata)
    return (
        NumericPIIMatch(
            pii_type=rule.pii_type,
            start=start,
            end=end,
            value=raw_value,
            normalized_value=normalized_value,
            confidence=confidence_value,
            rule_id=rule.rule_id,
            context=scope,
            metadata=metadata,
        ),
        rule.priority,
    )


def _spans_overlap(first: NumericPIIMatch, second: NumericPIIMatch) -> bool:
    return first.start < second.end and second.start < first.end


def _resolve_overlaps(candidates: Sequence[tuple[NumericPIIMatch, int]]) -> tuple[NumericPIIMatch, ...]:
    ranked_candidates: list[tuple[NumericPIIMatch, int]] = sorted(
        candidates,
        key=lambda item: (
            -item[1],
            -item[0].confidence,
            -(item[0].end - item[0].start),
            item[0].start,
        ),
    )
    selected_matches: list[NumericPIIMatch] = []
    for candidate, _priority in ranked_candidates:
        if not any(_spans_overlap(candidate, selected) for selected in selected_matches):
            selected_matches.append(candidate)
    return tuple(sorted(selected_matches, key=lambda item: (item.start, item.end)))


def find_numeric_pii(
    text: str,
    rules: Sequence[NumericPIIRule] | None = None,
) -> tuple[NumericPIIMatch, ...]:
    active_rules: Sequence[NumericPIIRule] = _DEFAULT_NUMERIC_RULES if rules is None else rules
    candidates: list[tuple[NumericPIIMatch, int]] = []
    for rule in active_rules:
        for regex_match in rule.pattern.finditer(text):
            candidate: tuple[NumericPIIMatch, int] | None = _candidate_from_match(
                text,
                rule,
                regex_match,
            )
            if candidate is not None:
                candidates.append(candidate)
    return _resolve_overlaps(candidates)


def _mask_matches(
    text: str,
    matches: Sequence[NumericPIIMatch],
    replacements: Mapping[NumericPIIType, str],
) -> str:
    parts: list[str] = []
    cursor: int = 0
    for match in matches:
        replacement: str = replacements.get(match.pii_type, f"[{match.pii_type}]")
        parts.append(text[cursor:match.start])
        parts.append(replacement)
        cursor = match.end
    parts.append(text[cursor:])
    return "".join(parts)


def mask_numeric_pii(
    text: str,
    replacement_by_type: Mapping[NumericPIIType, str] | None = None,
) -> str:
    replacements: Mapping[NumericPIIType, str] = replacement_by_type or {}
    matches: tuple[NumericPIIMatch, ...] = find_numeric_pii(text)
    return _mask_matches(text, matches, replacements)
