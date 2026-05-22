from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from num2words import num2words

DIGIT_WORDS: Final[dict[str, str]] = {
    "0": "ноль",
    "1": "один",
    "2": "два",
    "3": "три",
    "4": "четыре",
    "5": "пять",
    "6": "шесть",
    "7": "семь",
    "8": "восемь",
    "9": "девять",
}


def digits_to_words(number: str) -> str:
    return " ".join(DIGIT_WORDS[digit] for digit in number)


def number_to_words_ru(number: int) -> str:
    words: str = num2words(number, lang="ru")
    return words


def split_by_groups(number: str, groups: Sequence[int]) -> list[str]:
    result: list[str] = []
    pos: int = 0

    for group_size in groups:
        result.append(number[pos:pos + group_size])
        pos += group_size

    return result


def groups_to_words(groups: Sequence[str]) -> str:
    """Преобразует группы цифр в слова, явно проговаривая ведущие нули."""
    parts = []
    for group in groups:
        # если вся группа состоит только из нулей, произносим столько же раз «ноль»
        if all(ch == '0' for ch in group):
            parts.append(" ".join(["ноль"] * len(group)))
        elif group[0] == '0':
            # ведущий ноль: говорим "ноль" и затем числовое представление остатка
            parts.append("ноль " + number_to_words_ru(int(group[1:])))
        else:
            # без ведущего нуля – обычное числовое озвучивание
            parts.append(number_to_words_ru(int(group)))
    return " ".join(parts)


def verbalize_by_mode(value: str, mode: str, config: Mapping[str, Sequence[int]]) -> str:
    grouping: Sequence[int] = config[mode]
    groups: list[str] = split_by_groups(value, grouping)
    return groups_to_words(groups)


__all__: Final[list[str]] = [
    "DIGIT_WORDS",
    "digits_to_words",
    "groups_to_words",
    "number_to_words_ru",
    "split_by_groups",
    "verbalize_by_mode",
]
