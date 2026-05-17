from __future__ import annotations

import random
from typing import Final

from ..config.groupings import GROUPINGS
from ..config.lists import (
    BIRTH_CERTIFICATE_CYR_LETTER_NAME,
    BIRTH_CERTIFICATE_ROMAN_TO_WORD,
    BIRTH_CERTIFICATE_ROMAN_VALUES,
    BIRTH_CERTIFICATE_VALID_LETTERS,
)
from ..formatting.numbers import verbalize_by_mode


def generate_birth_certificate() -> dict[str, str]:
    roman: str = random.choice(BIRTH_CERTIFICATE_ROMAN_VALUES)
    letters: list[str] = random.choices(BIRTH_CERTIFICATE_VALID_LETTERS, k=2)
    series: str = f"{roman}-{''.join(letters)}"
    number: str = f"{random.randint(0, 999999):06d}"
    return {"series": series, "number": number}


def verbalize_birth_certificate(data: dict[str, str], mode: str = "groups") -> str:
    series: str = data["series"]
    number: str = data["number"]
    roman, letters_str = series.split("-")
    roman_word: str = BIRTH_CERTIFICATE_ROMAN_TO_WORD.get(roman, roman.lower())
    letter_words: list[str] = [BIRTH_CERTIFICATE_CYR_LETTER_NAME[ch] for ch in letters_str]
    number_words: str = verbalize_by_mode(number, mode, GROUPINGS["birth_certificate"])
    return " ".join([roman_word, "тире"] + letter_words + [number_words])


__all__: Final[list[str]] = [
    "generate_birth_certificate",
    "verbalize_birth_certificate",
]
