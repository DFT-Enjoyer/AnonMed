from __future__ import annotations

import random
from typing import Final

from ..config.groupings import GROUPINGS
from ..formatting.numbers import verbalize_by_mode


def calculate_snils_checksum(number9: str) -> str:
    total: int = sum(int(digit) * (9 - idx) for idx, digit in enumerate(number9))

    if total < 100:
        checksum: int = total
    elif total in (100, 101):
        checksum = 0
    else:
        checksum = total % 101
        if checksum == 100:
            checksum = 0

    return f"{checksum:02d}"


def generate_snils() -> str:
    base: str = f"{random.randint(0, 999999999):09d}"
    checksum: str = calculate_snils_checksum(base)
    return base + checksum


def verbalize_snils(snils: str, mode: str) -> str:
    return verbalize_by_mode(snils, mode, GROUPINGS["snils"])

def generate_snils_raw() -> str:
    """Возвращает СНИЛС в виде 11 цифр без дефисов/пробелов."""
    return generate_snils()

__all__: Final[list[str]] = [
    "calculate_snils_checksum",
    "generate_snils",
    "generate_snils_raw",
    "verbalize_snils",
]
