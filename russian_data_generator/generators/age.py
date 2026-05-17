from __future__ import annotations

import random
from typing import Final

from num2words import num2words


def generate_age(min_age: int = 1, max_age: int = 100) -> int:
    return random.randint(min_age, max_age)


def verbalize_age(age: int) -> str:
    words: str = num2words(age, lang="ru")
    return words


__all__: Final[list[str]] = ["generate_age", "verbalize_age"]
