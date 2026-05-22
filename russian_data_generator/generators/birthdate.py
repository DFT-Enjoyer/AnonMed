from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Final

from num2words import num2words

from ..config.lists import MONTHS
from ..formatting.numbers import digits_to_words


def generate_birthdate(min_age: int = 1, max_age: int = 90) -> date:
    today: date = date.today()
    start: date = today - timedelta(days=max_age * 365)
    end: date = today - timedelta(days=min_age * 365)
    delta: timedelta = end - start
    random_days: int = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

def verbalize_birthdate(dt: date, mode: str = "words") -> str:
    if mode == "digits":
        value: str = dt.strftime("%d%m%Y")
        return digits_to_words(value)

    day: str = num2words(dt.day, lang="ru")
    month: str = MONTHS[dt.month]
    year: str = num2words(dt.year, lang="ru")
    return f"{day} {month} {year} года"

def generate_birthdate_raw(min_age: int = 1, max_age: int = 90) -> str:
    """Возвращает дату рождения в формате 'ДД.ММ.ГГГГ' (например, '15.03.1992')."""
    dt = generate_birthdate(min_age, max_age)
    return dt.strftime("%d.%m.%Y")

__all__: Final[list[str]] = ["generate_birthdate", "verbalize_birthdate", "generate_birthdate_raw"]
