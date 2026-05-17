from __future__ import annotations

import random
from datetime import datetime
from typing import Final

from ..config.groupings import GROUPINGS
from ..config.lists import PASSPORT_MIN_YEAR, REGION_CODE_VALUES
from ..formatting.numbers import verbalize_by_mode


class PassportGenerator:
    def __init__(self) -> None:
        self.region_codes: list[str] = REGION_CODE_VALUES
        self.max_year: int = datetime.now().year

    def _generate_year(self) -> str:
        year: int = random.randint(PASSPORT_MIN_YEAR, self.max_year)
        return f"{year % 100:02d}"

    def _generate_serial_number(self) -> str:
        return f"{random.randint(0, 999999):06d}"

    def generate(self) -> str:
        region_code: str = random.choice(self.region_codes)
        year_code: str = self._generate_year()
        series: str = f"{region_code}{year_code}"
        number: str = self._generate_serial_number()
        return series + number


_passport_gen: PassportGenerator = PassportGenerator()


def generate_passport() -> str:
    return _passport_gen.generate()


def verbalize_passport(passport_data: str, mode: str = "groups") -> str:
    return verbalize_by_mode(passport_data, mode, GROUPINGS["passport"])


__all__: Final[list[str]] = [
    "PassportGenerator",
    "generate_passport",
    "verbalize_passport",
]
