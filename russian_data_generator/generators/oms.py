from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Final

from ..config.groupings import GROUPINGS
from ..config.lists import OMS_DUPLICATE_VALUES, OMS_DUPLICATE_WEIGHTS, REGION_CODE_VALUES
from ..formatting.numbers import verbalize_by_mode


class OMSGenerator:
    @staticmethod
    def _checksum_mod10_hl7(digits15: str) -> int:
        total: int = 0
        for i, ch in enumerate(reversed(digits15)):
            n: int = int(ch)
            if i % 2 == 0:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return (10 - total % 10) % 10

    def generate(self) -> str:
        region: str = random.choice(REGION_CODE_VALUES)
        duplicate: int = random.choices(
            OMS_DUPLICATE_VALUES,
            weights=OMS_DUPLICATE_WEIGHTS,
        )[0]

        birth_date, is_male = self._random_birth_date()
        year: str = birth_date.strftime("%Y")
        month: str = birth_date.strftime("%m")
        day: int = int(birth_date.strftime("%d"))

        if is_male:
            day += 50

        day_str: str = f"{day:02d}"
        sequence: str = f"{random.randint(1, 999999):06d}"
        first_digits: str = f"{region}{duplicate}{year}{month}{day_str}{sequence}"
        checksum: int = self._checksum_mod10_hl7(first_digits)
        return first_digits + str(checksum)

    def _random_birth_date(self) -> tuple[date, bool]:
        today: date = date.today()
        start_date: date = date(today.year - 100, 1, 1)
        end_date: date = today
        delta_days: int = (end_date - start_date).days
        random_date: date = start_date + timedelta(days=random.randint(0, delta_days))
        is_male: bool = random.choice([True, False])
        return random_date, is_male


_oms_gen: OMSGenerator = OMSGenerator()


def generate_oms() -> str:
    return _oms_gen.generate()


def verbalize_oms(oms: str, mode: str) -> str:
    return verbalize_by_mode(oms, mode, GROUPINGS["oms"])


__all__: Final[list[str]] = ["OMSGenerator", "generate_oms", "verbalize_oms"]
