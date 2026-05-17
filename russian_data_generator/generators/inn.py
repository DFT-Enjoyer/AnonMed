from __future__ import annotations

import random
from typing import Final

from ..config.groupings import GROUPINGS
from ..config.lists import INN_COEFFICIENTS_11, INN_COEFFICIENTS_12
from ..formatting.numbers import verbalize_by_mode


def generate_inn() -> str:
    digits: list[str] = [str(random.randint(0, 9)) for _ in range(10)]

    n11: int = sum(
        int(digit) * coefficient
        for digit, coefficient in zip(digits[:10], INN_COEFFICIENTS_11)
    ) % 11 % 10
    digits.append(str(n11))

    n12: int = sum(
        int(digit) * coefficient
        for digit, coefficient in zip(digits[:11], INN_COEFFICIENTS_12)
    ) % 11 % 10
    digits.append(str(n12))

    return "".join(digits)


def verbalize_inn(inn: str, mode: str = "digits") -> str:
    return verbalize_by_mode(inn, mode, GROUPINGS["inn"])


__all__: Final[list[str]] = ["generate_inn", "verbalize_inn"]
