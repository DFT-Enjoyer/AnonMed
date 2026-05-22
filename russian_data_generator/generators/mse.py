from __future__ import annotations

import random
from datetime import datetime
from typing import Final

from ..config.groupings import GROUPINGS
from ..formatting.numbers import verbalize_by_mode


class MSEGenerator:
    def __init__(self, max_year: int | None = None) -> None:
        self.max_year: int = datetime.now().year if max_year is None else max_year

    def generate(self) -> dict[str, str]:
        bureau: str = f"{random.randint(1, 99):02d}"
        year_short: str = f"{random.randint(0, self.max_year % 100):02d}"
        series: str = bureau + year_short
        number: str = f"{random.randint(1, 9999999):07d}"
        return {"series": series, "number": number}


_mse_gen: MSEGenerator = MSEGenerator()


def generate_mse() -> dict[str, str]:
    return _mse_gen.generate()


def verbalize_mse(mse: dict[str, str], mode: str = "groups") -> str:
    full: str = mse["series"] + mse["number"]
    return verbalize_by_mode(full, mode, GROUPINGS["mse"])

def generate_mse_raw() -> str:
    """Возвращает номер МСЭ в виде строки из 11 цифр (серия + номер)."""
    data = _mse_gen.generate()
    return data["series"] + data["number"]

__all__: Final[list[str]] = ["MSEGenerator", "generate_mse", "generate_mse_raw", "verbalize_mse"]
