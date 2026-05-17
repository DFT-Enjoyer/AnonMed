from __future__ import annotations

import random
from typing import Final

from ..config.groupings import GROUPINGS
from ..config.lists import DRIVER_LICENSE_REGION_CODES
from ..formatting.numbers import verbalize_by_mode


class DriverLicenseGenerator:
    def __init__(self) -> None:
        self.region_codes: list[int] = DRIVER_LICENSE_REGION_CODES

    def generate(self) -> str:
        region: str = f"{random.choice(self.region_codes):02d}"
        division: str = f"{random.randint(0, 99):02d}"
        series: str = region + division
        number: str = f"{random.randint(0, 999999):06d}"
        return series + number


_dl_gen: DriverLicenseGenerator = DriverLicenseGenerator()


def generate_driver_license() -> str:
    return _dl_gen.generate()


def verbalize_driver_license(license_num: str, mode: str = "groups") -> str:
    return verbalize_by_mode(license_num, mode, GROUPINGS["driver_license"])


__all__: Final[list[str]] = [
    "DriverLicenseGenerator",
    "generate_driver_license",
    "verbalize_driver_license",
]
