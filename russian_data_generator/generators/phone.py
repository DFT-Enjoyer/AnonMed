from __future__ import annotations

import random
from typing import Final

from ..config.groupings import GROUPINGS
from ..config.lists import MOBILE_PREFIXES, PREFIX_MAP
from ..formatting.numbers import verbalize_by_mode


def generate_mobile_phone() -> str:
    prefix: str = random.choice(MOBILE_PREFIXES)
    tail: str = f"{random.randint(0, 9999999):07d}"
    return f"7{prefix}{tail}"


def generate_landline_phone() -> str:
    return f"{random.randint(100000, 999999):06d}"


def verbalize_mobile_phone(phone: str, grouping: str, prefix: str) -> str:
    if prefix not in PREFIX_MAP:
        raise ValueError(f"prefix должен быть одним из {list(PREFIX_MAP.keys())}")

    prefix_words: str = PREFIX_MAP[prefix]
    rest: str = phone[1:]
    phone_words: str = verbalize_by_mode(rest, grouping, GROUPINGS["phone_mobile"])
    return f"{prefix_words} {phone_words}"


def verbalize_landline_phone(phone: str, grouping: str) -> str:
    return verbalize_by_mode(phone, grouping, GROUPINGS["phone_landline"])

def generate_mobile_phone_raw() -> str:
    """Возвращает мобильный номер в цифровом виде (11 цифр, начинается с 7)."""
    return generate_mobile_phone()

def generate_landline_phone_raw() -> str:
    """Возвращает городской номер в цифровом виде (6 цифр)."""
    return generate_landline_phone()

__all__: Final[list[str]] = [
    "generate_landline_phone",
    "generate_landline_phone_raw",
    "generate_mobile_phone",
    "generate_mobile_phone_raw",
    "verbalize_landline_phone",
    "verbalize_mobile_phone",
]
