from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from datetime import date
from typing import Final

from .config.groupings import GROUPINGS
from .config.lists import PREFIX_MAP
from .generators.age import generate_age, verbalize_age
from .generators.birth_certificate import generate_birth_certificate, verbalize_birth_certificate
from .generators.birthdate import generate_birthdate, verbalize_birthdate
from .generators.driver_license import generate_driver_license, verbalize_driver_license
from .generators.inn import generate_inn, verbalize_inn
from .generators.mse import generate_mse, verbalize_mse
from .generators.oms import generate_oms, verbalize_oms
from .generators.passport import generate_passport, verbalize_passport
from .generators.phone import (
    generate_landline_phone,
    generate_mobile_phone,
    verbalize_landline_phone,
    verbalize_mobile_phone,
)
from .generators.snils import generate_snils, verbalize_snils

DATA_TYPES: Final[list[str]] = list(GROUPINGS.keys()) + ["age"]


def _choose_grouping_mode(data_type: str) -> str:
    modes: list[str] = list(GROUPINGS[data_type].keys())
    return random.choice(modes)


def _choose_prefix() -> str:
    prefixes: list[str] = list(PREFIX_MAP.keys())
    return random.choice(prefixes)


def _build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Генератор устных представлений документов и номеров"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=DATA_TYPES,
        help="Тип генерируемых данных",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Количество примеров",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    for _ in range(args.count):
        if args.type == "phone_mobile":
            mobile_phone: str = generate_mobile_phone()
            mobile_grouping: str = _choose_grouping_mode("phone_mobile")
            prefix: str = _choose_prefix()
            print(verbalize_mobile_phone(mobile_phone, grouping=mobile_grouping, prefix=prefix))

        elif args.type == "phone_landline":
            landline_phone: str = generate_landline_phone()
            landline_grouping: str = _choose_grouping_mode("phone_landline")
            print(verbalize_landline_phone(landline_phone, grouping=landline_grouping))

        elif args.type == "snils":
            snils: str = generate_snils()
            snils_mode: str = _choose_grouping_mode("snils")
            print(verbalize_snils(snils, snils_mode))

        elif args.type == "passport":
            passport: str = generate_passport()
            passport_mode: str = _choose_grouping_mode("passport")
            print(verbalize_passport(passport, passport_mode))

        elif args.type == "birthdate":
            birthdate: date = generate_birthdate()
            birthdate_mode: str = _choose_grouping_mode("birthdate")
            print(verbalize_birthdate(birthdate, birthdate_mode))

        elif args.type == "inn":
            inn: str = generate_inn()
            inn_mode: str = _choose_grouping_mode("inn")
            print(verbalize_inn(inn, inn_mode))

        elif args.type == "oms":
            oms: str = generate_oms()
            oms_mode: str = _choose_grouping_mode("oms")
            print(verbalize_oms(oms, oms_mode))

        elif args.type == "age":
            age: int = generate_age()
            print(verbalize_age(age))

        elif args.type == "mse":
            mse: dict[str, str] = generate_mse()
            mse_mode: str = _choose_grouping_mode("mse")
            print(verbalize_mse(mse, mse_mode))

        elif args.type == "birth_certificate":
            certificate: dict[str, str] = generate_birth_certificate()
            certificate_mode: str = _choose_grouping_mode("birth_certificate")
            print(verbalize_birth_certificate(certificate, certificate_mode))

        elif args.type == "driver_license":
            driver_license: str = generate_driver_license()
            driver_license_mode: str = _choose_grouping_mode("driver_license")
            print(verbalize_driver_license(driver_license, driver_license_mode))

        else:
            raise ValueError(f"Неизвестный тип: {args.type}")


if __name__ == "__main__":
    main()


__all__: Final[list[str]] = ["main"]
