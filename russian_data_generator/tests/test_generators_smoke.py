from __future__ import annotations

import unittest
from datetime import date
from typing import Final

from ..config.groupings import GROUPINGS
from ..generators.age import generate_age, verbalize_age
from ..generators.birth_certificate import generate_birth_certificate, verbalize_birth_certificate
from ..generators.birthdate import generate_birthdate, verbalize_birthdate
from ..generators.driver_license import generate_driver_license, verbalize_driver_license
from ..generators.inn import generate_inn, verbalize_inn
from ..generators.mse import generate_mse, verbalize_mse
from ..generators.oms import generate_oms, verbalize_oms
from ..generators.passport import generate_passport, verbalize_passport
from ..generators.phone import (
    generate_landline_phone,
    generate_mobile_phone,
    verbalize_landline_phone,
    verbalize_mobile_phone,
)
from ..generators.snils import calculate_snils_checksum, generate_snils, verbalize_snils


class GeneratorSmokeTests(unittest.TestCase):
    def test_phone_generators_and_groupings(self) -> None:
        mobile_phone: str = generate_mobile_phone()
        landline_phone: str = generate_landline_phone()
        self.assertEqual(len(mobile_phone), 11)
        self.assertTrue(mobile_phone.startswith("7"))
        self.assertEqual(len(landline_phone), 6)
        self.assertIsInstance(verbalize_mobile_phone(mobile_phone, "groups", "+7"), str)
        self.assertIsInstance(verbalize_landline_phone(landline_phone, "groups"), str)

    def test_snils_checksum_and_groupings(self) -> None:
        snils: str = generate_snils()
        self.assertEqual(len(snils), 11)
        self.assertEqual(calculate_snils_checksum(snils[:9]), snils[9:])
        self.assertIsInstance(verbalize_snils(snils, "3-3-3-2"), str)

    def test_document_generators(self) -> None:
        passport: str = generate_passport()
        inn: str = generate_inn()
        oms: str = generate_oms()
        driver_license: str = generate_driver_license()
        self.assertEqual(len(passport), 10)
        self.assertEqual(len(inn), 12)
        self.assertGreaterEqual(len(oms), sum(GROUPINGS["oms"]["groups"]))
        self.assertEqual(len(driver_license), 10)
        self.assertIsInstance(verbalize_passport(passport, "groups"), str)
        self.assertIsInstance(verbalize_inn(inn, "groups"), str)
        self.assertIsInstance(verbalize_oms(oms, "groups"), str)
        self.assertIsInstance(verbalize_driver_license(driver_license, "groups"), str)

    def test_structured_generators(self) -> None:
        mse: dict[str, str] = generate_mse()
        certificate: dict[str, str] = generate_birth_certificate()
        self.assertEqual(len(mse["series"]), 4)
        self.assertEqual(len(mse["number"]), 7)
        self.assertEqual(len(certificate["number"]), 6)
        self.assertIsInstance(verbalize_mse(mse, "groups"), str)
        self.assertIsInstance(verbalize_birth_certificate(certificate, "groups"), str)

    def test_birthdate_and_age(self) -> None:
        birthdate: date = generate_birthdate()
        age: int = generate_age()
        self.assertIsInstance(birthdate, date)
        self.assertGreaterEqual(age, 1)
        self.assertLessEqual(age, 100)
        self.assertIsInstance(verbalize_birthdate(birthdate), str)
        self.assertIsInstance(verbalize_age(age), str)


__all__: Final[list[str]] = ["GeneratorSmokeTests"]


if __name__ == "__main__":
    unittest.main()
