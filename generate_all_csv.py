import argparse
import csv
import random
from russian_data_generator.config.groupings import GROUPINGS
from russian_data_generator.generators.phone import (
    generate_mobile_phone, generate_landline_phone,
    verbalize_mobile_phone, verbalize_landline_phone,
)
from russian_data_generator.generators.snils import generate_snils, verbalize_snils
from russian_data_generator.generators.passport import generate_passport, verbalize_passport
from russian_data_generator.generators.birthdate import generate_birthdate, verbalize_birthdate
from russian_data_generator.generators.inn import generate_inn, verbalize_inn
from russian_data_generator.generators.oms import generate_oms, verbalize_oms
from russian_data_generator.generators.age import generate_age, verbalize_age
from russian_data_generator.generators.mse import generate_mse, verbalize_mse
from russian_data_generator.generators.birth_certificate import generate_birth_certificate, verbalize_birth_certificate
from russian_data_generator.generators.driver_license import generate_driver_license, verbalize_driver_license


def generate_one_row() -> dict:
    """
    Возвращает словарь с вербализациями всех типов (ключ — тип, значение — строка слов).
    """
    # Мобильный телефон: нужны случайный префикс и случайная группировка
    phone_mob = generate_mobile_phone()
    mob_grouping = random.choice(list(GROUPINGS["phone_mobile"].keys()))  # строковый ключ, например "groups"
    mob_prefix = random.choice(["plus", "8", "7"])
    vm = verbalize_mobile_phone(phone_mob, prefix=mob_prefix, grouping=mob_grouping)

    # Городской телефон
    phone_land = generate_landline_phone()
    land_grouping = random.choice(list(GROUPINGS["phone_landline"].keys()))
    vg = verbalize_landline_phone(phone_land, grouping=land_grouping)

    # СНИЛС
    snils = generate_snils()
    snils_mode = random.choice(list(GROUPINGS["snils"].keys()))
    vs = verbalize_snils(snils, snils_mode)

    # Паспорт (возвращает dict)
    passport_data = generate_passport()
    passport_mode = random.choice(list(GROUPINGS["passport"].keys()))
    vpass = verbalize_passport(passport_data, passport_mode)

    # Дата рождения (возвращает date)
    bdate = generate_birthdate()
    bdate_mode = random.choice(list(GROUPINGS["birthdate"].keys()))
    vbdate = verbalize_birthdate(bdate, bdate_mode)

    # ИНН
    inn = generate_inn()
    inn_mode = random.choice(list(GROUPINGS["inn"].keys()))
    vinn = verbalize_inn(inn, inn_mode)

    # ОМС
    oms = generate_oms()
    oms_mode = random.choice(list(GROUPINGS["oms"].keys()))
    voms = verbalize_oms(oms, oms_mode)

    # Возраст (int)
    age = generate_age()
    vage = verbalize_age(age)

    # МСЭ (dict)
    mse_data = generate_mse()
    mse_mode = random.choice(list(GROUPINGS["mse"].keys()))
    vmse = verbalize_mse(mse_data, mse_mode)

    # Свидетельство о рождении (dict)
    cert_data = generate_birth_certificate()
    cert_mode = random.choice(list(GROUPINGS["birth_certificate"].keys()))
    vcert = verbalize_birth_certificate(cert_data, cert_mode)

    # Водительское удостоверение
    dl = generate_driver_license()
    dl_mode = random.choice(list(GROUPINGS["driver_license"].keys()))
    vdl = verbalize_driver_license(dl, dl_mode)

    return {
        "phone_mobile": vm,
        "phone_landline": vg,
        "snils": vs,
        "passport": vpass,
        "birthdate": vbdate,
        "inn": vinn,
        "oms": voms,
        "age": vage,
        "mse": vmse,
        "birth_certificate": vcert,
        "driver_license": vdl,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Генерация всех типов данных в одну CSV-таблицу (только устная форма)"
    )
    parser.add_argument("--count", "-n", type=int, default=1, help="Количество строк")
    parser.add_argument("--output", "-o", default="all_types.csv", help="Имя выходного CSV-файла")
    args = parser.parse_args()

    # Порядок колонок в CSV
    columns = [
        "phone_mobile",
        "phone_landline",
        "snils",
        "passport",
        "birthdate",
        "inn",
        "oms",
        "age",
        "mse",
        "birth_certificate",
        "driver_license",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for i in range(1, args.count + 1):
            row_data = generate_one_row()
            row = [row_data[col] for col in columns]
            writer.writerow(row)

    print(f"Сгенерировано {args.count} строк в файл {args.output}")


if __name__ == "__main__":
    main()