import argparse
import csv
import random
from russian_data_generator.config.groupings import GROUPINGS
from russian_data_generator.config.lists import PREFIX_MAP
from russian_data_generator.generators.phone import (
    generate_mobile_phone,
    generate_landline_phone,
    verbalize_mobile_phone,
    verbalize_landline_phone,
)
from russian_data_generator.generators.snils import generate_snils, verbalize_snils
from russian_data_generator.generators.passport import generate_passport, verbalize_passport
from russian_data_generator.generators.birthdate import generate_birthdate, verbalize_birthdate
from russian_data_generator.generators.inn import generate_inn, verbalize_inn
from russian_data_generator.generators.oms import generate_oms, verbalize_oms
from russian_data_generator.generators.age import generate_age, verbalize_age
from russian_data_generator.generators.mse import generate_mse, verbalize_mse
from russian_data_generator.generators.birth_certificate import (
    generate_birth_certificate,
    verbalize_birth_certificate,
)
from russian_data_generator.generators.driver_license import (
    generate_driver_license,
    verbalize_driver_license,
)


def generate_one_row():
    """
    Возвращает словарь с устными и сырыми данными, соответствующими друг другу.
    Ключи: <type> – устная форма, <type>_raw – цифровое (или буквенно-цифровое) представление.
    """
    row = {}

    # ----- Мобильный телефон -----
    mob = generate_mobile_phone()
    mob_grouping = random.choice(list(GROUPINGS["phone_mobile"].keys()))
    mob_prefix = random.choice(list(PREFIX_MAP.keys()))
    row["phone_mobile"] = verbalize_mobile_phone(mob, prefix=mob_prefix, grouping=mob_grouping)
    row["phone_mobile_raw"] = mob

    # ----- Городской телефон -----
    land = generate_landline_phone()
    land_grouping = random.choice(list(GROUPINGS["phone_landline"].keys()))
    row["phone_landline"] = verbalize_landline_phone(land, grouping=land_grouping)
    row["phone_landline_raw"] = land

    # ----- СНИЛС -----
    snils = generate_snils()
    snils_mode = random.choice(list(GROUPINGS["snils"].keys()))
    row["snils"] = verbalize_snils(snils, snils_mode)
    row["snils_raw"] = snils

    # ----- Паспорт -----
    passport = generate_passport()                     # строка из 10 цифр
    passport_mode = random.choice(list(GROUPINGS["passport"].keys()))
    row["passport"] = verbalize_passport(passport, passport_mode)
    row["passport_raw"] = passport

    # ----- Дата рождения -----
    bdate_dt = generate_birthdate()                    # объект date
    bdate_raw = bdate_dt.strftime("%d.%m.%Y")          # сырое представление
    bdate_mode = random.choice(list(GROUPINGS["birthdate"].keys()))
    row["birthdate"] = verbalize_birthdate(bdate_dt, bdate_mode)
    row["birthdate_raw"] = bdate_raw

    # ----- ИНН -----
    inn = generate_inn()
    inn_mode = random.choice(list(GROUPINGS["inn"].keys()))
    row["inn"] = verbalize_inn(inn, inn_mode)
    row["inn_raw"] = inn

    # ----- ОМС -----
    oms = generate_oms()
    oms_mode = random.choice(list(GROUPINGS["oms"].keys()))
    row["oms"] = verbalize_oms(oms, oms_mode)
    row["oms_raw"] = oms

    # ----- Возраст -----
    age = generate_age()
    row["age"] = verbalize_age(age)
    row["age_raw"] = str(age)

    # ----- МСЭ -----
    mse_data = generate_mse()                          # словарь с series и number
    mse_raw = mse_data["series"] + mse_data["number"]
    mse_mode = random.choice(list(GROUPINGS["mse"].keys()))
    row["mse"] = verbalize_mse(mse_data, mse_mode)
    row["mse_raw"] = mse_raw

    # ----- Свидетельство о рождении -----
    cert_data = generate_birth_certificate()           # словарь с series и number
    cert_raw = cert_data["series"] + cert_data["number"]
    cert_mode = random.choice(list(GROUPINGS["birth_certificate"].keys()))
    row["birth_certificate"] = verbalize_birth_certificate(cert_data, cert_mode)
    row["birth_certificate_raw"] = cert_raw

    # ----- Водительское удостоверение -----
    dl = generate_driver_license()
    dl_mode = random.choice(list(GROUPINGS["driver_license"].keys()))
    row["driver_license"] = verbalize_driver_license(dl, dl_mode)
    row["driver_license_raw"] = dl

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Генерация всех типов данных в CSV (устная форма + опционально цифровая)"
    )
    parser.add_argument("--count", "-n", type=int, default=1, help="Количество строк")
    parser.add_argument("--output", "-o", default="all_types.csv", help="Имя выходного CSV-файла")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Добавить столбцы с цифровыми (raw) данными сразу после устных",
    )
    args = parser.parse_args()

    # Порядок типов (устных колонок)
    verbal_order = [
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

    # Формируем заголовок в зависимости от флага --include-raw
    header = []
    for col in verbal_order:
        header.append(col)
        if args.include_raw:
            header.append(f"{col}_raw")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for _ in range(args.count):
            row_data = generate_one_row()
            # Собираем строку в том же порядке
            row = []
            for col in verbal_order:
                row.append(row_data[col])
                if args.include_raw:
                    row.append(row_data[f"{col}_raw"])
            writer.writerow(row)

    print(f"Сгенерировано {args.count} строк в {args.output}")


if __name__ == "__main__":
    main()