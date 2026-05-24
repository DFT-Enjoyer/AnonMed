#!/usr/bin/env python3
"""
Объединяющий скрипт для подготовки объединённого датасета.

Запускается из папки gen_all или из любого места; самостоятельно определяет
корень проекта по своему расположению (ожидает, что gen_all находится в корне).

Шаги:
1. Случайный отбор N строк из файлов-источников (DataForGen).
2. Генерация цифровых персональных данных (raw) через generate_numbers/generate_all_csv.py.
3. Транслитерация колонок email и nicks_with_at через transclit/translit.py.
4. Сохранение итогового CSV с полным набором колонок.
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "full_address",
    "nicks_with_at",
    "email",
    "name",
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
    "full_company_name",
]

# Сопоставление файлов и нужных колонок (порядок важен)
SOURCE_FILES = {
    "full_address_only.csv": ["full_address"],
    "name_email_filtered.csv": ["email", "name"],   # порядок: сначала email, потом name
    "companies_names_only.csv": ["full_company_name"],
    "nicks_with_at.csv": ["nicks_with_at"],          # исходная колонка tg_nicks будет переименована
}

# Переименование колонок для nicks_with_at.csv
COLUMN_RENAME = {
    "tg_nicks": "nicks_with_at"
}

# ---------------------------------------------------------------------------
# Определение корня проекта
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR            # теперь корнем считается сама папка со скриптом

# Проверка папок
REQUIRED_DIRS = ["DataForGen", "generate_numbers", "transclit"]
for d in REQUIRED_DIRS:
    if not (PROJECT_ROOT / d).is_dir():
        print(f"Ошибка: папка '{d}' не найдена в {PROJECT_ROOT}.", file=sys.stderr)
        sys.exit(1)

DATA_FOR_GEN = PROJECT_ROOT / "DataForGen"
GEN_NUMBERS_SCRIPT = PROJECT_ROOT / "generate_numbers" / "generate_all_csv.py"
TRANSLIT_SCRIPT = PROJECT_ROOT / "transclit" / "translit.py"

# ---------------------------------------------------------------------------
# Функции
# ---------------------------------------------------------------------------

def load_random_rows(data_dir: Path, n: int) -> dict:
    """
    Загружает n случайных неповторяющихся строк из каждого файла в SOURCE_FILES.
    Возвращает словарь {имя_колонки: [значение1, ...]} длиной n.
    При нехватке строк или отсутствии нужной колонки печатает ошибку и завершает работу.
    """
    random.seed()  # можно зафиксировать seed для воспроизводимости
    combined = {}

    for filename, columns in SOURCE_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Ошибка: файл {filepath} не найден.", file=sys.stderr)
            sys.exit(1)

        # Первый проход: проверка наличия нужных колонок
        with open(filepath, "r", encoding="utf-8-sig") as f:
            temp_reader = csv.DictReader(f, skipinitialspace=True)
            available_cols = temp_reader.fieldnames

            for col in columns:
                # Определяем, под каким именем колонка ожидается в файле
                source_col = col
                for orig, target in COLUMN_RENAME.items():
                    if target == col:
                        source_col = orig
                        break
                if source_col not in available_cols:
                    print(f"Ошибка: колонка '{source_col}' не найдена в {filename}. "
                          f"Доступные колонки: {available_cols}", file=sys.stderr)
                    sys.exit(1)

        # Второй проход: резервуарная выборка n строк
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            sample = []
            for i, row in enumerate(reader):
                if i < n:
                    sample.append(row)
                else:
                    j = random.randint(0, i)
                    if j < n:
                        sample[j] = row

            if len(sample) < n:
                print(f"Ошибка: в файле {filename} недостаточно строк. "
                      f"Доступно: {len(sample)}, требуется: {n}.", file=sys.stderr)
                sys.exit(1)

        # Извлекаем нужные колонки с учётом переименования
        for col in columns:
            source_col = col
            for orig, target in COLUMN_RENAME.items():
                if target == col:
                    source_col = orig
                    break
            target_col = col
            combined[target_col] = [row[source_col] for row in sample]

    return combined


def generate_numbers(count: int, output_path: str) -> None:
    """Вызывает generate_all_csv.py с параметрами --count, --output и --include-raw."""
    if not GEN_NUMBERS_SCRIPT.exists():
        print(f"Ошибка: скрипт генерации чисел не найден: {GEN_NUMBERS_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, str(GEN_NUMBERS_SCRIPT),
        "--count", str(count),
        "--output", str(output_path),
        "--include-raw"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Ошибка при генерации чисел:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Генерация чисел завершена.")


def clean_number_columns(csv_path: str) -> list:
    """
    Оставляет только колонки с суффиксом '_raw', удаляет суффикс.
    Возвращает список словарей с чистыми данными.
    """
    cleaned_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_columns = [c for c in reader.fieldnames if c.endswith("_raw")]
        if not raw_columns:
            print("Ошибка: в сгенерированном файле нет колонок с суффиксом '_raw'.", file=sys.stderr)
            sys.exit(1)

        rename = {c: c[:-4] for c in raw_columns}  # phone_mobile_raw -> phone_mobile
        for row in reader:
            cleaned = {rename[orig]: row[orig] for orig in raw_columns}
            cleaned_rows.append(cleaned)

    return cleaned_rows


def merge_text_and_numbers(text_columns: dict, numbers_rows: list) -> list:
    """Объединяет текстовые и числовые колонки в список строк согласно REQUIRED_COLUMNS."""
    merged = []
    for i in range(len(numbers_rows)):
        row = {}
        for col in REQUIRED_COLUMNS:
            if col in text_columns:
                row[col] = text_columns[col][i]
            else:
                row[col] = numbers_rows[i].get(col, "")
        merged.append(row)
    return merged


def apply_transliteration(input_csv: str, output_csv: str, columns: list) -> None:
    """Запускает transclit/translit.py в режиме 'words' для указанных колонок."""
    if not TRANSLIT_SCRIPT.exists():
        print(f"Ошибка: скрипт транслитерации не найден: {TRANSLIT_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, str(TRANSLIT_SCRIPT),
        "words",
        input_csv,
        output_csv,
        "--columns", *columns
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Ошибка транслитерации:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Транслитерация завершена.")


def main():
    parser = argparse.ArgumentParser(
        description="Сборка объединённого датасета персональных данных."
    )
    parser.add_argument("--n", type=int, default=10000,
                        help="Количество строк для выборки (по умолчанию 10000)")
    parser.add_argument("--output-dir", type=str, default="script_data",
                        help="Папка для выходного файла")
    parser.add_argument("--output-file", type=str, default="script_data.csv",
                        help="Имя выходного CSV")
    args = parser.parse_args()

    # 1. Загрузка текстовых данных
    print(f"Загрузка {args.n} случайных строк из файлов DataForGen...")
    text_data = load_random_rows(DATA_FOR_GEN, args.n)

    # 2. Генерация чисел
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        num_temp = tmp.name
    print("Генерация числовых данных...")
    generate_numbers(args.n, num_temp)
    numbers_rows = clean_number_columns(num_temp)
    os.unlink(num_temp)  # удаляем временный файл

    # 3. Объединение текстовых и числовых данных
    print("Объединение колонок...")
    merged_rows = merge_text_and_numbers(text_data, numbers_rows)

    # Сохраняем промежуточный CSV перед транслитерацией
    temp_combined = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.DictWriter(temp_combined, fieldnames=REQUIRED_COLUMNS)
    writer.writeheader()
    writer.writerows(merged_rows)
    temp_combined.close()

    # 4. Транслитерация
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file
    print("Применение транслитерации к email и nicks_with_at...")
    apply_transliteration(temp_combined.name, str(output_path), ["email", "nicks_with_at"])
    os.unlink(temp_combined.name)

    print(f"Готово. Итоговый файл сохранён: {output_path}")
    print(f"Количество строк: {args.n}")


if __name__ == "__main__":
    main()