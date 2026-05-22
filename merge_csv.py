#python merge_csv.py data1.csv data2.csv -n 50 -o combined.csv

import argparse
import csv
from pathlib import Path

def read_first_n_rows(filename: str, n: int) -> tuple[list[str], list[list[str]]]:
    """
    Читает первые n строк CSV-файла.
    Возвращает заголовок (список имён колонок) и список строк (списки значений).
    """
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append(row)
    return header, rows

def merge_horizontal(file1: str, file2: str, n: int, output: str, drop_index: bool = True):
    """
    Объединяет первые n строк из file1 и file2 по горизонтали и записывает в output.
    Параметр drop_index: если True, удаляет столбцы с именем 'index' (регистр игнорируется).
    """
    header1, rows1 = read_first_n_rows(file1, n)
    header2, rows2 = read_first_n_rows(file2, n)

    # Убираем колонки с именем 'index' при необходимости
    def filter_header_and_rows(header, rows):
        idx_cols = [i for i, col in enumerate(header) if col.strip().lower() == "index"]
        new_header = [col for i, col in enumerate(header) if i not in idx_cols]
        new_rows = []
        for row in rows:
            new_row = [val for i, val in enumerate(row) if i not in idx_cols]
            new_rows.append(new_row)
        return new_header, new_rows

    if drop_index:
        header1, rows1 = filter_header_and_rows(header1, rows1)
        header2, rows2 = filter_header_and_rows(header2, rows2)

    # Разрешаем конфликты имён: добавляем суффикс "_1" / "_2" если имена совпадают
    all_columns = list(header1)
    for col in header2:
        if col in header1:
            # Если колонка уже есть, добавляем к ней суффикс
            col_new = col + "_2"
        else:
            col_new = col
        all_columns.append(col_new)

    # Проверяем, что количество строк совпадает
    if len(rows1) != len(rows2):
        raise ValueError(f"Файлы содержат разное количество строк после фильтрации: {len(rows1)} и {len(rows2)}")

    # Запись результата
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(all_columns)
        for r1, r2 in zip(rows1, rows2):
            writer.writerow(r1 + r2)

    print(f"Объединённый файл сохранён в {output}, строк: {len(rows1)}")

def main():
    parser = argparse.ArgumentParser(
        description="Склеить два CSV по первым N строкам (горизонтально)."
    )
    parser.add_argument("file1", help="Путь к первому CSV-файлу")
    parser.add_argument("file2", help="Путь ко второму CSV-файлу")
    parser.add_argument("-n", type=int, required=True, help="Количество первых строк для объединения")
    parser.add_argument("-o", "--output", default="merged.csv", help="Имя выходного файла")
    parser.add_argument("--keep-index", action="store_true", help="Не удалять колонку 'index'")
    args = parser.parse_args()

    merge_horizontal(args.file1, args.file2, args.n, args.output, drop_index=not args.keep_index)

if __name__ == "__main__":
    main()