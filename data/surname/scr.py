#!/usr/bin/env python3
import json
import csv

INPUT_FILE = "surnames_table.jsonl"
OUTPUT_FILE = "surnames.csv"

def main():
    surnames = []
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:  # пропускаем пустые строки
                continue
            try:
                obj = json.loads(line)
                if "text" in obj:
                    surnames.append(obj["text"])
                else:
                    print(f"Предупреждение: строка {line_num} не содержит поля 'text'")
            except json.JSONDecodeError as e:
                print(f"Ошибка в строке {line_num}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["surnames"])  # заголовок
        for surname in surnames:
            writer.writerow([surname])

    print(f"Готово! Выгружено {len(surnames)} фамилий в файл {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
