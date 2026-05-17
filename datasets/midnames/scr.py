#!/usr/bin/env python3
import json
import csv

INPUT_FILE = "midnames_table.jsonl"
OUTPUT_FILE = "midnames.csv"

def main():
    midnames = []
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj:
                    midnames.append(obj["text"])
                else:
                    print(f"Предупреждение: строка {line_num} не содержит поля 'text'")
            except json.JSONDecodeError as e:
                print(f"Ошибка в строке {line_num}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["midnames"])   # заголовок
        for name in midnames:
            writer.writerow([name])

    print(f"Готово! Выгружено {len(midnames)} отчеств в файл {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
