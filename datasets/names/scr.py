import json
import csv

# Путь к исходному файлу и выходному CSV
input_file = 'names_table.jsonl'
output_file = 'names.csv'

# Чтение JSONL и запись в CSV
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    
    writer = csv.writer(outfile)
    writer.writerow(['names'])  # заголовок таблицы
    
    for line in infile:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        writer.writerow([data['text']])

print(f"Файл {output_file} успешно создан.")
