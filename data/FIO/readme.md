Для генерации новой пачки имен достаточно использовать три файла с вероятностями с таким кодом:

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
from collections import defaultdict
from typing import List, Dict, Tuple

def load_with_gender(filename: str):
    """
    Загружает CSV с колонками: text, probability, gender.
    Возвращает словарь: {'m': (items_list, probs_list), 'f': (items_list, probs_list)}
    Вероятности внутри каждого пола нормализуются.
    Также возвращает общий список всех имён с исходными вероятностями для выбора пола.
    """
    data_by_gender = defaultdict(lambda: ([], []))  # gender -> (items, probs)
    all_items = []      # все элементы (для выбора пола)
    all_probs = []      # исходные вероятности (не нормализованные)

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # пропускаем заголовок
        for row in reader:
            if len(row) >= 3:
                text = row[0].strip()
                prob = float(row[1])
                gender = row[2].strip().lower()
                if gender not in ('m', 'f'):
                    continue
                # для общего списка
                all_items.append(text)
                all_probs.append(prob)
                # для гендерных списков
                items, probs = data_by_gender[gender]
                items.append(text)
                probs.append(prob)

    # Нормализация вероятностей внутри каждого пола
    normalized_by_gender = {}
    for gender, (items, probs) in data_by_gender.items():
        total = sum(probs)
        if total > 0:
            norm_probs = [p / total for p in probs]
        else:
            norm_probs = [1.0/len(items)] if items else []
        normalized_by_gender[gender] = (items, norm_probs)

    # Нормализация общих вероятностей (для выбора пола через имя)
    total_all = sum(all_probs)
    if total_all > 0:
        all_probs_norm = [p / total_all for p in all_probs]
    else:
        all_probs_norm = [1.0/len(all_items)] if all_items else []

    return normalized_by_gender, (all_items, all_probs_norm)

def generate_unique_fio_with_gender(
    names_data: Tuple[Dict, Tuple[List, List]],
    surnames_data: Tuple[Dict, Tuple[List, List]],
    midnames_data: Tuple[Dict, Tuple[List, List]],
    count: int = 10000
) -> List[str]:
    """
    Генерирует уникальные полные ФИО (строки через пробел).
    Сначала выбирается имя из общего распределения, определяется пол,
    затем фамилия и отчество из соответствующих гендерных списков.
    """
    names_by_gender, (all_names, all_name_probs) = names_data
    surnames_by_gender, _ = surnames_data  # общий список нам не нужен
    midnames_by_gender, _ = midnames_data

    # Проверка наличия данных для каждого пола
    for gender in ('m', 'f'):
        if gender not in names_by_gender or not names_by_gender[gender][0]:
            raise ValueError(f"Нет имён для пола {gender}")
        if gender not in surnames_by_gender or not surnames_by_gender[gender][0]:
            raise ValueError(f"Нет фамилий для пола {gender}")
        if gender not in midnames_by_gender or not midnames_by_gender[gender][0]:
            raise ValueError(f"Нет отчеств для пола {gender}")

    unique_fios = set()
    max_combinations = len(all_names) * sum(len(lst) for lst, _ in surnames_by_gender.values()) * sum(len(lst) for lst, _ in midnames_by_gender.values())
    if max_combinations < count:
        # Оценочно, но точнее проверить невозможно, просто предупредим
        print(f"Предупреждение: максимальное число возможных комбинаций может быть меньше {count}.")

    while len(unique_fios) < count:
        # 1. Выбираем имя из общего распределения (определяем пол)
        name = random.choices(all_names, weights=all_name_probs, k=1)[0]
        # Находим пол этого имени (нужно по данным, как определить? ищем в names_by_gender)
        gender = None
        for g, (items, _) in names_by_gender.items():
            if name in items:
                gender = g
                break
        if gender is None:
            continue  # на всякий случай, если имя не найдено в гендерных списках

        # 2. Выбираем фамилию соответствующего пола
        surnames_list, surnames_probs = surnames_by_gender[gender]
        surname = random.choices(surnames_list, weights=surnames_probs, k=1)[0]

        # 3. Выбираем отчество соответствующего пола
        midnames_list, midnames_probs = midnames_by_gender[gender]
        midname = random.choices(midnames_list, weights=midnames_probs, k=1)[0]

        # Формируем полное ФИО
        full_name = f"{surname} {name} {midname}"
        unique_fios.add(full_name)

    return list(unique_fios)

def save_fio_to_csv(fio_list: List[str], output_file: str, header: bool = True):
    """Сохраняет список ФИО в CSV, одна колонка."""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(['full_name'])
        for fio in fio_list:
            writer.writerow([fio])

def main():
    # Имена файлов (предполагается, что лежат в текущей папке)
    name_file = "names_probability.csv"
    surname_file = "surnames_probability.csv"
    midname_file = "midnames_probability.csv"

    print("Загрузка данных с учётом пола...")
    names_data = load_with_gender(name_file)
    surnames_data = load_with_gender(surname_file)
    midnames_data = load_with_gender(midname_file)

    # Статистика по загруженным данным
    for label, data in [("Имена", names_data), ("Фамилии", surnames_data), ("Отчества", midnames_data)]:
        by_gender, _ = data
        print(f"{label}: мужских={len(by_gender.get('m', ([],[]))[0])}, женских={len(by_gender.get('f', ([],[]))[0])}")

    print("Генерация 10000 уникальных ФИО с согласованием по полу...")
    fio_list = generate_unique_fio_with_gender(
        names_data, surnames_data, midnames_data,
        count=10000
    )

    output_file = "generated_fio.csv"
    save_fio_to_csv(fio_list, output_file, header=True)
    print(f"Готово! Сгенерировано {len(fio_list)} уникальных ФИО. Результат в {output_file}")

if __name__ == "__main__":
    main()



Структура папок (тут есть лишнее):

fedor@fedor-Aspire-A315-44P:~/Project$ ls
 FIO                        resume
 generated_fio.csv          scr_for_generating_FIO
 midnames                   scr_for_probability
 midnames_probability.csv   scr.py
 names                      surname
 names_probability.csv      surnames_probability.csv
 readme.md                 'Не подтвержден 433062.crdownload'
fedor@fedor-Aspire-A315-44P:~/Project$ 


Качает в generated_fio.csv из скрипта scr_for_generating_FIO используя файлы с probability

