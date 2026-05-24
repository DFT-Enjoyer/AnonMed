#!/usr/bin/env python3
"""
Полный пайплайн генерации и обработки медицинских диалогов.

Этапы:
1. Генерация диалогов через YandexGPT -> dialogs.jsonl
2. Extract (нарезка поддиалогов 20-70 слов, 1-2 спана) -> subdialogs.jsonl
3. Удаление символов перевода строки \n и \r -> no_newlines.jsonl
4. Фильтрация (удаление @, дубликатов, биграмм) -> filtered_no_newlines.jsonl
5. Очистка пунктуации и замена сокращений -> final_cleaned.jsonl
6. Валидация спанов -> вывод отчёта в папку validation_errors/
"""

import subprocess
import sys
import shutil
import json
from pathlib import Path

# -------------------------------
# НАСТРОЙКИ (можно редактировать)
# -------------------------------
GENERATE_SCRIPT = "scr_generate_llm_requests.py"
EXTRACT_SCRIPT = "extract_subdialogs.py"
REMOVE_NL_SCRIPT = "remove_newlines.py"
FILTER_SCRIPT = "filter_duplicates.py"
CLEAN_SCRIPT = "cleaning_punctuation.py"
VALIDATE_SCRIPT = "validate_spans.py"

# Имена промежуточных и итоговых файлов
RAW_JSONL = "dialogs.jsonl"
REQUIRED_INPUT_FOR_EXTRACT = "test_data_without_punct"
EXTRACT_JSONL = "subdialogs.jsonl"
EXTRACT_TXT = "subdialogs.txt"
NO_NL_JSONL = "no_newlines.jsonl"
FILTERED_JSONL = "filtered_subdialogs.jsonl"
CLEANED_JSONL = "final_cleaned_subdialogs.jsonl"
FINAL_TXT = "final_cleaned_subdialogs.txt"

# Список файлов, которые будут удалены после успешного выполнения (очистка)
INTERMEDIATE_FILES = [
    RAW_JSONL, REQUIRED_INPUT_FOR_EXTRACT,
    EXTRACT_JSONL, EXTRACT_TXT, NO_NL_JSONL, FILTERED_JSONL
]
CLEANUP_INTERMEDIATE = True  # удалять ли промежуточные файлы

# -------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------------
def run_command(cmd, description):
    print(f"\n>>> {description}")
    print(f"    Выполняется: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f" Ошибка при выполнении: {description}")
        sys.exit(result.returncode)
    print(f" {description} завершён успешно.")

def ensure_file_exists(filepath, description):
    if not Path(filepath).exists():
        print(f" Ожидаемый файл {filepath} не найден. {description}")
        sys.exit(1)

def build_txt_from_jsonl(jsonl_path, txt_path):
    """Создаёт текстовую версию JSONL (с разделителями и id)."""
    with open(jsonl_path, 'r', encoding='utf-8') as fj, \
         open(txt_path, 'w', encoding='utf-8') as ft:
        first = True
        for line in fj:
            obj = json.loads(line)
            if not first:
                ft.write('\n=====\n')
            first = False
            ft.write(f"[id: {obj.get('id', 'unknown')}]\n")
            ft.write(obj.get('text', ''))
    print(f"   Создан {txt_path} из {jsonl_path}")

# -------------------------------
# ОСНОВНАЯ ЛОГИКА
# -------------------------------
def main():
    # 1. Генерация
    run_command([sys.executable, GENERATE_SCRIPT], "1. Генерация диалогов YandexGPT")
    ensure_file_exists(RAW_JSONL, "Скрипт генерации не создал dialogs.jsonl")

    # 2. Extract (нарезка поддиалогов)
    shutil.copy2(RAW_JSONL, REQUIRED_INPUT_FOR_EXTRACT)
    print(f"   Скопировано {RAW_JSONL} -> {REQUIRED_INPUT_FOR_EXTRACT}")
    run_command([sys.executable, EXTRACT_SCRIPT], "2. Extract: нарезка поддиалогов 20–70 слов")
    ensure_file_exists(EXTRACT_JSONL, "Extract не создал subdialogs.jsonl")
    ensure_file_exists(EXTRACT_TXT, "Extract не создал subdialogs.txt")

    # 3. Удаление \n и \r, корректировка спанов
    run_command([sys.executable, REMOVE_NL_SCRIPT, EXTRACT_JSONL, NO_NL_JSONL],
                "3. Удаление переводов строк и корректировка спанов")
    ensure_file_exists(NO_NL_JSONL, "remove_newlines не создал no_newlines.jsonl")

    # 4. Фильтрация (удаление @, дубликатов, биграмм)
    run_command([sys.executable, FILTER_SCRIPT, NO_NL_JSONL, FILTERED_JSONL, "--check"],
                "4. Фильтрация дубликатов и запрещённых конструкций")
    ensure_file_exists(FILTERED_JSONL, "Фильтрация не создала filtered_subdialogs.jsonl")

    # 5. Очистка пунктуации и замена сокращений (опционально)
    run_command([sys.executable, CLEAN_SCRIPT, FILTERED_JSONL, CLEANED_JSONL],
                "5. Очистка пунктуации, приведение регистра, замена сокращений")
    ensure_file_exists(CLEANED_JSONL, "Очистка пунктуации не создала финальный JSONL")

    # 6. Создание текстовой версии финальных данных
    build_txt_from_jsonl(CLEANED_JSONL, FINAL_TXT)

    # 7. Валидация спанов
    run_command([sys.executable, VALIDATE_SCRIPT], "6. Валидация спанов в финальных файлах")

    # 8. Очистка промежуточных файлов
    if CLEANUP_INTERMEDIATE:
        print("\nОчистка промежуточных файлов...")
        for f in INTERMEDIATE_FILES:
            if Path(f).exists():
                Path(f).unlink()
                print(f"  Удалён {f}")

    print("\n Пайплайн успешно выполнен!")
    print(f"Итоговые файлы: {CLEANED_JSONL} и {FINAL_TXT}")
    print(f"Отчёт валидации сохранён в папке validation_errors/")

if __name__ == "__main__":
    main()