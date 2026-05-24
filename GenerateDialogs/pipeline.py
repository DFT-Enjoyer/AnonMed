#!/usr/bin/env python3
"""
Полный пайплайн с валидацией:
1. scr_generate_llm_requests.py -> dialogs.jsonl
2. extract_subdialogs.py -> subdialogs.jsonl / subdialogs.txt
3. filter_duplicates.py -> filtered_subdialogs.jsonl
4. cleaning_punctuation.py -> final_cleaned_subdialogs.jsonl
5. validate_spans.py (проверка финальных файлов)
"""

import subprocess
import sys
import shutil
import json
from pathlib import Path

# -------------------------------
# НАСТРОЙКИ
# -------------------------------
GENERATE_SCRIPT = "scr_generate_llm_requests.py"
EXTRACT_SCRIPT = "extract_subdialogs.py"
FILTER_SCRIPT = "filter_duplicates.py"
CLEAN_SCRIPT = "cleaning_punctuation.py"
VALIDATE_SCRIPT = "validate_spans.py"   # адаптированная версия

# Имена файлов
RAW_JSONL = "dialogs.jsonl"
REQUIRED_INPUT_FOR_EXTRACT = "test_data_without_punct"
EXTRACT_JSONL = "subdialogs.jsonl"
EXTRACT_TXT = "subdialogs.txt"
FILTERED_JSONL = "filtered_subdialogs.jsonl"
CLEANED_JSONL = "final_cleaned_subdialogs.jsonl"
FINAL_TXT = "final_cleaned_subdialogs.txt"

INTERMEDIATE_FILES = [RAW_JSONL, REQUIRED_INPUT_FOR_EXTRACT, EXTRACT_JSONL, EXTRACT_TXT, FILTERED_JSONL, "dialogs.txt"]
CLEANUP_INTERMEDIATE = True

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
    run_command([sys.executable, GENERATE_SCRIPT], "Генерация диалогов YandexGPT")
    ensure_file_exists(RAW_JSONL, "Скрипт генерации не создал dialogs.jsonl")

    # 2. Extract
    shutil.copy2(RAW_JSONL, REQUIRED_INPUT_FOR_EXTRACT)
    print(f"   Скопировано {RAW_JSONL} -> {REQUIRED_INPUT_FOR_EXTRACT}")
    run_command([sys.executable, EXTRACT_SCRIPT], "Extract: нарезка поддиалогов 20–70 слов")
    ensure_file_exists(EXTRACT_JSONL, "Extract не создал subdialogs.jsonl")
    ensure_file_exists(EXTRACT_TXT, "Extract не создал subdialogs.txt")

    # 3. Фильтрация
    run_command([sys.executable, FILTER_SCRIPT, EXTRACT_JSONL, FILTERED_JSONL, "--check"],
                "Фильтрация дубликатов и запрещённых конструкций")
    ensure_file_exists(FILTERED_JSONL, "Фильтрация не создала filtered_subdialogs.jsonl")

    # 4. Очистка пунктуации
    run_command([sys.executable, CLEAN_SCRIPT, FILTERED_JSONL, CLEANED_JSONL],
                "Очистка пунктуации, приведение регистра, замена сокращений")
    ensure_file_exists(CLEANED_JSONL, "Очистка пунктуации не создала финальный JSONL")

    # 5. Создание текстовой версии финальных данных
    build_txt_from_jsonl(CLEANED_JSONL, FINAL_TXT)

    # 6. Валидация спанов
    run_command([sys.executable, VALIDATE_SCRIPT], "Валидация спанов в финальных файлах")
    # Валидатор сам выведет результат, но если он вернёт ненулевой код, оркестратор остановится.
    # Если хотим игнорировать ошибки валидации, можно добавить проверку, но лучше остановиться.

    # 7. Очистка промежуточных файлов
    if CLEANUP_INTERMEDIATE:
        print("\nОчистка промежуточных файлов...")
        for f in INTERMEDIATE_FILES:
            if Path(f).exists():
                Path(f).unlink()
                print(f"  Удалён {f}")

    print("\n Пайплайн успешно выполнен!")
    print(f"Итоговые файлы: {CLEANED_JSONL}, {FINAL_TXT}")
    print(f"Отчёт валидации сохранён в папке {Path(VALIDATE_SCRIPT).parent / 'validation_errors'}")

if __name__ == "__main__":
    main()
