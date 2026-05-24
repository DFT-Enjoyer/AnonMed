#!/usr/bin/env python3
"""
Полный пайплайн генерации медицинских диалогов.

Создаёт два варианта датасета:
1. С сохранением символов перевода строки (\n)
2. Без \n (текст слитный)

Каждый вариант проходит:
- фильтрацию (удаление @, дубликатов, биграмм)
- очистку пунктуации (удаление лишних символов, замена сокращений)
- валидацию спанов
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
REMOVE_NL_SCRIPT = "remove_newlines.py"
FILTER_SCRIPT = "filter_duplicates.py"
CLEAN_SCRIPT = "cleaning_punctuation.py"
VALIDATE_SCRIPT = "validate_spans.py"

# Имена исходных и промежуточных файлов
RAW_JSONL = "dialogs.jsonl"
REQUIRED_INPUT_FOR_EXTRACT = "test_data_without_punct"
EXTRACT_JSONL = "subdialogs.jsonl"
EXTRACT_TXT = "subdialogs.txt"

# Финальные файлы (вариант с \n)
FINAL_WITH_NL_JSONL = "final_with_newlines.jsonl"
FINAL_WITH_NL_TXT = "final_with_newlines.txt"

# Финальные файлы (вариант без \n)
FINAL_WITHOUT_NL_JSONL = "final_without_newlines.jsonl"
FINAL_WITHOUT_NL_TXT = "final_without_newlines.txt"

# Временные файлы для каждой ветки
TEMP_FILTERED_A = "temp_filtered_with_nl.jsonl"
TEMP_NO_NL = "temp_no_newlines.jsonl"
TEMP_FILTERED_B = "temp_filtered_without_nl.jsonl"

# Список всех промежуточных файлов для удаления в конце
ALL_INTERMEDIATE = [
    RAW_JSONL, REQUIRED_INPUT_FOR_EXTRACT,
    EXTRACT_JSONL, EXTRACT_TXT,
    TEMP_FILTERED_A, TEMP_NO_NL, TEMP_FILTERED_B
]

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
    """Создаёт текстовую версию JSONL с разделителями и id."""
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

def validate_file(filepath, description):
    """
    Запускает validate_spans.py на указанном файле.
    Валидатор ищет файл с суффиксом 'final_cleaned_subdialogs',
    поэтому временно копируем целевой файл под этим именем.
    """
    target_name = "final_cleaned_subdialogs.jsonl"
    shutil.copy2(filepath, target_name)
    run_command([sys.executable, VALIDATE_SCRIPT], f"Валидация {description}")
    Path(target_name).unlink()

def process_branch(input_jsonl, final_jsonl, final_txt, branch_name):
    """
    Обрабатывает одну ветку:
    - фильтрация
    - очистка пунктуации
    - создание .txt
    - валидация
    - удаление временных фильтрованных файлов
    """
    # Фильтрация
    filtered = f"temp_{branch_name}_filtered.jsonl"
    run_command([sys.executable, FILTER_SCRIPT, input_jsonl, filtered, "--check"],
                f"Фильтрация ({branch_name})")
    ensure_file_exists(filtered, f"Фильтрация не создала {filtered}")

    # Очистка пунктуации
    run_command([sys.executable, CLEAN_SCRIPT, filtered, final_jsonl],
                f"Очистка пунктуации ({branch_name})")
    ensure_file_exists(final_jsonl, f"Очистка не создала {final_jsonl}")

    # Создание текстовой версии
    build_txt_from_jsonl(final_jsonl, final_txt)

    # Валидация
    validate_file(final_jsonl, f"{branch_name} ({final_jsonl})")

    # Удаляем временный отфильтрованный файл
    Path(filtered).unlink()

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

    # 3. Ветка A: без удаления \n
    print("\n" + "="*60)
    print("Обработка варианта С ПЕРЕНОСАМИ СТРОК (\\n)")
    print("="*60)
    process_branch(EXTRACT_JSONL, FINAL_WITH_NL_JSONL, FINAL_WITH_NL_TXT, "with_nl")

    # 4. Ветка B: с удалением \n
    print("\n" + "="*60)
    print("Обработка варианта БЕЗ ПЕРЕНОСОВ СТРОК (без \\n)")
    print("="*60)
    # Сначала удаляем \n и \r
    run_command([sys.executable, REMOVE_NL_SCRIPT, EXTRACT_JSONL, TEMP_NO_NL],
                "Удаление переводов строк")
    ensure_file_exists(TEMP_NO_NL, "remove_newlines не создал temp_no_newlines.jsonl")
    # Затем обрабатываем как обычную ветку
    process_branch(TEMP_NO_NL, FINAL_WITHOUT_NL_JSONL, FINAL_WITHOUT_NL_TXT, "without_nl")
    # Удаляем временный файл без \n
    Path(TEMP_NO_NL).unlink()

    # 5. Очистка всех промежуточных файлов
    if CLEANUP_INTERMEDIATE:
        print("\nОчистка промежуточных файлов...")
        for f in ALL_INTERMEDIATE:
            if Path(f).exists():
                Path(f).unlink()
                print(f"  Удалён {f}")

    print("\n🎉 Пайплайн успешно выполнен!")
    print(f"Файлы с переносами строк: {FINAL_WITH_NL_JSONL} и {FINAL_WITH_NL_TXT}")
    print(f"Файлы без переносов строк: {FINAL_WITHOUT_NL_JSONL} и {FINAL_WITHOUT_NL_TXT}")
    print("Отчёты валидации сохранены в папке validation_errors/")

if __name__ == "__main__":
    main()