import requests
import json
import csv
import re
import time
import random
from typing import List, Dict, Any, Tuple

# ========== НАСТРОЙКИ ==========
API_KEY = ""                     # Замените на свой
FOLDER_ID = "b1gj1ffh6inspq494e65"          # Ваш folder_id
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt-5.1/latest"
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

REQUEST_DELAY = 0.5
MAX_TOKENS = 4000
TEMPERATURE = 0.7

FIELD_NAME_RU = {
    "full_address": "полный адрес проживания",
    "nicks_with_at": "ник в телеграме (приложение)",
    "email": "е-майл почта",
    "full_company_name": "полное наименование компании",
    "name": "ФИО (фамилия, имя, отчество)",
    "phone_mobile": "номер мобильного телефона",
    "phone_landline": "номер городского телефона",
    "snils": "номер СНИЛС",
    "passport": "номер паспорта",
    "birthdate": "дата рождения (день, месяц, год словами)",
    "inn": "ИНН",
    "oms": "номер полиса ОМС",
    "age": "возраст",
    "mse": "номер справки МСЭ",
    "birth_certificate": "номер свидетельства о рождении",
    "driver_license": "номер водительского удостоверения"
}

def call_yandexgpt(prompt: str) -> str:
    payload = {
        "modelUri": MODEL_URI,
        "completionOptions": {"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS},
        "messages": [{"role": "user", "text": prompt}]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["result"]["alternatives"][0]["message"]["text"].strip()

def generate_dialog_with_tags(selected_items: List[Tuple[str, str]], target_word_count: int) -> str:
    items_str = "\n".join([f"  - {type_}: {value}" for type_, value in selected_items])
    prompt = f"""Ты — генератор медицинских диалогов. Создай реалистичный разговор на тему здравоохранения (приём у врача, регистратура, страховой случай, вызов скорой) между двумя людьми. 

Диалог должен содержать примерно {target_word_count} слов. Формат: "A: ...\\nB: ...\\nA: ...".

В диалоге обязательно нужно использовать следующие персональные данные и реквизиты (каждый хотя бы один раз). Они должны фигурировать именно в том качестве, которое указано:

{items_str}

Делай предложения максимально естественными.

 Ты должен менять слова грамматически (склонять по падежам, добавлять предлоги, менять окончания) так, чтобы предложения стали корректными с точки зрения грамматики. Строго запрещено менять фактическую информацию.
 Нельзя преобразовывать буквенную запись чисел в численную. Обязательно меняй форму числительных для корректного диалога.
 Каждое вхождение значения должно быть обёрнуто в теги [S] и [/S] вместе с грамматическими изменениями. Генерируй максимально правдоподобные тексты.

Например:
 "Дата рождения: [S]шестого января одна тысяча девятьсот сорок девятого года[/S]"
Выведи только текст диалога с тегами, без лишних пояснений."""
    return call_yandexgpt(prompt)

def remove_tags_and_get_spans(marked_text: str) -> Tuple[str, List[Dict]]:
    clean_chars = []
    spans = []
    i = 0
    n = len(marked_text)
    inside_span = False
    current_span_text = []
    span_start_pos = -1

    while i < n:
        if marked_text.startswith('[S]', i):
            i += 3
            inside_span = True
            current_span_text = []
            span_start_pos = len(clean_chars)
            continue
        if marked_text.startswith('[/S]', i):
            i += 4
            if inside_span:
                span_text = ''.join(current_span_text)
                spans.append({
                    "span1": span_text,
                    "begin": span_start_pos,
                    "end": len(clean_chars)
                })
                inside_span = False
            continue
        ch = marked_text[i]
        if inside_span:
            current_span_text.append(ch)
        clean_chars.append(ch)
        i += 1

    if inside_span and current_span_text:
        spans.append({
            "span1": ''.join(current_span_text),
            "begin": span_start_pos,
            "end": len(clean_chars)
        })

    return ''.join(clean_chars), spans

def main():
    input_csv = "combined_by_columns.csv"
    output_txt = "dialogs.txt"
    output_jsonl = "dialogs.jsonl"

    with open(input_csv, "r", encoding="utf-8") as csvfile, \
         open(output_txt, "w", encoding="utf-8") as txt_out, \
         open(output_jsonl, "w", encoding="utf-8") as jsonl_out:

        reader = csv.reader(csvfile)
        headers = next(reader)
        print(f"Заголовки: {headers}")

        for idx, row in enumerate(reader, start=1):
            if not row or all(cell == "" for cell in row):
                continue

            available = []
            for col_idx, value in enumerate(row):
                if value.strip():
                    field_en = headers[col_idx]
                    field_ru = FIELD_NAME_RU.get(field_en, field_en)
                    available.append((field_ru, value.strip()))

            if not available:
                print(f"Строка {idx}: нет данных, пропуск")
                continue

            k = random.randint(1, 3)
            selected = random.sample(available, k)
            print(f"\nСтрока {idx}: выбрано {k} сущностей")
            for t, v in selected:
                print(f"  - {t}: {v[:60]}...")

            target_words = random.randint(2 * k, 8 * k)
            print(f"  -> целевое число слов: {target_words}")

            try:
                marked_dialog = generate_dialog_with_tags(selected, target_words)
            except Exception as e:
                print(f"  Ошибка: {e}")
                continue

            clean_dialog, spans = remove_tags_and_get_spans(marked_dialog)
            real_word_count = len(clean_dialog.split())
            print(f"  -> получено слов: {real_word_count}")

            txt_out.write(f"=== Диалог {idx} ===\n{clean_dialog}\n\n---\n\n")
            record = {
                "id": idx,
                "selected_entities": [{"type": t, "value": v} for t, v in selected],
                "target_word_count": target_words,
                "text": clean_dialog,
                "spans": spans
            }
            jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(REQUEST_DELAY)

    print(f"\nГотово! Файлы: {output_txt}, {output_jsonl}")

if __name__ == "__main__":
    main()