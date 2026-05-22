import requests
import json
import csv
import random
import time
from typing import List, Dict, Tuple

# ========== НАСТРОЙКИ ==========
API_KEY = ""
FOLDER_ID = "b1gj1ffh6inspq494e65"
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt-5.1/latest"
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

REQUEST_DELAY = 0.5
MAX_TOKENS = 4000
TEMPERATURE = 1  # повысил для разнообразия

SCENARIOS_FILE = "scenarios.txt"

FIELD_NAME_RU = {
    "full_address": "полный адрес проживания",
    "nicks_with_at": "ник в телеграме",
    "email": "е-майл почта",
    "full_company_name": "полное наименование компании",
    "name": "ФИО",
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

FIELD_WEIGHTS = {
    "nicks_with_at": 0.2,
    "email": 0.3,
    "phone_mobile": 0.8,
    "phone_landline": 0.2,
    "snils": 1.0,
    "passport": 1.0,
    "birthdate": 0.8,
    "oms": 0.9,
    "inn": 0.7,
    "age": 1.0,
    "mse": 0.4,
    "birth_certificate": 0.2,
    "driver_license": 0.2,
    "full_address": 1.0,
    "name": 1.0,
    "full_company_name": 0.3
}
DEFAULT_WEIGHT = 1.0

def load_scenarios(file_path: str) -> List[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            scenarios = [line.strip() for line in f if line.strip()]
        if not scenarios:
            print("Предупреждение: файл сценариев пуст, использую стандартные.")
            return [
                "Вручение военного билета (категория «В» — ограниченно годен)",
                "Вызов врача на дом для умирающего (Запись через регистратора)",
                "Разговор в реанимации (Врач-реаниматолог и сын пациента после инсульта)",
                "Приём у терапевта с жалобами на давление",
                "Оформление полиса ОМС в регистратуре"
            ]
        return scenarios
    except FileNotFoundError:
        print(f"Файл {file_path} не найден, использую стандартные сценарии.")
        return [
            "Вручение военного билета (категория «В» — ограниченно годен)",
            "Вызов врача на дом для умирающего (Запись через регистратора)",
            "Разговор в реанимации (Врач-реаниматолог и сын пациента после инсульта)"
        ]

def select_entities(row: List[str], headers: List[str]) -> List[Tuple[str, str]]:
    available = []
    for col_idx, value in enumerate(row):
        if value.strip():
            field_en = headers[col_idx]
            field_ru = FIELD_NAME_RU.get(field_en, field_en)
            weight = FIELD_WEIGHTS.get(field_en, DEFAULT_WEIGHT)
            available.append((field_en, field_ru, value.strip(), weight))
    if not available:
        return []
    k = random.choice([1, 2])
    k = min(k, len(available))
    selected = []
    remaining = available[:]
    for _ in range(k):
        if not remaining:
            break
        weighted_items = [(i, item[3]) for i, item in enumerate(remaining)]
        total = sum(w for _, w in weighted_items)
        r = random.uniform(0, total)
        acc = 0.0
        chosen_idx = None
        for idx, w in weighted_items:
            acc += w
            if r <= acc:
                chosen_idx = idx
                break
        if chosen_idx is None:
            chosen_idx = 0
        chosen = remaining.pop(chosen_idx)
        selected.append((chosen[1], chosen[2]))
    return selected

def call_yandexgpt(prompt: str) -> str:
    payload = {
        "modelUri": MODEL_URI,
        "completionOptions": {"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS},
        "messages": [{"role": "user", "text": prompt}]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["result"]["alternatives"][0]["message"]["text"].strip()

def generate_dialog_with_tags(selected_items: List[Tuple[str, str]], target_word_count: int, scenario: str) -> str:
    items_str = "\n".join([f"  - {type_}: {value}" for type_, value in selected_items])
    prompt = f"""Ты — генератор разнообразного медицинского диалога на тему: {scenario}

Твоя задача:
1. Сгенерируй короткий диалог (примерно {target_word_count} слов, от 20 до 50). Диалог должен быть максимально естественным, на русском языке, соответствовать медицинской тематике. Если с заданном сценарием получается неестественно, можешь его изменить.
Диалог должен быть формата [контекст] [персональные данные] [контекст], но можешь при необходимости поменять его структуру.
2. В диалоге обязательно используй ВСЕ следующие персональные данные (каждое хотя бы один раз): {items_str}. Поменяй форму данных согласно правилам языка так, чтобы фактическая информация не  изменилась. Если у тебя есть буквенный формат чисел, то ты можешь преобразовать их в цифровой формат. Строго следи за тем, чтобы количество цифр не изменилось.
Строго следи за падежами, склонениями и прочей грамматической информацией для генерации правдоподобного диалога.
3. Каждое вхождение персональных данных (вместе с грамматическими изменениями) оберни в теги [S] и [/S].
4. Выводи только текст диалога в формате:
   А: реплика
   Б: реплика
   ...
   Не добавляй пояснений, не пиши "Сценарий: ...". Только диалог.
5. Строго следи за окончаниями вставленных слов (персоналных данных).
6. Не добавляй новую информацию в span, если нет какой - то информации. Если тебе для диалога нужны какие - то данные, а их нет в сценарии, то поменяй диалог.
7. Следи за родом, если, например, ФИО женское, то не сочетай его с вручением военного билета. Следи за этим, то есть нужно сочетать персональные данные и сценарий.


Пример:
А: Ваш полис ОМС — [S]три тысячи пятьсот два сто одиннадцать две тысячи семьсот шестьдесят восемь семь тысяч шестьсот сорок[/S]?
Б: Да, и дата рождения [S]шестое января одна тысяча девятьсот сорок девятого года[/S].
Ты должен не менять цифровые данные при их преобразовании. Также если у тебя появляется ФИО, то ты не должен его дублировать:
А: Здравствуйте, меня зовут Карлюга Ирина Евгеньевна Карлюга Ирина Евгеньевна. Так строго запрещено, нужно по одному разу указывать ФИО.
Аналогично запрещено это:
А: +7 994 687-26-4. Так нельзя. Нужно 14 символов.

Сгенерируй диалог."""
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
                    "span": span_text,
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
            "span": ''.join(current_span_text),
            "begin": span_start_pos,
            "end": len(clean_chars)
        })

    return ''.join(clean_chars), spans

def main():
    input_csv = "combined_by_columns.csv"
    output_txt = "dialogs.txt"
    output_jsonl = "dialogs.jsonl"
    scenarios = load_scenarios(SCENARIOS_FILE)
    print(f"Загружено сценариев: {len(scenarios)}")

    with open(input_csv, "r", encoding="utf-8") as csvfile, \
         open(output_txt, "w", encoding="utf-8") as txt_out, \
         open(output_jsonl, "w", encoding="utf-8") as jsonl_out:

        reader = csv.reader(csvfile)
        headers = next(reader)
        print(f"Заголовки: {headers}")

        for idx, row in enumerate(reader, start=1):
            if not row or all(cell == "" for cell in row):
                continue

            selected = select_entities(row, headers)
            if not selected:
                print(f"Строка {idx}: нет данных, пропуск")
                continue

            chosen_scenario = random.choice(scenarios)
            print(f"\nСтрока {idx}: выбрано {len(selected)} сущностей")
            for t, v in selected:
                print(f"  - {t}: {v[:60]}...")
            print(f"  -> сценарий: {chosen_scenario[:80]}...")

            target_words = random.randint(20, 50)
            if len(selected) == 3:
                target_words = min(50, target_words + 5)
            print(f"  -> целевое число слов: {target_words}")

            try:
                marked_dialog = generate_dialog_with_tags(selected, target_words, chosen_scenario)
            except Exception as e:
                print(f"  Ошибка генерации: {e}")
                continue

            clean_dialog, spans = remove_tags_and_get_spans(marked_dialog)
            real_word_count = len(clean_dialog.split())
            print(f"  -> получено слов: {real_word_count}")

            txt_out.write(f"=== Диалог {idx} (сценарий: {chosen_scenario}) ===\n{clean_dialog}\n\n---\n\n")
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
