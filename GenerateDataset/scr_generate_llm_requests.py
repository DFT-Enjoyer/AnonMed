import requests
import json
import csv
import random
import time
import re
from typing import List, Dict, Tuple, Optional

# ========== НАСТРОЙКИ ==========
API_KEY = "your_api_key"
FOLDER_ID = "b1gj1ffh6inspq494e65"
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt-5.1/latest"
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

REQUEST_DELAY = 0.5
MAX_TOKENS = 8000
TEMPERATURE = 1
PROB_SCENARIO = 0.7          # вероятность выбрать сценарий из первых 50 строк

SCENARIOS_FILE = "scenarious_from_max.txt"

# Словарь русских названий полей (для красивого вывода)
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

# Соответствие требований из сценария ключам в CSV (только для подсказок)
REQUIREMENT_MAP = {
    "ФИО": "name",
    "телефон": "phone_mobile",
    "e-mail": "email",
    "Telegram": "nicks_with_at",
    "адрес": "full_address",
    "СНИЛС": "snils",
    "полис ОМС": "oms",
    "паспорт": "passport",
    "ИНН": "inn",
    "дата рождения": "birthdate",
    "возраст": "age",
    "место работы": "full_company_name",
    "данные МСЭ": "mse",
    "водит. удостоверение": "driver_license",
    "св-во о рождении": "birth_certificate"
}

def load_first_50_scenarios(file_path: str) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """
    Загружает первые 50 строк файла, парсит каждую в (описание, список_требований).
    Если файл не найден или строк меньше 50 – возвращает пустой список.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Работаем без сценариев.")
        return []
    # Берём первые 50 строк
    first_50 = lines[:50]
    parsed = []
    for line in first_50:
        description, requirements = parse_requirements(line)
        parsed.append((description, requirements))
    return parsed

def parse_requirements(scenario_line: str) -> Tuple[str, List[Tuple[str, int]]]:
    """Разделяет строку на описание и список требований вида (название, множитель)."""
    delimiter = "Детальное анкетирование:"
    if delimiter not in scenario_line:
        return scenario_line, []
    parts = scenario_line.split(delimiter, 1)
    description = parts[0].strip()
    req_part = parts[1].strip()
    if req_part.endswith('.'):
        req_part = req_part[:-1]
    requirements = []
    for req in req_part.split(','):
        req = req.strip()
        if not req:
            continue
        match = re.search(r'[×x]\s*(\d+)', req)
        if match:
            multiplier = int(match.group(1))
            field_name = re.sub(r'\s*[×x]\s*\d+\s*$', '', req).strip()
        else:
            multiplier = 1
            field_name = req.strip()
        requirements.append((field_name, multiplier))
    return description, requirements

def call_yandexgpt(prompt: str) -> str:
    payload = {
        "modelUri": MODEL_URI,
        "completionOptions": {"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS},
        "messages": [{"role": "user", "text": prompt}]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["result"]["alternatives"][0]["message"]["text"].strip()

def generate_dialog_with_all_data(
    all_entities: List[Tuple[str, str]],
    target_word_count: int,
    scenario_description: str,
    requirements: List[Tuple[str, int]]
) -> str:
    """
    Генерирует диалог, передавая все поля CSV.
    requirements – список (название_требования, множитель) для рекомендаций.
    """
    # Формируем строку со всеми данными
    all_items_str = "\n".join([f"  - {type_}: {value}" for type_, value in all_entities])

    # Формируем рекомендации из требований (если есть)
    req_str = ""
    if requirements:
        req_lines = []
        for req_name, mult in requirements:
            csv_key = REQUIREMENT_MAP.get(req_name)
            if csv_key:
                ru_name = FIELD_NAME_RU.get(csv_key, csv_key)
                req_lines.append(f"  - {ru_name} (желательно использовать до {mult} раз)")
            else:
                req_lines.append(f"  - {req_name} (желательно использовать до {mult} раз)")
        req_str = "Старайся преимущественно использовать следующие поля:\n" + "\n".join(req_lines) + "\n\n"
    else:
        req_str = "Можешь использовать любые поля из списка ниже (старайся не более 4 различных).\n\n"

    # Пример диалога в нужном стиле (ваш оригинал)
    few_shot_example = """— здравствуйте проходите садитесь и что вас беспокоит
— здравствуйте это мой сын райгородский андрей михайлович он уже несколько дней жалуется на боль в горле и кашель
— давайте по порядку когда родился
— родился двадцать шестого декабря две тысячи восьмого года
— угу четырнадцать лет так симптомы когда начались
— да наверное около пяти дней назад сначала просто першило в горле а потом появился кашель и температура поднялась
— температура сколько была
— сначала около тридцати семи и пяти а вчера до тридцати восьми поднялась
— горло болит сильно
— да особенно при глотании и когда разговаривает сразу жалуется
— давайте посмотрим горло откройте ротик шире угу видим покраснение и небольшое воспаление миндалин давайте послушаю лёгкие дышите глубже так есть небольшой хрип но не сильный
— а это опасно
— нет не сильно опасно но нужно пропить курс антибиотиков и полоскать горло раствором фурацилина три раза в день и обильное тёплое питьё морсы чай с мёдом
— понятно а какие антибиотики вы назначите
— я выпишу рецепт на амоксициллин принимать по таблетке три раза в день после еды курс семь дней и ещё пропейте витамины для укрепления иммунитета
— хорошо а когда нам снова к вам прийти
— приходите через неделю если состояние не улучшится или будет ухудшаться сразу обращайтесь в приемное отделение
— понятно спасибо доктор теперь регистратор оформит карточку повторите данные пожалуйста ваше фио полностью
— райгородский андрей михайлович
— дата рождения
— двадцать шестого декабря две тысячи восьмого года
— адрес проживания
— ставропольский край р-н александровский с александровское ул войтика д 39
— телефон для связи
— плюс семь девятьсот девяносто четыре шестьсот восемьдесят семь двадцать шесть четыре
— е-майл
— 00123o@mail.ru
— хорошо всё записываю завтра начинайте лечение и следите за состоянием"""

    prompt = f"""Ты — генератор естественных, разговорных, детализированных медицинских диалогов на русском языке.

Сценарий: {scenario_description}

Вот пример диалога в нужном стиле (без запятых, с междометиями, как субтитры фильма):
{few_shot_example}

Твоя задача: сгенерировать диалог длиной примерно слов (до 70 слов) между участниками, указанными в сценарии (например, врач, пациент, регистратор, мама, ребёнок и т.д.). Диалог должен быть очень живым, с повторами, уточнениями, междометиями ("угу", "ага", "так", "давайте"), обращениями по имени, профессиональными медицинскими оборотами, но в разговорном стиле.

Вот все персональные данные человека (из CSV):
{all_items_str}

{req_str}
Правила:
- Данные склоняй по падежам, согласуй с предлогами.
- Не меняй фактическую информацию.
- Не дублируй ФИО дважды подряд (сделай перерыв в 1-2 реплики).
- Каждое вхождение персонального данного оберни в теги [S] и [/S].
- Выводи только текст диалога, без лишних пояснений. Не добавляй "Сценарий: ...", не пиши "Пример закончен".
- Строго не используй запятые, пиши как в примере — сплошной поток, но с интуитивными паузами.
- Если каких-то данных не хватает — можешь добавить вымышленные детали, но не меняя основные данные.
- Не используй цифры, пиши только в том формате, в котором они даны.
- Диалог должен быть небольшим, ты обязан сделать его до 100 слов.
Строго генерируй диалог в формате реплик.
Сгенерируй диалог строго в указанном стиле"""
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

    # Загружаем первые 50 расширенных сценариев
    scenarios = load_first_50_scenarios(SCENARIOS_FILE)
    if not scenarios:
        print("Не найдено ни одного сценария (первые 50 строк). Будем генерировать без сценариев.")
    else:
        print(f"Загружено {len(scenarios)} расширенных сценариев (первые 50 строк).")
    print(f"Вероятность использования сценария: {PROB_SCENARIO}")

    with open(input_csv, "r", encoding="utf-8") as csvfile, \
         open(output_txt, "w", encoding="utf-8") as txt_out, \
         open(output_jsonl, "w", encoding="utf-8") as jsonl_out:

        reader = csv.reader(csvfile)
        headers = next(reader)
        print(f"Заголовки: {headers}")

        for idx, row in enumerate(reader, start=1):
            if not row or all(cell == "" for cell in row):
                continue

            # Собираем все непустые поля в красивом виде (русское имя, значение)
            all_entities = []
            row_dict = {}
            for col_idx, value in enumerate(row):
                if value.strip():
                    field_en = headers[col_idx]
                    ru_name = FIELD_NAME_RU.get(field_en, field_en)
                    all_entities.append((ru_name, value.strip()))
                    row_dict[field_en] = value.strip()

            if not all_entities:
                print(f"Строка {idx}: нет данных, пропуск")
                continue

            target_words = random.randint(250, 500)

            # Решаем, использовать ли сценарий (только если есть загруженные сценарии)
            use_scenario = random.random() < PROB_SCENARIO and len(scenarios) > 0
            if not use_scenario:
                # Без сценария: передаём все поля, без требований
                scenario_description = "Медицинский диалог (сценарий не задан)"
                requirements = []
                scenario_str_display = "БЕЗ СЦЕНАРИЯ"
                print(f"\nСтрока {idx}: всего полей {len(all_entities)}")
                print(f"  -> сценарий: {scenario_str_display}")
                print(f"  -> целевое число слов: {target_words}")
                try:
                    marked_dialog = generate_dialog_with_all_data(
                        all_entities, target_words, scenario_description, requirements
                    )
                except Exception as e:
                    print(f"  Ошибка генерации: {e}")
                    continue
                selected_info_for_json = None  # для JSONL
            else:
                # Выбираем случайный сценарий из первых 50
                scenario_index = random.randint(0, len(scenarios)-1)
                scenario_description, requirements = scenarios[scenario_index]
                scenario_str_display = f"[СЦЕНАРИЙ] {scenario_description[:70]}..."
                print(f"\nСтрока {idx}: всего полей {len(all_entities)}")
                print(f"  -> сценарий: {scenario_str_display}")
                print(f"  -> требований: {len(requirements)}")
                print(f"  -> целевое число слов: {target_words}")
                try:
                    marked_dialog = generate_dialog_with_all_data(
                        all_entities, target_words, scenario_description, requirements
                    )
                except Exception as e:
                    print(f"  Ошибка генерации: {e}")
                    continue
                # Сохраним требования в JSONL для справки
                selected_info_for_json = [{"req_name": rn, "multiplier": mult} for rn, mult in requirements]

            clean_dialog, spans = remove_tags_and_get_spans(marked_dialog)
            real_word_count = len(clean_dialog.split())
            print(f"  -> получено слов: {real_word_count}")

            txt_out.write(f"=== Диалог {idx} (сценарий: {scenario_str_display}) ===\n{clean_dialog}\n\n---\n\n")
            record = {
                "id": idx,
                "all_entities": [{"type": t, "value": v} for t, v in all_entities],
                "scenario_requirements": selected_info_for_json,
                "target_word_count": target_words,
                "text": clean_dialog,
                "spans": spans
            }
            jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(REQUEST_DELAY)

    print(f"\nГотово! Файлы: {output_txt}, {output_jsonl}")

if __name__ == "__main__":
    main()