import requests
import json
import csv
import random
import time
import re
from typing import List, Dict, Tuple, Optional

# ========== НАСТРОЙКИ ==========
API_KEY = "Your_API_key"
FOLDER_ID = "your_folder_id"
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt-5.1/latest"
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

REQUEST_DELAY = 0.5
MAX_TOKENS = 8000
TEMPERATURE = 1
PROB_SCENARIO = 0.7

SCENARIOS_FILE = "scenarios.txt"

# Максимальное количество запросов к LLM. Установите нужное число (например, 5, 10, 100).
# Если поставить None, ограничение будет снято (обработаются все строки CSV).
MAX_REQUESTS = "your_number_of_dialogs"

# Словарь русских названий полей (для красивого вывода)
FIELD_NAME_RU = {
    "full_address": "полный адрес проживания",
    "telegram_nicks": "ник в телеграме",
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

# Обратный маппинг: русское название -> английский ключ (на случай, если модель ошибётся)
RU_TO_EN = {ru: en for en, ru in FIELD_NAME_RU.items()}

REQUIREMENT_MAP = {
    "ФИО": "name",
    "телефон": "phone_mobile",
    "e-mail": "email",
    "Telegram": "telegram_nicks",
    "адрес": "full_address",
    "снилс": "snils",
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

def load_all_scenarios(file_path: str) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """Загружает ВСЕ строки файла, каждая строка -> (описание, список_требований)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Работаем без сценариев.")
        return []
    parsed = []
    for line in lines:
        description, requirements = parse_requirements(line)
        parsed.append((description, requirements))
    return parsed


def parse_requirements(scenario_line: str) -> Tuple[str, List[Tuple[str, int]]]:
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
    all_items_str = "\n".join([f"  - {type_}: {value}" for type_, value in all_entities])

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

    few_shot_example = """День добрый проходите садитесь и что вас беспокоит
— Добрый день это мой сын райгородский андрей михайлович он уже несколько дней жалуется на боль в горле и кашель
— давайте по порядку когда родился
— родился [S:birthdate]26.12.2008[/S]
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
— [S:name]райгородский андрей михайлович[/S]
— дата рождения
— [S:birthdate]26.12.2008[/S]
— адрес проживания
— [S:full_address]ставропольский край р-н александровский с александровское ул войтика д 39[/S]
— телефон для связи
— [S:phone_mobile]+79946872641[/S]
— е майл
— [S:email]п точка гурева собака тотмахлеб точка ру[/S]
— хорошо всё записываю завтра начинайте лечение и следите за состоянием"""

    prompt = f"""

Ты — генератор меддиалогов. Сначала составь план, затем диалог.

## ОБЯЗАТЕЛЬНЫЙ ЧЕК-ЛИСТ (каждый пункт проверить):
1. [S:ключ]значение[/S] для каждого персонального данного
2. Один и тот же ключ не повторяется в двух соседних репликах
3. Возможные ключи: (full_address,telegram_nicks,email,name,phone_mobile,phone_landline,snils,passport,birthdate,inn,oms,age,mse,birth_certificate,driver_license,full_company_name)
4. Даты только числами: ДД.ММ.ГГГГ
5. Email точно как в данных, без изменений.
6. Telegram точно как в данных
7. Все слова должны быть на русском. Email и telegram меняй на емэйл и телеграм соответственно.
8. Диалоги должны быть маскимально естественными, обязательно с междометиями. Все должно быть как в реальном диалоге. Не должно быть пунктуации. Пиши одним потоком, как в примере:
{few_shot_example}


## Правильный формат:
- Дата: "14.01.1993" (не "четырнадцатое января")
- Номер телефона: "79534954787" (не "семь девять пять три четыре девять пять сорок семь восемь семь")
- С тегом: [S:birthdate]14.01.1993[/S], [S:phone_mobile]79534954787[/S]

## Сценарий: {scenario_description}
## Данные:
{all_items_str}

Твои действия:
Напиши диалог на ~{target_word_count} слов. Каждая реплика с новой строки, тире в начале.
"""
    return call_yandexgpt(prompt)

def remove_tags_and_get_spans(marked_text: str) -> Tuple[str, List[Dict]]:
    """
    Извлекает теги вида [S:label]...[/S] и [S]...[/S] (без метки).
    Корректно вычисляет позиции begin/end в чистом тексте.
    Возвращает (чистый_текст, список_spans).
    """
    clean_chars = []
    spans = []
    i = 0
    n = len(marked_text)
    inside_span = False
    span_start_pos = -1
    current_label = None
    span_data_chars = []  # накапливаем содержимое span для поля data

    while i < n:
        # Открывающий тег с меткой [S:...]
        if marked_text.startswith('[S:', i):
            end_bracket = marked_text.find(']', i)
            if end_bracket != -1:
                label = marked_text[i+3:end_bracket]
                i = end_bracket + 1
                inside_span = True
                span_start_pos = len(clean_chars)  # позиция начала span в чистом тексте
                current_label = label
                span_data_chars = []
                continue
        # Открывающий тег без метки [S]
        elif marked_text.startswith('[S]', i):
            i += 3
            inside_span = True
            span_start_pos = len(clean_chars)
            current_label = None
            span_data_chars = []
            continue

        # Закрывающий тег [/S]
        if marked_text.startswith('[/S]', i):
            i += 4
            if inside_span:
                span_data = ''.join(span_data_chars)
                label = current_label if current_label is not None else "unknown"
                spans.append({
                    "begin": span_start_pos,
                    "end": len(clean_chars),  # текущая длина clean_chars после добавления всех символов
                    "label": label,
                    "data": span_data
                })
                inside_span = False
                current_label = None
            continue

        # Обычный символ
        ch = marked_text[i]
        clean_chars.append(ch)
        if inside_span:
            span_data_chars.append(ch)
        i += 1

    # Если остался незакрытый span (на случай обрыва)
    if inside_span:
        span_data = ''.join(span_data_chars)
        label = current_label if current_label is not None else "unknown"
        spans.append({
            "begin": span_start_pos,
            "end": len(clean_chars),
            "label": label,
            "data": span_data
        })

    return ''.join(clean_chars), spans

def main():
    max_requests = MAX_REQUESTS

    input_csv = "DataPI.csv"
    output_txt = "dialogs.txt"
    output_jsonl = "dialogs.jsonl"

    scenarios = load_all_scenarios(SCENARIOS_FILE)
    if not scenarios:
        print("Не найдено ни одного сценария (первые 50 строк). Будем генерировать без сценариев.")
    else:
        print(f"Загружено {len(scenarios)} расширенных сценариев (первые 50 строк).")
    print(f"Вероятность использования сценария: {PROB_SCENARIO}")
    if max_requests:
        print(f"Максимальное количество запросов к LLM: {max_requests}")
    else:
        print("Максимальное количество запросов к LLM: без ограничений")

    with open(input_csv, "r", encoding="utf-8") as csvfile, \
         open(output_txt, "w", encoding="utf-8") as txt_out, \
         open(output_jsonl, "w", encoding="utf-8") as jsonl_out:

        reader = csv.reader(csvfile)
        headers = next(reader)
        print(f"Заголовки: {headers}")

        successful_requests = 0

        for idx, row in enumerate(reader, start=1):
            if max_requests is not None and successful_requests >= max_requests:
                print(f"\nДостигнуто максимальное количество запросов ({max_requests}). Остановка.")
                break

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

            target_words = random.randint(50, 250)

            use_scenario = random.random() < PROB_SCENARIO and len(scenarios) > 0
            if not use_scenario:
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
            else:
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

            # Проверка на наличие тегов (хотя бы один)
            if not re.search(r'\[S:?\w*\]', marked_dialog):
                print("  -> ВНИМАНИЕ: теги не найдены, диалог пропущен")
                continue

            clean_dialog, spans = remove_tags_and_get_spans(marked_dialog)
            real_word_count = len(clean_dialog.split())
            print(f"  -> получено слов: {real_word_count}, тегов найдено: {len(spans)}")

            txt_out.write(f"=== Диалог {idx} (сценарий: {scenario_str_display}) ===\n{clean_dialog}\n\n---\n\n")
            record = {
                "id": idx,
                "target_word_count": target_words,
                "text": clean_dialog,
                "spans": spans
            }
            jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            successful_requests += 1
            time.sleep(REQUEST_DELAY)

    print(f"\nГотово! Успешных запросов: {successful_requests}")
    print(f"Файлы: {output_txt}, {output_jsonl}")

if __name__ == "__main__":
    main()
