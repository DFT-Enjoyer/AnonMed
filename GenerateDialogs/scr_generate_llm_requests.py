import requests
import json
import csv
import random
import time
import re
from typing import List, Dict, Tuple, Optional

# ========== НАСТРОЙКИ ==========
API_KEY = 
FOLDER_ID = 
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt-5.1/latest"
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

HEADERS = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

REQUEST_DELAY = 0.5
MAX_TOKENS = 8000
TEMPERATURE = 1
PROB_SCENARIO = 0.3          # вероятность взять сценарий из файла, иначе сгенерировать

SCENARIOS_FILE = "scenarios.txt"
MAX_REQUESTS = 

# Словарь русских названий полей
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
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Работаем без фиксированных сценариев.")
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

def call_yandexgpt(prompt: str, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> str:
    payload = {
        "modelUri": MODEL_URI,
        "completionOptions": {"temperature": temperature, "maxTokens": max_tokens},
        "messages": [{"role": "user", "text": prompt}]
    }
    resp = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["result"]["alternatives"][0]["message"]["text"].strip()

def generate_scenario(all_entities: List[Tuple[str, str]]) -> str:
    """Генерирует медицинский сценарий на основе списка ПДн."""
    entities_str = "\n".join([f"  - {type_}: {value}" for type_, value in all_entities])
    prompt = f"""
Ты — генератор медицинских сценариев. На основе предоставленных персональных данных придумай короткий реалистичный медицинский сценарий (на 10–30 слов). Сценарий должен описывать ситуацию, в которой эти данные могут естественно использоваться (например, приём у врача, оформление документов, вызов скорой и т.п.). Упомяни участников разговора (врач, пациент, регистратор и т.д.).

Данные:
{entities_str}

Напиши только сценарий одной строкой, без лишних пояснений.
"""
    return call_yandexgpt(prompt, temperature=0.5, max_tokens=200)

def generate_dialog_with_all_data(
    all_entities: List[Tuple[str, str]],
    target_word_count: int,
    scenario_description: str,
) -> str:
    all_items_str = "\n".join([f"  - {type_}: {value}" for type_, value in all_entities])

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

## Сценарий: {scenario_description}

## Данные (персональные данные человека):
{all_items_str}

Ты — генератор меддиалогов. Сначала составь план, затем диалог.

## ОБЯЗАТЕЛЬНЫЙ ЧЕК-ЛИСТ (каждый пункт проверить):
1. [S:ключ]значение[/S] для каждого персонального данного(используй только то, что дано, каждый предоставленный ключ минимум один раз)
2. Один и тот же ключ не повторяется в двух соседних репликах
3. Возможные ключи: (full_address,telegram_nicks,email,name,phone_mobile,phone_landline,snils,passport,birthdate,inn,oms,age,mse,birth_certificate,driver_license,full_company_name)
4. Даты только числами: ДД.ММ.ГГГГ
5. Email точно как в данных, без изменений.
6. Telegram точно как в данных
7. Все слова должны быть на русском. Email и telegram меняй на емэйл и телеграм соответственно, не меняя вид самих адресов.
. Диалог должен звучать как живая устная речь. Используй разговорные элементы (ага, угу, ну, э-э, ммм, да, так, точно) в умеренном количестве.
   Не вставляй их в каждую реплику. Они должны появляться естественно:
   - когда персонаж колеблется, вспоминает или думает (например, пациент припоминает дату);
   - при подтверждении или переспрашивании (врач: «ага, понял»; регистратор: «так, дальше»);
   - в неформальных коротких реакциях («ой», «ну-у»).
   В ритмичных вопросах-ответах (например, быстрое перечисление документов) они не обязательны, но иногда вставляй.
   Все должно быть как в реальном диалоге. Не должно быть пунктуации. Пиши одним потоком, как в примере:
{few_shot_example}
9. Если данных много, распределяй их по нескольким репликам (например, регистратор спрашивает по одному‑два поля за раз), чтобы диалог звучал естественно.

## Правильный формат:
- Дата: "14.01.1993" (не "четырнадцатое января")
- Номер телефона: "79534954787" (не "семь девять пять три четыре девять пять сорок семь восемь семь")
- С тегом: [S:birthdate]14.01.1993[/S], [S:phone_mobile]79534954787[/S]

Твои действия: сгенерируй диалог, строго следуя правилам.
"""
    return call_yandexgpt(prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

def remove_tags_and_get_spans(marked_text: str) -> Tuple[str, List[Dict]]:
    clean_chars = []
    spans = []
    i = 0
    n = len(marked_text)
    inside_span = False
    span_start_pos = -1
    current_label = None
    span_data_chars = []

    while i < n:
        if marked_text.startswith('[S:', i):
            end_bracket = marked_text.find(']', i)
            if end_bracket != -1:
                label = marked_text[i+3:end_bracket]
                i = end_bracket + 1
                inside_span = True
                span_start_pos = len(clean_chars)
                current_label = label
                span_data_chars = []
                continue
        elif marked_text.startswith('[S]', i):
            i += 3
            inside_span = True
            span_start_pos = len(clean_chars)
            current_label = None
            span_data_chars = []
            continue

        if marked_text.startswith('[/S]', i):
            i += 4
            if inside_span:
                span_data = ''.join(span_data_chars)
                label = current_label if current_label is not None else "unknown"
                spans.append({
                    "begin": span_start_pos,
                    "end": len(clean_chars),
                    "label": label,
                    "data": span_data
                })
                inside_span = False
                current_label = None
            continue

        ch = marked_text[i]
        clean_chars.append(ch)
        if inside_span:
            span_data_chars.append(ch)
        i += 1

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

    # Загружаем фиксированные сценарии из файла
    fixed_scenarios = load_all_scenarios(SCENARIOS_FILE)
    if fixed_scenarios:
        print(f"Загружено {len(fixed_scenarios)} фиксированных сценариев.")
    else:
        print("Фиксированные сценарии не загружены, будем генерировать все сценарии через LLM.")
    print(f"Вероятность использования фиксированного сценария: {PROB_SCENARIO}")
    if max_requests:
        print(f"Максимальное количество запросов к LLM: {max_requests}")

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

            # Собираем все непустые поля
            all_entities = []
            for col_idx, value in enumerate(row):
                if value.strip():
                    field_en = headers[col_idx]
                    ru_name = FIELD_NAME_RU.get(field_en, field_en)
                    all_entities.append((ru_name, value.strip()))

            if not all_entities:
                print(f"Строка {idx}: нет данных, пропуск")
                continue

            target_words = random.randint(50, 250)

            # Решаем, откуда брать сценарий
            use_fixed = random.random() < PROB_SCENARIO and fixed_scenarios
            if use_fixed:
                # Берём случайный фиксированный сценарий
                scenario_index = random.randint(0, len(fixed_scenarios)-1)
                scenario_description, _ = fixed_scenarios[scenario_index]  # требования не используем
                scenario_str_display = f"[ФИКСИРОВАННЫЙ] {scenario_description[:70]}..."
                print(f"\nСтрока {idx}: всего полей {len(all_entities)}")
                print(f"  -> сценарий: {scenario_str_display}")
            else:
                # Генерируем новый сценарий через LLM
                scenario_description = generate_scenario(all_entities)
                scenario_str_display = f"[СГЕНЕРИРОВАННЫЙ] {scenario_description[:70]}..."
                print(f"\nСтрока {idx}: всего полей {len(all_entities)}")
                print(f"  -> сценарий: {scenario_str_display}")

            print(f"  -> целевое число слов: {target_words}")

            # Генерация диалога на основе сценария
            try:
                marked_dialog = generate_dialog_with_all_data(
                    all_entities, target_words, scenario_description
                )
            except Exception as e:
                print(f"  Ошибка генерации диалога: {e}")
                continue

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