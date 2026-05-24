import json
import sys
import re

def count_words(text):
    return len(text.split())

def clean_punctuation_and_lower(old_text):
    """
    Удаляет все символы, кроме:
    - букв (превращает в нижний регистр)
    - цифр
    - пробельных символов
    - точки '.', только если окружена цифрами (слева и справа)
    Возвращает (cleaned_text, new_index)
    """
    new_index = [-1] * len(old_text)
    cleaned_chars = []
    new_pos = 0
    for i, ch in enumerate(old_text):
        if ch.isalpha():
            cleaned_chars.append(ch.lower())
            new_index[i] = new_pos
            new_pos += 1
        elif ch.isdigit():
            cleaned_chars.append(ch)
            new_index[i] = new_pos
            new_pos += 1
        elif ch.isspace():
            cleaned_chars.append(ch)
            new_index[i] = new_pos
            new_pos += 1
        elif ch == '.':
            left_ok = (i > 0 and old_text[i-1].isdigit())
            right_ok = (i+1 < len(old_text) and old_text[i+1].isdigit())
            if left_ok and right_ok:
                cleaned_chars.append('.')
                new_index[i] = new_pos
                new_pos += 1
            else:
                new_index[i] = -1
        else:
            new_index[i] = -1
    cleaned_text = ''.join(cleaned_chars)
    return cleaned_text, new_index

def adjust_span_positions(begin, end, new_index, text_len):
    """
    По старым begin/end возвращает новые координаты после очистки пунктуации.
    text_len - длина исходного текста (для проверки границ).
    """
    # Защита от выхода за границы
    if begin < 0:
        begin = 0
    if end > text_len:
        end = text_len
    if begin >= end:
        return None, None

    first = None
    last = None
    for i in range(begin, end):
        if i < len(new_index) and new_index[i] != -1:
            if first is None:
                first = new_index[i]
            last = new_index[i]
    if first is None:
        return None, None
    return first, last + 1

# Словарь сокращений (целые слова) -> полная форма
ABBR_MAP = {
    "г": "город",
    "ул": "улица",
    "д": "дом",
    "к": "квартира",
    "обл": "область",
    "пер": "переулок",
    "пл": "площадь",
    "пр": "проспект",
    "ш": "шоссе",
    "наб": "набережная",
    "бульв": "бульвар",
    "рн": "район",
    "пркт": "проспект"
}

def expand_address_abbreviations(text, spans):
    """
    Заменяет сокращения только внутри адресных спанов.
    Возвращает (new_text, new_spans).
    """
    # Отбираем адресные спаны, сортируем по убыванию begin (чтобы обрабатывать с конца)
    addr_spans = []
    for idx, span in enumerate(spans):
        label = span.get("label", "").lower()
        if "address" in label:   # охватывает "address", "full_address" и т.п.
            addr_spans.append((idx, span))
    if not addr_spans:
        return text, spans

    # Сортируем по убыванию begin
    addr_spans.sort(key=lambda x: x[1]["begin"], reverse=True)

    new_text = text
    new_spans = list(spans)  # копия

    for orig_idx, span in addr_spans:
        old_begin = span["begin"]
        old_end = span["end"]
        old_data = span["data"]

        # Применяем замену только внутри old_data
        new_data = old_data
        for abbr, full in ABBR_MAP.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            new_data = re.sub(pattern, full, new_data)

        if new_data == old_data:
            continue

        # Заменяем в тексте
        new_text = new_text[:old_begin] + new_data + new_text[old_end:]
        delta = len(new_data) - (old_end - old_begin)

        # Обновляем текущий спан
        span["data"] = new_data
        span["end"] = span["begin"] + len(new_data)

        # Сдвигаем все последующие спаны
        for i in range(len(new_spans)):
            s = new_spans[i]
            if s["begin"] >= old_end:
                s["begin"] += delta
                s["end"] += delta

    return new_text, new_spans

def process_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    old_text = obj.get("text", "")
    if not old_text:
        return obj

    # Шаг 1: удаление пунктуации и приведение регистра
    cleaned_text, new_index = clean_punctuation_and_lower(old_text)

    # Шаг 2: обновляем spans согласно очистке
    spans = obj.get("spans", [])
    new_spans = []
    for span in spans:
        begin = span.get("begin")
        end = span.get("end")
        if begin is not None and end is not None:
            # Защита от некорректных границ
            if end > len(old_text):
                end = len(old_text)
            if begin < 0:
                begin = 0
            if begin >= end:
                continue
            new_begin, new_end = adjust_span_positions(begin, end, new_index, len(old_text))
            if new_begin is not None:
                span["begin"] = new_begin
                span["end"] = new_end
                span["data"] = cleaned_text[new_begin:new_end]
                new_spans.append(span)
            # если None, значит спан состоял из одной пунктуации — пропускаем
        else:
            new_spans.append(span)
    obj["spans"] = new_spans
    obj["text"] = cleaned_text

    # Шаг 3: замена сокращений в адресных спанах
    if new_spans:
        expanded_text, expanded_spans = expand_address_abbreviations(obj["text"], new_spans)
        obj["text"] = expanded_text
        obj["spans"] = expanded_spans

    # Шаг 4: пересчёт target_word_count
    obj["target_word_count"] = count_words(obj["text"])

    return obj

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            obj = process_line(line)
            if obj is not None:
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cleaning_punctuation.py input.jsonl output.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
