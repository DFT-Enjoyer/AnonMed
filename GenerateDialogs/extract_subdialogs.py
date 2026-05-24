#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

INPUT_FILE = "test_data_without_punct"   # измените на ваш файл
OUTPUT_TXT = "subdialogs.txt"
OUTPUT_JSONL = "subdialogs.jsonl"
MIN_WORDS = 20
MAX_WORDS = 70

def word_count(text: str) -> int:
    return len(text.split())

def split_into_utterances(text: str) -> list[str]:
    return text.split('\n')

def get_utterance_bounds(utterances: list[str]) -> list[tuple[int, int]]:
    offset = 0
    bounds = []
    for u in utterances:
        start = offset
        end = offset + len(u)
        bounds.append((start, end))
        offset = end + 1
    return bounds

def count_spans_in_range(spans: list[dict], bounds: list[tuple[int, int]], start_reply: int, end_reply: int) -> int:
    """
    Возвращает количество спанов, которые полностью лежат внутри реплик [start_reply, end_reply].
    """
    if not spans:
        return 0
    # Глобальные границы диапазона
    range_start = bounds[start_reply][0]
    range_end = bounds[end_reply][1]  # конец последней реплики (без учёта \n после неё)
    count = 0
    for sp in spans:
        if sp['begin'] >= range_start and sp['end'] <= range_end:
            count += 1
    return count

def choose_window_by_spans_and_length(utterances: list[str], spans: list[dict]) -> tuple[int, int] | None:
    """
    Выбирает окно из целых реплик (start, end), которое:
    - содержит 1 или 2 спана
    - имеет длину слов в [MIN_WORDS, MAX_WORDS]
    Если таких окон нет, то ищет с 2 спанами (любая длина) с минимальным отклонением,
    затем с 1 спаном, затем без спанов.
    Возвращает (start_reply, end_reply) или None.
    """
    n = len(utterances)
    if n == 0:
        return None
    bounds = get_utterance_bounds(utterances)
    # Соберём все возможные окна
    windows = []  # (start, end, word_count, span_count)
    for start in range(n):
        wc = 0
        for end in range(start, n):
            # Текст окна (целые реплики)
            text_window = '\n'.join(utterances[start:end+1])
            wc = word_count(text_window)
            span_count = count_spans_in_range(spans, bounds, start, end)
            windows.append((start, end, wc, span_count))
    
    # 1. Ищем окна с span_count 1 или 2 и длиной в диапазоне
    good = [(s, e) for (s, e, w, sc) in windows if sc in (1,2) and MIN_WORDS <= w <= MAX_WORDS]
    if good:
        return random.choice(good)
    
    # 2. Ищем с 2 спанами, минимизируем отклонение длины
    windows_2 = [(s, e, w) for (s, e, w, sc) in windows if sc == 2]
    if windows_2:
        best_dev = float('inf')
        best_windows = []
        for s, e, w in windows_2:
            if w < MIN_WORDS:
                dev = MIN_WORDS - w
            elif w > MAX_WORDS:
                dev = w - MAX_WORDS
            else:
                dev = 0
            if dev < best_dev:
                best_dev = dev
                best_windows = [(s, e)]
            elif dev == best_dev:
                best_windows.append((s, e))
        return random.choice(best_windows)
    
    # 3. Ищем с 1 спаном, минимизируем отклонение
    windows_1 = [(s, e, w) for (s, e, w, sc) in windows if sc == 1]
    if windows_1:
        best_dev = float('inf')
        best_windows = []
        for s, e, w in windows_1:
            if w < MIN_WORDS:
                dev = MIN_WORDS - w
            elif w > MAX_WORDS:
                dev = w - MAX_WORDS
            else:
                dev = 0
            if dev < best_dev:
                best_dev = dev
                best_windows = [(s, e)]
            elif dev == best_dev:
                best_windows.append((s, e))
        return random.choice(best_windows)
    
    # 4. Без спанов — откат к fallback (старая логика)
    return choose_window_fallback(utterances)

def choose_window_fallback(utterances: list[str]) -> tuple[int, int] | None:
    """Старая логика: случайное окно из целых реплик длиной 20–70 слов, иначе минимальное отклонение."""
    n = len(utterances)
    if n == 0:
        return None
    windows = []
    for start in range(n):
        wc = 0
        for end in range(start, n):
            wc += word_count(utterances[end])
            windows.append((start, end, wc))
    good = [(s, e) for (s, e, w) in windows if MIN_WORDS <= w <= MAX_WORDS]
    if good:
        return random.choice(good)
    best = []
    best_dev = float('inf')
    for s, e, w in windows:
        if w < MIN_WORDS:
            dev = MIN_WORDS - w
        else:
            dev = w - MAX_WORDS
        if dev < best_dev:
            best_dev = dev
            best = [(s, e)]
        elif dev == best_dev:
            best.append((s, e))
    return random.choice(best) if best else None

def adjust_spans_for_window(original_spans: list[dict],
                            utterances: list[str],
                            start_reply: int,
                            end_reply: int) -> list[dict]:
    """Пересчитывает спаны для выбранного окна (целые реплики)."""
    bounds = get_utterance_bounds(utterances)
    prefix_len = compute_prefix_length(utterances, start_reply)
    new_spans = []
    for sp in original_spans:
        old_begin, old_end = sp['begin'], sp['end']
        # Проверяем, что спан полностью внутри диапазона реплик
        range_start = bounds[start_reply][0]
        range_end = bounds[end_reply][1]
        if old_begin < range_start or old_end > range_end:
            continue
        new_begin = old_begin - prefix_len
        new_end = old_end - prefix_len
        new_sp = sp.copy()
        new_sp['begin'] = new_begin
        new_sp['end'] = new_end
        new_spans.append(new_sp)
    return new_spans

def compute_prefix_length(utterances: list[str], start_idx: int) -> int:
    if start_idx == 0:
        return 0
    total_len = sum(len(u) for u in utterances[:start_idx])
    total_len += (start_idx - 1)
    return total_len

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Ошибка: {INPUT_FILE} не найден в {Path.cwd()}")
        return

    total_lines = 0
    processed = 0
    skipped = 0
    new_records = []
    dialogs_with_id = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines = line_num
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Строка {line_num}: ошибка JSON — {e}")
                skipped += 1
                continue

            record_id = record.get('id')
            original_text = record.get('text', '')
            original_spans = record.get('spans', [])

            if not original_text:
                new_records.append(record)
                dialogs_with_id.append((record_id, ""))
                processed += 1
                continue

            utterances = split_into_utterances(original_text)
            window = choose_window_by_spans_and_length(utterances, original_spans)
            if window is None:
                # ничего не выбрано – копируем исходный диалог
                new_records.append(record)
                dialogs_with_id.append((record_id, original_text))
                processed += 1
                continue

            start_reply, end_reply = window
            selected = utterances[start_reply:end_reply+1]
            new_text = '\n'.join(selected)
            new_spans = adjust_spans_for_window(original_spans, utterances, start_reply, end_reply)

            new_record = record.copy()
            new_record['text'] = new_text
            new_record['spans'] = new_spans
            new_records.append(new_record)
            dialogs_with_id.append((record_id, new_text))
            processed += 1

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fj:
        for rec in new_records:
            fj.write(json.dumps(rec, ensure_ascii=False) + '\n')

    with open(OUTPUT_TXT, 'w', encoding='utf-8') as ft:
        for i, (rec_id, dialog_text) in enumerate(dialogs_with_id):
            if i > 0:
                ft.write('\n=====\n')
            if rec_id is not None:
                ft.write(f"[id: {rec_id}]\n")
            ft.write(dialog_text)

    print(f"Всего строк в файле: {total_lines}")
    print(f"Успешно обработано записей: {processed}")
    print(f"Пропущено из-за ошибок: {skipped}")
    print(f"Результаты сохранены в {OUTPUT_JSONL} и {OUTPUT_TXT}")

if __name__ == '__main__':
    # random.seed(42)
    main()
