#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

INPUT_FILE = "medium_dialogs.jsonl"
OUTPUT_TXT = "subdialog.txt"
OUTPUT_JSONL = "subdialog.jsonl"
MIN_WORDS = 30
MAX_WORDS = 60

def word_count(text: str) -> int:
    return len(text.split())

def split_into_utterances(text: str) -> list[str]:
    return text.split('\n')

def compute_prefix_length(utterances: list[str], start_idx: int) -> int:
    if start_idx == 0:
        return 0
    total_len = sum(len(u) for u in utterances[:start_idx])
    total_len += (start_idx - 1)
    return total_len

def adjust_spans(spans: list[dict], utterances: list[str], start: int, end: int) -> list[dict]:
    offset = 0
    bounds = []
    for i, u in enumerate(utterances):
        begin = offset
        end_orig = offset + len(u)
        bounds.append((begin, end_orig))
        offset = end_orig + 1

    prefix_len = compute_prefix_length(utterances, start)
    new_spans = []
    for sp in spans:
        old_begin = sp['begin']
        old_end = sp['end']
        span_reply = None
        for i, (b, e) in enumerate(bounds):
            if b <= old_begin < e:
                span_reply = i
                break
        if span_reply is None:
            continue
        if not (start <= span_reply <= end):
            continue
        if old_end > bounds[span_reply][1]:
            continue
        new_begin = old_begin - prefix_len
        new_end = old_end - prefix_len
        new_sp = sp.copy()
        new_sp['begin'] = new_begin
        new_sp['end'] = new_end
        new_spans.append(new_sp)
    return new_spans

def choose_window(utterances: list[str]) -> tuple[int, int] | None:
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

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Ошибка: {INPUT_FILE} не найден в {Path.cwd()}")
        return

    total_lines = 0
    processed = 0
    skipped = 0
    new_records = []          # для JSONL
    dialogs_with_id = []      # для TXT: список (id, text)

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

            record_id = record.get('id', None)  # может быть None, если нет поля id
            original_text = record.get('text', '')
            if not original_text:
                new_records.append(record)
                dialogs_with_id.append((record_id, ""))
                processed += 1
                continue

            utterances = split_into_utterances(original_text)
            window = choose_window(utterances)
            if window is None:
                new_records.append(record)
                dialogs_with_id.append((record_id, original_text))
                processed += 1
                continue

            start, end = window
            selected = utterances[start:end+1]
            new_text = '\n'.join(selected)
            new_spans = adjust_spans(record.get('spans', []), utterances, start, end)

            new_record = record.copy()
            new_record['text'] = new_text
            new_record['spans'] = new_spans
            new_records.append(new_record)
            dialogs_with_id.append((record_id, new_text))
            processed += 1

    # Запись JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fj:
        for rec in new_records:
            fj.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Запись TXT с id
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
    # Для воспроизводимости раскомментируйте следующую строку:
    # random.seed(42)
    main()