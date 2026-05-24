#!/usr/bin/env python3
import json
import sys

def remove_newlines_and_fix_spans(record):
    text = record.get('text', '')
    if not text:
        return record
    # Массив смещений: сколько \n удалено до каждой позиции
    n = len(text)
    shift = [0] * (n + 1)
    removed = 0
    for i, ch in enumerate(text):
        if ch == '\n' or ch == '\r':
            removed += 1
        shift[i+1] = removed
    # Новый текст без \n и \r
    new_text = ''.join(ch for ch in text if ch not in '\n\r')
    # Пересчитываем спаны
    new_spans = []
    for span in record.get('spans', []):
        begin = span.get('begin')
        end = span.get('end')
        if begin is None or end is None:
            continue
        if begin < 0 or end > n or begin >= end:
            continue
        new_begin = begin - shift[begin]
        new_end = end - shift[end]
        if new_begin < 0 or new_end > len(new_text) or new_begin >= new_end:
            continue
        new_data = new_text[new_begin:new_end]
        span['begin'] = new_begin
        span['end'] = new_end
        span['data'] = new_data
        new_spans.append(span)
    record['text'] = new_text
    record['spans'] = new_spans
    record['target_word_count'] = len(new_text.split())
    return record

def main(in_file, out_file):
    with open(in_file, 'r', encoding='utf-8') as fin, \
         open(out_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            new_rec = remove_newlines_and_fix_spans(rec)
            fout.write(json.dumps(new_rec, ensure_ascii=False) + '\n')
    print(f"Удалены \\n и \\r из {in_file}, результат {out_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python remove_newlines.py input.jsonl output.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])