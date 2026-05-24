#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys

SUFFIX = 'final_cleaned_subdialogs'   # ищем файлы с таким суффиксом
ERRORS_DIR = 'validation_errors'
DIRECTORY = '.'

def load_records(filepath):
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def save_records(records, filepath):
    if filepath.endswith('.jsonl'):
        with open(filepath, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

def validate_record(record, doc_idx):
    text = record.get('text', '')
    errors = []
    if not text:
        errors.append({'error': 'No text field', 'doc_idx': doc_idx})
        return errors, None

    spans = record.get('spans', [])
    for sp_idx, span in enumerate(spans):
        begin = span.get('begin')
        end = span.get('end')
        data = span.get('data', '')
        label = span.get('label', 'unknown')

        if begin is None or end is None:
            errors.append({
                'doc_idx': doc_idx,
                'span_idx': sp_idx,
                'error': 'Missing begin/end',
                'span': span
            })
            continue
        if begin < 0 or end > len(text):
            errors.append({
                'doc_idx': doc_idx,
                'span_idx': sp_idx,
                'error': f'Out of bounds: {begin}-{end}, text length={len(text)}',
                'span': span
            })
            continue
        extracted = text[begin:end]
        if extracted != data:
            errors.append({
                'doc_idx': doc_idx,
                'span_idx': sp_idx,
                'error': 'Text mismatch',
                'expected': data,
                'extracted': extracted,
                'span': span,
                'context': text[max(0, begin-20):min(len(text), end+20)]
            })
    return errors, record if errors else None

def find_files(root=DIRECTORY, suffix=SUFFIX):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(f'{suffix}.json') or fname.endswith(f'{suffix}.jsonl'):
                files.append(os.path.join(dirpath, fname))
    return files

def main():
    root = DIRECTORY
    if not os.path.exists(root):
        print(f"Папка {root} не найдена")
        sys.exit(1)

    os.makedirs(ERRORS_DIR, exist_ok=True)

    files = find_files(root)
    if not files:
        print(f"Не найдено файлов с суффиксом {SUFFIX}")
        sys.exit(1)

    print(f"Проверка файлов (суффикс {SUFFIX}):\n")
    total_records = 0
    error_records = 0
    total_errors = 0

    for fpath in files:
        rel_path = os.path.relpath(fpath, root)
        records = load_records(fpath)
        if not records:
            continue
        total_records += len(records)
        file_errors = []
        error_records_for_file = 0
        error_objects = []

        for idx, rec in enumerate(records):
            errs, bad_rec = validate_record(rec, idx)
            if errs:
                file_errors.extend(errs)
                error_records_for_file += 1
                error_records += 1
                total_errors += len(errs)
                if bad_rec:
                    error_objects.append(bad_rec)

        if file_errors:
            print(f"❌ {rel_path}: {error_records_for_file} записей с ошибками, всего ошибок {len(file_errors)}")
            base_name = os.path.basename(fpath).replace('.json', '').replace('.jsonl', '')
            out_dir = os.path.join(ERRORS_DIR, base_name)
            os.makedirs(out_dir, exist_ok=True)
            if error_objects:
                orig_path = os.path.join(out_dir, 'errors_original.jsonl')
                save_records(error_objects, orig_path)
            report_path = os.path.join(out_dir, 'errors_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'file': rel_path,
                    'errors': file_errors
                }, f, ensure_ascii=False, indent=2)
            print(f"   Детали сохранены в {out_dir}")
        else:
            print(f"✅ {rel_path}: корректно ({len(records)} записей)")

    print("\n" + "="*60)
    print(f"ИТОГО: проверено записей {total_records}, из них с ошибками {error_records}")
    print(f"Общее количество ошибок (несовпадений text): {total_errors}")
    if error_records > 0:
        print(f"Ошибочные записи сохранены в папке {ERRORS_DIR}")
    else:
        print("Все проверенные файлы корректны. 🎉")

if __name__ == '__main__':
    main()
