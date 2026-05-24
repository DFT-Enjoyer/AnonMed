#!/usr/bin/env python3
import argparse
import json
import string
import sys

def remove_punctuation(text: str) -> str:
    """Удаляет все знаки пунктуации из текста."""
    return text.translate(str.maketrans('', '', string.punctuation))

def is_bad(text: str) -> bool:
    """
    Returns True if the text contains any of the forbidden patterns:
    1. Presence of '@' anywhere in the text (checked before cleaning).
    2. After removing punctuation, any duplicate block after splitting by newline.
    3. After removing punctuation, within any block:
       - two identical consecutive words ('a a'), or
       - a repeated two-word sequence ('a b ... a b').
    """
    # Rule 1: ban '@' (before cleaning)
    if '@' in text:
        return True

    # Split into blocks by newline, keep original for block comparison
    blocks_original = text.split('\n')
    # Rule 2: duplicate blocks (compare after removing punctuation)
    blocks_clean = [remove_punctuation(b) for b in blocks_original]
    if len(set(blocks_clean)) < len(blocks_clean):
        return True

    # Rule 3: duplicate words / word pairs inside each cleaned block
    for block in blocks_clean:
        words = block.split()
        if len(words) < 2:
            continue
        seen_bigrams = set()
        for i in range(len(words) - 1):
            # проверка на два одинаковых слова подряд
            if words[i] == words[i + 1]:
                return True
            bigram = (words[i], words[i + 1])
            if bigram in seen_bigrams:
                return True
            seen_bigrams.add(bigram)
    return False

def main():
    parser = argparse.ArgumentParser(
        description='Filter JSONL file: remove lines with @ symbol, duplicate blocks, or repeated words/bigrams.'
    )
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('output', help='Output JSONL file (good lines only)')
    parser.add_argument('--check', action='store_true',
                        help='Print IDs of removed lines and reject percentage')
    args = parser.parse_args()

    removed_ids = []
    total_lines = 0

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        for raw_line in fin:
            total_lines += 1
            stripped_line = raw_line.rstrip('\n')
            if not stripped_line:
                fout.write(raw_line)
                continue

            try:
                obj = json.loads(stripped_line)
            except json.JSONDecodeError as e:
                print(f'Warning: skipping malformed JSON line: {e}', file=sys.stderr)
                continue

            text = obj.get('text', '')
            if is_bad(text):
                removed_ids.append(obj.get('id'))
            else:
                fout.write(raw_line)

    if args.check:
        print("Removed IDs:")
        for rid in removed_ids:
            print(rid)
        if total_lines > 0:
            percent = (len(removed_ids) / total_lines) * 100
            print(f"Процент брака = {percent:.2f}%")
        else:
            print("Файл пуст")

if __name__ == '__main__':
    main()
