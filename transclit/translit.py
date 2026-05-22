import csv
import argparse
import sys
import random
from num2words import num2words

# ---------- Словари замен ----------

STRICT_MAP = {
    'A': 'А', 'B': 'Б', 'C': 'Ц', 'D': 'Д', 'E': 'Е', 'F': 'Ф', 'G': 'Г',
    'H': 'Х', 'I': 'И', 'J': 'ДЖ', 'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н',
    'O': 'О', 'P': 'П', 'Q': 'К', 'R': 'Р', 'S': 'С', 'T': 'Т', 'U': 'У',
    'V': 'В', 'W': 'В', 'X': 'Кс', 'Y': 'Й', 'Z': 'З',
    'a': 'а', 'b': 'б', 'c': 'ц', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
    'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
    'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
    'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'й', 'z': 'з',
    '.': 'точка', '-': 'дефис', '_': 'подчеркивание',
}

SPELL_MAP = {
    'A': 'эй', 'B': 'би', 'C': 'си', 'D': 'ди', 'E': 'и', 'F': 'эф',
    'G': 'гэ', 'H': 'эйч', 'I': 'ай', 'J': 'джей', 'K': 'ка',
    'L': 'эл', 'M': 'эм', 'N': 'эн', 'O': 'оу', 'P': 'пи', 'Q': 'кью',
    'R': 'эр', 'S': 'эс', 'T': 'ти', 'U': 'ю', 'V': 'ви', 'W': 'дабл-ю',
    'X': 'экс', 'Y': 'уай', 'Z': 'зед',
    'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и', 'f': 'эф',
    'g': 'гэ', 'h': 'эйч', 'i': 'ай', 'j': 'джей', 'k': 'ка',
    'l': 'эл', 'm': 'эм', 'n': 'эн', 'o': 'оу', 'p': 'пи', 'q': 'кью',
    'r': 'эр', 's': 'эс', 't': 'ти', 'u': 'ю', 'v': 'ви', 'w': 'дабл-ю',
    'x': 'экс', 'y': 'уай', 'z': 'зед',
    '0': 'ноль', '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
    '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь', '9': 'девять',
    '.': 'точка', '-': 'дефис', '_': 'подчеркивание'
}

# ---------- Обработка цифр для режима words ----------

def digits_to_words(digit_str: str) -> str:
    """Преобразует строку из цифр в русскую речь по правилам:
    - ведущие нули: каждый читается как 'ноль'
    - оставшиеся цифры группируются по две; если группа начинается с нуля, читаем 'ноль цифра'
    - иначе число из двух цифр читается целиком (через num2words)
    - последняя одиночная цифра (не ноль) читается как слово
    """
    i = 0
    words = []
    n = len(digit_str)
    
    while i < n:
        if digit_str[i] == '0':
            words.append('ноль')
            i += 1
            continue
        
        if i + 1 < n:
            pair = digit_str[i:i+2]
            if pair[0] == '0':
                words.append('ноль')
                words.append(num2words(int(pair[1]), lang='ru'))
                i += 2
            else:
                words.append(num2words(int(pair), lang='ru'))
                i += 2
        else:
            words.append(num2words(int(digit_str[i]), lang='ru'))
            i += 1
    
    return ' '.join(words)


def transliterate_words_part(part: str) -> str:
    """Обрабатывает часть email (до или после @) в режиме words,
    разделяя буквенные блоки, цифры и знаки пунктуации пробелами."""
    tokens = []
    i = 0
    n = len(part)
    buf = []  # буфер для букв

    def flush_buf():
        if buf:
            tokens.append(''.join(buf))
            buf.clear()

    while i < n:
        ch = part[i]
        if ch.isalpha():
            buf.append(STRICT_MAP.get(ch, ch))
            i += 1
        elif ch.isdigit():
            flush_buf()
            j = i
            while j < n and part[j].isdigit():
                j += 1
            digit_str = part[i:j]
            digit_text = digits_to_words(digit_str)
            tokens.append(digit_text)
            i = j
        else:
            flush_buf()
            tokens.append(STRICT_MAP.get(ch, ch))
            i += 1

    flush_buf()
    return ' '.join(tokens)


# ---------- Функция транслитерации ----------

def transliterate(text: str, mode: str = 'words') -> str:
    if mode == 'random':
        sub_mode = random.choice(['words', 'spell'])
        return transliterate(text, sub_mode)
    
    parts = text.split('@')
    
    if mode == 'words':
        translated_parts = [transliterate_words_part(part) for part in parts]
        return ' собака '.join(translated_parts)
    
    elif mode == 'spell':
        translated_parts = []
        for part in parts:
            names = [SPELL_MAP.get(ch, ch) for ch in part]
            translated_parts.append(' '.join(names))
        return ' собака '.join(translated_parts)
    
    else:
        raise ValueError(f"Неизвестный режим: {mode}. Используйте 'words', 'spell' или 'random'.")


# ---------- Обработка CSV ----------

def process_csv(input_file: str, output_file: str, mode: str, columns: list = None):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("CSV файл пуст или не содержит заголовков.")

        if columns is None:
            columns = reader.fieldnames

        rows = []
        for row in reader:
            new_row = {}
            for field in reader.fieldnames:
                if field in columns:
                    original = row[field]
                    new_row[field] = transliterate(original, mode)
                else:
                    new_row[field] = row[field]
            rows.append(new_row)

    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Готово! Результат сохранён в {output_file}")


# ---------- Командная строка ----------

def main():
    parser = argparse.ArgumentParser(
        description="Транслитерация почтовых адресов и ников Telegram в русские буквы."
    )
    parser.add_argument('mode', choices=['words', 'spell', 'random'],
                        help="words – чтение слитно (буквы), цифры и знаки отдельно; spell – побуквенная диктовка; random – случайный выбор")
    parser.add_argument('input_csv', help="Путь к входному CSV-файлу")
    parser.add_argument('output_csv', help="Путь к выходному CSV-файлу")
    parser.add_argument('--columns', nargs='+',
                        help="Названия столбцов для обработки (по умолчанию все)")

    args = parser.parse_args()
    process_csv(args.input_csv, args.output_csv, args.mode, args.columns)


if __name__ == '__main__':
    main()