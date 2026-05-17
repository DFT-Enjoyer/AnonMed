# ASR Integer Extractor

Небольшой Python-пакет для неразрушающего извлечения целочисленных спанов из шумного русского ASR-текста.

Главная идея: препроцессинг не удаляет остальной текст. Пакет хранит исходные позиции найденных чисел и умеет заменить только числовые фрагменты, сохранив всё остальное содержимое строки.

## Что поддерживается

- Цифры, уже записанные цифрами: `текст5435453...` → `5435453`.
- Обычные русские числительные: `сорок два` → `42`, `одна тысяча пять` → `1005`.
- Диктовка последовательности цифр: `девять три два четыре` → `9324`.
- Смешанный ASR-вывод: `двадцать 5` → `25`, `минус семь` → `-7`.
- Integer-only policy: `двенадцать с половиной` → `12`, `12 с половиной` → `12`.
- Fuzzy-correction для ограниченного словаря числительных: `двадцат пять` → `25`.
- Возврат `raw`, `normalized`, `start`, `end`, `kind`, `status`, `confidence`.

## Коммерческий профиль

Ядро не использует GPL-зависимости, `pymorphy`-словари или внешние нейросетевые сервисы. Единственная обязательная зависимость — `numpy`, используемая для типизированного confidence scoring. FastAPI вынесен в optional extra.

## Установка для разработки

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m unittest discover -s tests
```

## Пример

```python
from asr_integer_extractor import IntegerExtractor, replace_integer_spans

text = "номер девять три два четыре, потом 43249329 и текст5435453..."
extractor = IntegerExtractor()
items = extractor.extract(text)

print([item.value for item in items])
# ['9324', '43249329', '5435453']

print(replace_integer_spans(text))
# номер 9324, потом 43249329 и текст5435453...
```

## CLI

```bash
asr-integer-extract "номер девять три два четыре, потом 43249329"
```

## Backend-интеграция

Используйте `IntegerExtractor.extract(text)` для structured response. Для REST-слоя можно поставить optional extra:

```bash
pip install -e .[api]
uvicorn asr_integer_extractor.api:create_app --factory
```

## Ограничения

Это deterministic/fuzzy extractor, а не полный русский ITN. Он намеренно ориентирован на восстановление целых чисел и последовательностей цифр из ASR, а не на даты, валюты, адреса или проценты.
