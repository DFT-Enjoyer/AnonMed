# AnonMed

`AnonMed` теперь оформлен как более общий Python-проект, в котором текущая логика ASR-очистки и нормализации чисел живет не как отдельный основной пакет, а как preprocessing-слой.

Основной вектор проекта:

- корневой пакет: `anonmed`
- текущий функциональный слой: `anonmed.preprocessing.asr`
- совместимость со старым API: `asr_integer_extractor`

Это оставляет пространство для будущих ML-моделей, обучения, инференса и доменных пайплайнов без привязки всей кодовой базы к одному узкому extractor-модулю.

## Структура

```text
src/
  anonmed/
    api.py
    cli.py
    preprocessing/
      asr/
        confidence.py
        disfluency.py
        fuzzy_matching.py
        number_extractor.py
        number_parser.py
        numeric_lexicon.py
        pipeline.py
        punctuation.py
        tokenization.py
        types.py
  asr_integer_extractor/
    ...
```

Старый пакет `asr_integer_extractor` сохранён как совместимый shim-слой поверх нового `anonmed.preprocessing.asr`.

## Что делает preprocessing-слой

- удаляет дисфлюэнции и filler-слова из русского ASR-текста
- удаляет пунктуацию с сохранением числовых разделителей, доменов, email и URL
- извлекает и нормализует целые числа и последовательности цифр
- возвращает audit trail по удалённым и защищённым span-ам

## Установка для разработки

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Новый импорт

```python
from anonmed.preprocessing import ASRTextPreprocessingPipeline

pipeline = ASRTextPreprocessingPipeline()
result = pipeline.run("ну, эм, номер один два три")

print(result.cleaned_text)
# номер один два три

print(result.normalized_text)
# номер 123
```

Можно работать и точечно:

```python
from anonmed.preprocessing.asr import remove_disfluencies, remove_punctuation

print(remove_disfluencies("Ну, эм, я как бы диктую номер один два три."))
# я диктую номер один два три.

print(remove_punctuation("сайт test.com, код 12.05.2026!"))
# сайт test.com код 12.05.2026
```

## Совместимость со старым API

Старые импорты продолжают работать:

```python
from asr_integer_extractor import IntegerExtractor, PunctuationFilterConfig, run_asr_normalization
```

Старый CLI тоже сохранён:

```bash
asr-integer-extract "ну эм номер один два" --run
```

Новый CLI проекта:

```bash
anonmed-preprocess "ну эм номер один два" --run
python -m anonmed.cli "ну эм номер один два" --run
```

## API

```bash
pip install -e .[api]
uvicorn anonmed.api:create_app --factory
```

Поддерживаются и старые, и новые маршруты:

- `/v1/asr-integer/parse`
- `/v1/asr-integer/punctuation-clean`
- `/v1/asr-integer/run`
- `/v1/preprocessing/asr/parse`
- `/v1/preprocessing/asr/punctuation-clean`
- `/v1/preprocessing/asr/run`

## Ограничения

Это по-прежнему deterministic/fuzzy preprocessing-компонент, а не полноценная ML-модель и не полный русский ITN. Он специально оставлен в preprocessing-слое, чтобы дальше поверх него можно было развивать уже более крупные части `AnonMed`.
=======
Это deterministic/fuzzy extractor, а не полный русский ITN. Он намеренно ориентирован на восстановление целых чисел и последовательностей цифр из ASR, а не на даты, валюты, адреса или проценты.


## Ссылка на датасеты

https://disk.360.yandex.ru/d/m4rh5c9qIxh3rg
