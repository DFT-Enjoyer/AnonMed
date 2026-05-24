# AnonMed

`AnonMed` — Python-проект для детерминированного препроцессинга шумного русского ASR-текста и последующего rule-based поиска числовых персональных данных в нормализованном тексте.

Кодовая база намеренно разделена на два слоя:

- `anonmed.preprocessing`: очистка ASR-текста и числовая нормализация;
- `anonmed.anonymization`: поиск и маскирование числовых персональных данных.

Такое разделение позволяет держать очистку входного текста независимой от логики анонимизации и оставляет возможность для дальнейшего добавления модельных пайплайнов.

## Что делает проект

### Слой препроцессинга

Пайплайн препроцессинга рассчитан на шумные русскоязычные ASR-транскрипты. Он:

- удаляет речевые сбои, междометия и слова-паразиты;
- удаляет пунктуацию, сохраняя защищённые разделители;
- сохраняет пунктуацию внутри числовых паттернов, доменов, URL и email;
- нормализует проговорённые числа в письменную цифровую форму;
- возвращает аудиторскую информацию об удалённых и защищённых фрагментах.

Типичные примеры:

- `телефон восемь девять один три один два три четыре пять шесть семь` -> `телефон 89131234567`
- `справка мсэ номер ноль восемь семь четыре два три дробь две тысячи двадцать один` -> `справка мсэ номер 087423 дробь 2021`

### Слой анонимизации

Слой анонимизации работает поверх нормализованного текста и находит числовые персональные данные с использованием контекстных правил.

Сейчас поддерживаются следующие типы числовых сущностей:

- `PHONE`
- `SNILS`
- `PASSPORT`
- `DATE_BIRTH`
- `OMS`
- `INN`
- `AGE`
- `MSE`
- `BIRTH_CERTIFICATE`
- `DRIVER_LICENSE`

Этот слой умеет:

- возвращать структурированные совпадения;
- нормализовать найденные значения к каноническому виду;
- маскировать найденные фрагменты в тексте;
- запускаться как полный end-to-end пайплайн вместе с препроцессингом.

## Структура пакета

```text
src/
  anonmed/
    __init__.py
    api.py
    cli.py
    anonymization/
      __init__.py
      numeric_pii.py
      pipeline.py
    preprocessing/
      __init__.py
      asr/
        __init__.py
        confidence.py
        disfluency.py
        fuzzy_matching.py
        number_extractor.py
        number_parser.py
        numeric_lexicon.py
        pipeline.py
        punctuation.py
        repetition.py
        tokenization.py
        types.py

scripts/
  evaluate_numeric_pii_metrics.py
```

## Установка

Для разработки:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

Для использования API:

```bash
pip install -e .[api]
```

## Основные публичные API

### 1. Запуск только препроцессинга

```python
from anonmed.preprocessing import run_asr_normalization

result = run_asr_normalization(
    "ну, эм, телефон восемь девять один три один два три четыре пять шесть семь"
)

print(result.cleaned_text)
# телефон восемь девять один три один два три четыре пять шесть семь

print(result.normalized_text)
# телефон 89131234567
```

Возвращаемый объект содержит:

- `original_text`
- `disfluency_cleaned_text`
- `punctuation_cleaned_text`
- `cleaned_text`
- `normalized_text`
- `removed_spans`
- `punctuation_removed_spans`
- `punctuation_protected_spans`
- `integer_spans`

### 2. Дедупликация повторяющихся ASR-реплик

```python
from anonmed.preprocessing import ASRUtterance, deduplicate_asr_utterances

result = deduplicate_asr_utterances(
    [
        ASRUtterance("адрес семь", start=0.0, end=1.2),
        ASRUtterance(
            "адрес семь квартира двенадцать",
            start=1.4,
            end=3.0,
        ),
    ]
)

print(result.raw_transcript)
# адрес семь адрес семь квартира двенадцать

print(result.clean_transcript)
# адрес семь квартира двенадцать
```

Слой дедупликации повторов намеренно сделан консервативным:

- он сохраняет исходную транскрипцию и отдельно формирует очищенную транскрипцию;
- он сравнивает только локальные кандидатные реплики с учётом времени и расстояния между репликами;
- он работает без меток говорящих, но может использовать необязательные поля `speaker`, `start`, `end` и `confidence`, если они доступны;
- он защищает вероятные семантические изменения, например изменение чисел или смену отрицания, если только новая реплика не содержит явного маркера исправления.

### 3. Запуск поиска числовых ПДн по нормализованному тексту

```python
from anonmed.anonymization import find_numeric_pii, mask_numeric_pii
from anonmed.preprocessing import run_asr_normalization

result = run_asr_normalization(
    "телефон восемь девять один три один два три четыре пять шесть семь"
)
matches = find_numeric_pii(result.normalized_text)

print(matches[0].pii_type)
# PHONE

print(matches[0].normalized_value)
# +79131234567

print(mask_numeric_pii(result.normalized_text))
# телефон [PHONE]
```

### 4. Запуск полного end-to-end пайплайна

```python
from anonmed.anonymization import run_numeric_pii_pipeline

result = run_numeric_pii_pipeline(
    "паспорт серия сорок пять одиннадцать семьсот восемьдесят девять "
    "триста двадцать четыре"
)

print(result.preprocessing_result.normalized_text)
# паспорт серия 4511789324

print(result.masked_text)
# паспорт серия [PASSPORT]
```

Возвращаемый объект содержит:

- `original_text`
- `preprocessing_result`
- `matches`
- `masked_text`

## CLI

После установки в editable-режиме можно использовать:

```bash
anonmed-preprocess "ну эм номер один два" --run
python -m anonmed.cli "ну эм номер один два" --run
```

Поддерживаемые режимы CLI:

- режим по умолчанию: вывести извлечённые числовые фрагменты в JSON;
- `--replace`: заменить только числовые фрагменты;
- `--clean`: удалить только речевые сбои и междометия;
- `--punctuation-clean`: удалить только пунктуацию;
- `--run`: запустить полный пайплайн препроцессинга и вывести нормализованный текст;
- `--run-json`: запустить полный пайплайн препроцессинга и вывести структурированный JSON;
- `--keep-punctuation`: отключить удаление пунктуации в режимах `--run` и `--run-json`.

Примеры:

```bash
anonmed-preprocess "ну, эм, номер один два три" --run
# номер 123

anonmed-preprocess "сайт test.com, код один два" --punctuation-clean
# сайт test.com код один два
```

## HTTP API

Запуск API:

```bash
uvicorn anonmed.api:create_app --factory
```

Текущий HTTP API предоставляет эндпоинты препроцессинга:

- `/v1/asr-integer/parse`
- `/v1/asr-integer/punctuation-clean`
- `/v1/asr-integer/run`
- `/v1/preprocessing/asr/parse`
- `/v1/preprocessing/asr/punctuation-clean`
- `/v1/preprocessing/asr/run`

На данный момент HTTP API покрывает препроцессинг и извлечение числовых фрагментов на этом слое. Анонимизация числовых ПДн сейчас доступна через Python API и скрипты, но не вынесена в отдельные HTTP-эндпоинты.

## Скрипт оценки

В репозитории есть скрипт оценки качества поиска числовых ПДн на JSONL-датасетах:

```bash
.venv/bin/python scripts/evaluate_numeric_pii_metrics.py gt_asr.jsonl
.venv/bin/python scripts/evaluate_numeric_pii_metrics.py gt_asr.jsonl --json
```

Скрипт:

- запускает полный пайплайн `text -> preprocessing -> numeric PII`;
- оценивает только числовые типы персональных данных;
- считает `precision`, `recall`, `f1`, `accuracy`;
- считает `hard` и `soft` метрики;
- отдельно оценивает `hard_negatives`.

### Hard и soft метрики

- `hard`: строгое сопоставление на уровне упоминаний по типу и каноническому значению;
- `soft`: дедуплицированное и более мягкое сопоставление для фрагментированных числовых сущностей внутри одной записи.

### Артефакты

Каждый запуск создаёт директорию:

```text
artifacts/YYYY-MM-DD/HH-MM-SS/
```

Внутри неё:

- `dataset_after_preprocessing.jsonl`
- `dataset_after_model.jsonl`
- `metrics.json`
- `run_metadata.json`

`dataset_after_preprocessing.jsonl` хранит исходный текст записи вместе с `preprocessed_text`.

`dataset_after_model.jsonl` хранит:

- `preprocessed_text`
- `masked_text`
- найденные моделью совпадения с типом, фрагментом, исходным значением и каноническим значением.

Корневую директорию для артефактов можно переопределить:

```bash
.venv/bin/python scripts/evaluate_numeric_pii_metrics.py gt_asr.jsonl --artifacts-root artifacts
```

## Проектные замечания

Текущая реализация является детерминированной и rule-based. Это не универсальная русскоязычная ITN-система и не обученная NER-модель.

Это сделано намеренно:

- препроцессинг остаётся сфокусированным на очистке текста и числовой нормализации;
- анонимизация остаётся сфокусированной на контекстном поиске числовых ПДн;
- оценка может измерять качество всего пайплайна, при этом отдельные слои остаются интерпретируемыми.

## Текущие ограничения

- Качество поиска числовых ПДн зависит от качества препроцессинга. Если проговорённые числа нормализованы неправильно, нижележащий слой анонимизации их пропустит.
- `DATE_BIRTH` остаётся самым сложным классом, потому что устные формы дат значительно разнообразнее обычных последовательностей цифр.
- Некоторые документные идентификаторы всё ещё сильно зависят от контекста и могут требовать расширения лексического покрытия под новые ASR-стили.
- HTTP API пока не предоставляет отдельный эндпоинт анонимизации.

## Рекомендуемый рабочий процесс

Для локальной отладки:

1. Запустить препроцессинг и проверить `normalized_text`.
2. Запустить `run_numeric_pii_pipeline(...)` на том же тексте.
3. При оценке на датасете сначала посмотреть `artifacts/.../dataset_after_preprocessing.jsonl`.
4. Затем посмотреть `artifacts/.../dataset_after_model.jsonl`, чтобы понять, что именно было найдено и замаскировано.

Обычно это позволяет быстро определить, где возникла ошибка: на этапе нормализации проговорённых чисел или в самих правилах поиска числовых ПДн.

## Ссылка на датасеты

https://disk.360.yandex.ru/d/m4rh5c9qIxh3rg
