# AnonMed

`AnonMed` - Python-пакет для анонимизации медицинских ASR-текстов на русском языке.

Главный пользовательский вход сейчас:

```python
from anonmed import PIIAnonymizer, anonymize_pii
```

Проект умеет прогонять текст через полный пайплайн:

```text
исходный текст
-> preprocessing
-> rule-based detection
-> ML detection
-> merge / resolve candidates
-> postprocessing
-> masking / restore original text
-> результат
```

При этом каждый слой можно импортировать и использовать отдельно.

## Что Анонимизируется

Основной сценарий - медицинские диалоги и диктовки после ASR, где персональные данные могут быть
записаны словами, с повторами, без пунктуации или с ошибками распознавания.

Поддерживаемые rule-based числовые типы:

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

ML-слой может добавлять нечисловые сущности, например `PER`, `ADDRESS`, `EMAIL`, если выбранная
модель их возвращает. Текущие зарегистрированные модели находятся в `anonmed.ml.registry`; среди
них есть `natasha_per`, `GLiNER2`, `Qwen06B` и тестовая `example`.

## Быстрый Старт

```python
from anonmed import PIIAnonymizer

anonymizer = PIIAnonymizer()

result = anonymizer(
    "ну, телефон: восемь девять один три один два три четыре пять шесть семь!"
)

print(result.preprocessed_text)
# телефон 89131234567

print(result.masked_text)
# ну, телефон: [PHONE]!
```

`PIIAnonymizer()` без ML-модели запускает preprocessing, rule-based detection и postprocessing.
ML не загружается и тяжёлые ML-зависимости не импортируются.

## Одноразовый Запуск

Для простого разового вызова есть convenience-функция:

```python
from anonmed import anonymize_pii

result = anonymize_pii(
    "паспорт серия сорок пять одиннадцать номер семьсот восемьдесят девять триста двадцать четыре",
    use_ml=False,
)

print(result.masked_text)
# паспорт серия [PASSPORT]
```

Для сервиса, батчей и частых вызовов лучше создавать `PIIAnonymizer` один раз: объект кеширует
ресурсы и лениво инициализирует модель.

## Настройка Пайплайна

Все настройки конкретного запуска можно передавать простыми keyword-аргументами:

```python
from anonmed import PIIAnonymizer

anonymizer = PIIAnonymizer(ml_model="natasha_per")

result = anonymizer(
    text,
    use_preprocessing=True,
    use_rules=True,
    use_ml=True,
    use_postprocessing=True,
    remove_punctuation=False,
    normalize_numbers=True,
    normalize_contacts=True,
    normalize_dates=True,
    pii_types=("PHONE", "SNILS", "PASSPORT", "OMS"),
    ml_labels=("PER",),
    masking_strategy="type",
)
```

`None` означает "не переопределять дефолт". Например, `use_ml=None` берёт значение из
конфига, а `use_ml=False` явно выключает ML.

Приоритет настроек:

1. Встроенные defaults.
2. `default_config` из `PIIAnonymizer(...)`.
3. `config` в конкретном вызове.
4. Явные keyword-аргументы или `flags` в конкретном вызове.

Последний уровень всегда сильнее предыдущих.

## Advanced Config API

Для воспроизводимых запусков, тестов, CLI и серверного режима можно использовать typed configs:

```python
from anonmed import (
    MLDetectionConfig,
    PIIAnonymizer,
    PIIAnonymizerConfig,
    PostProcessingConfig,
    PreprocessingConfig,
    RuleDetectionConfig,
)

config = PIIAnonymizerConfig(
    preprocessing=PreprocessingConfig(
        enabled=True,
        remove_punctuation=False,
        normalize_numbers=True,
        normalize_contacts=True,
    ),
    rules=RuleDetectionConfig(
        enabled=True,
        pii_types=("PHONE", "SNILS", "PASSPORT"),
    ),
    ml=MLDetectionConfig(
        enabled=False,
        labels=("PER", "ADDRESS", "EMAIL"),
    ),
    postprocessing=PostProcessingConfig(
        enabled=True,
        restore_non_pii=True,
        masking_strategy="type",
    ),
)

anonymizer = PIIAnonymizer(default_config=config)
result = anonymizer(text)
```

Внутри каждый запуск приводится к `ResolvedPIIAnonymizerConfig`, поэтому логика defaults не
размазана по этапам.

## Результат

`PIIAnonymizer` возвращает не строку, а структурированный `PIIAnonymizationResult`:

```python
result.original_text
result.preprocessed_text
result.masked_text
result.masked_preprocessed_text
result.masked_original_text

result.candidates
result.rule_candidates
result.ml_candidates

result.postprocessed_mentions
result.entity_groups

result.preprocessing_result
result.postprocessing_result
result.config
result.warnings
```

Для JSON-подобного вывода:

```python
payload = result.to_dict(include_debug=True)
```

## Вызов Отдельных Этапов

Фасад можно использовать не только end-to-end:

```python
from anonmed import PIIAnonymizer

anonymizer = PIIAnonymizer(ml_model="natasha_per")

preprocessed = anonymizer.preprocess(
    text,
    remove_punctuation=False,
    normalize_numbers=True,
)

rule_candidates = anonymizer.detect_by_rules(
    preprocessed.normalized_text,
    pii_types=("PHONE", "SNILS", "PASSPORT"),
)

ml_candidates = anonymizer.detect_by_ml(
    preprocessed.normalized_text,
    labels=("PER",),
)

merged = anonymizer.merge_candidates(
    preprocessed.normalized_text,
    rule_candidates + ml_candidates,
)

postprocessed = anonymizer.postprocess(
    original_text=text,
    preprocessed_text=preprocessed.normalized_text,
    candidates=merged,
    preprocessing_result=preprocessed,
)

print(postprocessed.masked_original_text)
```

Это полезно для отладки: можно понять, где именно потерялась сущность - в preprocessing, правилах,
ML, merge или postprocessing.

## Ленивые Импорты И Ресурсы

`PIIAnonymizer` устроен так, чтобы не импортировать тяжёлые компоненты раньше времени:

- preprocessing-компоненты создаются при первом включённом preprocessing;
- rule-based detector импортируется при первом `detect_by_rules`;
- ML runner и модель создаются только при первом `detect_by_ml` или `__call__` с `use_ml=True`;
- postprocessing вызывается только при включённом этапе.

Пример: этот код не должен загружать ML runner:

```python
from anonmed import PIIAnonymizer

anonymizer = PIIAnonymizer(ml_model="Qwen06B")
result = anonymizer("телефон 89131234567", use_ml=False)
```

## CLI

После установки доступны две команды:

```bash
anonmed "ну, телефон: восемь девять один три один два три четыре пять шесть семь!" --anonymize --no-ml
anonmed-preprocess "ну, номер один два три" --run
```

`anonmed-preprocess` оставлен для совместимости; фактически оба entrypoint ведут в
`anonmed.cli:main`.

Основные режимы:

- режим по умолчанию: вывести извлечённые числовые фрагменты в JSON;
- `--replace`: заменить только числовые фрагменты;
- `--clean`: удалить только речевые сбои и междометия;
- `--punctuation-clean`: удалить только пунктуацию;
- `--run`: запустить preprocessing и вывести нормализованный текст;
- `--run-json`: запустить preprocessing и вывести структурированный JSON;
- `--anonymize`: запустить новый PII anonymization pipeline и вывести `masked_text`;
- `--anonymize-json`: вывести структурированный результат анонимизации.

Полезные флаги анонимизации:

```bash
anonmed "..." --anonymize --ml-model natasha_per --use-ml
anonmed "..." --anonymize --no-ml --pii-types PHONE,SNILS,OMS
anonmed "..." --anonymize --keep-punctuation --masking-strategy same_length
anonmed "..." --anonymize-json --post-processing-mode production_safe
```

## Модули И Импорты

Можно импортировать как верхнеуровневый фасад, так и отдельные модули.

### `anonmed`

Удобный публичный вход:

```python
from anonmed import PIIAnonymizer, anonymize_pii
from anonmed import PreprocessingConfig, RuleDetectionConfig, MLDetectionConfig
from anonmed import PostProcessingConfig, PIIAnonymizerConfig
```

Также отсюда лениво экспортируются основные типы preprocessing, anonymization и numeric PII.

### `anonmed.anonymizer`

Главный фасад нового API:

- `PIIAnonymizer`
- `anonymize_pii`
- `anonymize`
- config dataclasses
- resolved config dataclasses
- `PIIAnonymizationResult`

Используйте этот модуль для пользовательского end-to-end API.

### `anonmed.preprocessing`

ASR preprocessing:

- удаление дисфлюенций и filler-слов;
- удаление пунктуации с защитой email/domain/URL и числовых паттернов;
- нормализация проговорённых чисел;
- нормализация документных номеров, дат рождения и контактов;
- дедупликация повторяющихся ASR-реплик;
- alignment между исходным и нормализованным текстом.

Пример:

```python
from anonmed.preprocessing import run_asr_normalization

result = run_asr_normalization("ну, номер один два три")
print(result.normalized_text)
# номер 123
```

### `anonmed.anonymization`

Низкоуровневая rule-based анонимизация и postprocessing:

- `collect_numeric_pii_candidates`
- `find_numeric_pii`
- `mask_numeric_pii`
- `resolve_pii_candidates`
- `run_pii_post_processing`
- `run_numeric_pii_pipeline`
- `PIICandidate`
- `PostProcessingResult`

Этот слой полезен, если нужен прямой доступ к numeric rules или к механике разрешения пересечений.

Пример:

```python
from anonmed.anonymization import find_numeric_pii

matches = find_numeric_pii("телефон 89131234567")
print(matches[0].pii_type)
# PHONE
```

### `anonmed.ml`

ML primitives, registry, datasets, metrics and runners:

- `PIIModel`
- `TextDocument`, `AnnotationSet`, `Span`
- `ModelRunner`
- `ModelRunnerResult`
- `build_model`
- evaluation metrics and dataset helpers

`ModelRunner` теперь умеет возвращать не только masked string, но и структурный результат:

```python
from anonmed.ml.pipelines.runner import ModelRunner

runner = ModelRunner(model="natasha_per")
result = runner.run("Пациент Иванов Иван Иванович пришел.")

print(result.masked_text)
print(result.spans)
print(result.annotation)
```

Старое поведение сохранено:

```python
masked_text = runner("Пациент Иванов Иван Иванович пришел.")
```

### `anonmed.api`

HTTP API на FastAPI. Сейчас покрывает preprocessing endpoints:

- `/v1/asr-integer/parse`
- `/v1/asr-integer/punctuation-clean`
- `/v1/asr-integer/run`
- `/v1/preprocessing/asr/parse`
- `/v1/preprocessing/asr/punctuation-clean`
- `/v1/preprocessing/asr/run`

Запуск:

```bash
uvicorn anonmed.api:create_app --factory
```

### `anonmed.cli`

CLI entrypoint для preprocessing и anonymization:

```bash
python -m anonmed.cli "телефон 89131234567" --anonymize --no-ml
```

## Установка

Для разработки:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
.venv/bin/python -m pytest
```

Для HTTP API:

```bash
pip install -e .[api]
```

Для ML-компонентов:

```bash
pip install -e .[ml]
```

Некоторые модели могут требовать дополнительные пакеты или локально доступные веса. Например,
`Qwen06B` использует `transformers` и модель `Qwen/Qwen3-0.6B`.

## Скрипт Оценки

Для оценки PII на JSONL-датасетах по всем 15 типам из презентации:

```bash
.venv/bin/python scripts/evaluate_pii_metrics.py gt_asr.jsonl
.venv/bin/python scripts/evaluate_pii_metrics.py gt_asr.jsonl --json
.venv/bin/python scripts/evaluate_pii_metrics.py gt_asr.jsonl --ml-model natasha_per --use-ml
```

Скрипт:

- запускает `PIIAnonymizer`;
- оценивает `PER`, `PHONE`, `ADDRESS`, `SNILS`, `PASSPORT`, `EMAIL`, `DATE_BIRTH`, `OMS`,
  `INN`, `AGE`, `WORKPLACE`, `MSE`, `BIRTH_CERTIFICATE`, `DRIVER_LICENSE`, `TELEGRAM`;
- считает `precision`, `recall`, `f1`, `accuracy`;
- считает hard/soft, span, character, privacy и alignment-метрики;
- дополнительно добавляет блок `ml_metrics` на базе `anonmed.ml.metrics.utils`;
- отдельно оценивает `hard_negatives`;
- пишет instance-файлы с промежуточными результатами.

## Проектные Замечания

- Публичные LLM не используются в финальном пайплайне, чтобы не выносить персональные данные наружу.
- Детерминированный preprocessing остаётся отдельным слоем, потому что качество numeric PII сильно
  зависит от нормализации ASR.
- Rule-based numeric PII остаётся интерпретируемым и хорошо отлаживаемым.
- ML-часть подключается как дополнительный слой, прежде всего для нечисловых сущностей.
- HTTP API пока не предоставляет отдельный endpoint анонимизации; новый anonymization pipeline
  доступен через Python API и CLI.

## Рекомендуемый Workflow Для Отладки

1. Проверить `result.preprocessing_result.normalized_text`.
2. Проверить `result.rule_candidates`.
3. Если включён ML, проверить `result.ml_candidates`.
4. Проверить `result.candidates` после merge.
5. Проверить `result.postprocessed_mentions` и `result.masked_original_text`.

Обычно этого достаточно, чтобы понять, на каком этапе возникла ошибка.


## Датасеты

https://disk.360.yandex.ru/d/m4rh5c9qIxh3rg
