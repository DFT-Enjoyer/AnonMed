## Абстракции

Имеется 3 абстрактных сущности:
- `Dataset`
- `Model`
- `Metric`

### Dataset
Набор сэмплов вида `(TextDocument, AnnotationSet)`
- `TextDocument` - это полилог с ролями.
- `AnnotationSet` - это span'ы (`Span`) с персональными данными внутри реплик полилога.

### Model
Обёртка над моделью: 
- принимает набор текстов `TextDocument`.
- возвращает соответствующий набор меток `AnnotationSet`, внутри себя обращаясь к модели.

### Metric
Обощающий класс для метрик:
- принимает `Dataset` и соответствующий набор `AnnotationSet`, выданный моделью `Model`.
- выдаёт `MetricResult` - словарь с некоторыми измеренными величинами.


## Структура

- В папке `core` есть файл `types.py`, содержащий базовые классы, которые используются описанными выше классами, файл `snapshot.py`, содержащий реализацию создания snapshot'а объекта `Dataset`.
- В папках `datasets` и `models` есть файлы `base.py`, содержащие соответствующий папке абстрактный класс, и файлы с конкретными реализациями этого класса.
- В папке `metrics` есть файл `base.py`, содержащий абстрактный класс `Metric`, файлы с конкретными реализациями этого класса, а так же файл `utils.py` с реализацией служебных функций общего назначения.
- В папке `evaluation` есть файл `evaluator.py`, содержащий универсальный скрипт прогона модели по сэмплам из датасета с сопутствующим вычислением метрик.
- В папке `pipelines` есть файлы, содержащие скрипты, реализующие конкретный пайплайн: подсчёт метрик для модели, анализ датасета и т.д.
- В папке `configs` есть файлы, представляющие собой конфигурационные файлы для контроля параметров запуска пайплайна.
- Файл `registry.py` содержит словари, сопоставляющие тектовому названию из конфига билдер конкретного объекта, реализующего какой-то из классов `Dataset`, `Model`, `Metric`.
- Файл `fabric.py` содержит функции для сборки объектов, непосредственно участвующих в пайплайнах.


## Расширение и запуск

Для добавления новой реализации одной из сущностей `Dataset`, `Model`, `Metric`, нужно:
- Создать в нужной папке соответствующий класс, наследующийся от абстрактного класса.
- Экспортировать этот класс из папки.
- Добавить билдер объекта этого класса в соответствующий словарь в `registry.py`.

Для запуска нужно написать добавить скрипт пайплайна в папку `pipelines` и создать в папке `configs` новый конфигурационный файл, в котором указан ключ билдера нужного класса (ключ находится в файле `registry.py`). Пример со структурой конфига приведён в `configs/example.yaml`, пример со скриптом пайплайна приведён в `pipelines/example.py`. 

Локальные wrapper'ы для `data/in_the_wild_datasets`: `in_the_wild_russian_pii_speech`, `in_the_wild_russian_news_ner`, `in_the_wild_russian_names_addresses`, `in_the_wild_dialog_pii`, `in_the_wild_controlled_synthetic_pii`, `in_the_wild_medical_notes_pii`.

Пайплайн `example.py` запускается командой `python3 -m anonmed.ml.pipelines.example src/anonmed/ml/configs/example.yaml` из корня проекта.

Результаты запуска сохраняются в отдельную директорию внутри `outputs.instance_dir`:

<!-- Что бы визуализировать полученные результаты использовать команду: `python3 -m anonmed.ml.visualization.dashboard --instance-root instance --output instance/dashboard.html`
В `instance` сгенерится файл `dashboard.html`. -->

```text
instance/<run.name>/<YYYY-MM-DD_HH-MM-SS_microseconds>/
```

Например, для `run.name: example` отчёт и snapshot будут лежать в `instance/example/.../report.json` и `instance/example/.../dataset_snapshot.json`.

Dashboard по всем run'ам из `instance` можно собрать в статический HTML без backend-сервера:

```bash
python3 -m anonmed.ml.visualization.dashboard --instance-root instance --output instance/dashboard.html
```

По умолчанию raw-тексты из snapshot'ов не встраиваются в HTML. Для отладки с примерами можно явно добавить `--include-samples`.


## Детали

- Класс `Span` использует полуинтервалы в индексации.
- Индексы в `Span` отсчитываются от начала реплики соответствующего `TextLine`.
