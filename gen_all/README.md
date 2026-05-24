# merge_script.py — сборка объединённого датасета персональных данных

Скрипт `merge_script.py` собирает единый CSV-файл со всеми требуемыми полями, комбинируя текстовые данные из готовых списков, генерируя числовые персональные данные и транслитерируя ники и email-адреса.

## Назначение

Создание набора данных, содержащего следующие колонки:

- `full_address` — полный адрес (текст)
- `nicks_with_at` — Telegram-ники, переведённые в устную форму (транслитерация + разделение цифр, замена `@` → «собака»)
- `email` — email-адрес, приведённый к устной форме (транслитерация + цифры словами)
- `name` — имя и фамилия
- `phone_mobile` — мобильный телефон (цифровая запись)
- `phone_landline` — городской телефон
- `snils` — СНИЛС
- `passport` — паспортные данные
- `birthdate` — дата рождения
- `inn` — ИНН
- `oms` — полис ОМС
- `age` — возраст (число лет)
- `mse` — данные об инвалидности
- `birth_certificate` — свидетельство о рождении
- `driver_license` — водительское удостоверение
- `full_company_name` — полное наименование организации

## Требуемая структура проекта

Скрипт находится в папке `gen_all`, расположенной в корне проекта.  
Корень проекта должен содержать следующие директории и файлы:

.
├── DataForGen/ # Исходные текстовые данные
│ ├── companies_names_only.csv # Колонка: full_company_name
│ ├── full_address_only.csv # Колонка: full_address
│ ├── name_email_filtered.csv # Колонки: email, name
│ └── nicks_with_at.csv # Колонка: tg_nicks (переименовывается в nicks_with_at)
│
├── generate_numbers/ # Генератор числовых данных
│ ├── generate_all_csv.py # Основной скрипт генерации
│ └── README.md
│
├── transclit/ # Транслитератор
│ ├── translit.py # Скрипт транслитерации
│ └── README.md
│
├── gen_all/ # <-- папка со скриптом объединения
│ └── merge_script.py
│
└── requirements.txt # (опционально) зависимости

- Python 3.10+
- `num2words` — установить через `pip install num2words`
- `russian_data_generator` — пакет для генерации числовых ПД (должен быть доступен в `generate_numbers/` или установлен в окружение)

При необходимости установите пакет генератора:
```bash
pip install ./russian_data_generator





python gen_all/merge_script.py --n <количество строк> --output-dir <папка для результата> --output-file <имя выходного файла>

Пример
bash
# Из корня проекта
python gen_all/merge_script.py --n 10000 --output-dir output --output-file full_dataset.csv




