# Russian Spoken Data Generator

Генератор синтетических персональных данных с вербализацией.

Поддерживает:
- телефоны
- СНИЛС
- ИНН
- ОМС
- паспорт
- даты рождения
- возраст
- МСЭ
- свидетельства о рождении
- водительские удостоверения

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

Генератор запускается как модуль Python.

```bash
# СНИЛС (10 примеров)
python -m russian_data_generator.cli --type snils --count 10

# Мобильный телефон (5 примеров)
python -m russian_data_generator.cli --type phone_mobile --count 5

# Паспорт (3 примера)
python -m russian_data_generator.cli --type passport --count 3

# Дата рождения (1 пример)
python -m russian_data_generator.cli --type birthdate

# ИНН, ОМС, МСЭ, свидетельство, права — аналогично
```
