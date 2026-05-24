# AnonMed

`AnonMed` is a Python project for deterministic preprocessing of noisy Russian ASR text and rule-based detection of numeric PII on top of that normalized text.

The codebase is intentionally split into two layers:

- `anonmed.preprocessing`: ASR cleanup and numeric normalization
- `anonmed.anonymization`: numeric PII detection and masking

That separation keeps input cleanup independent from anonymization logic and leaves room for future model-based pipelines.

## What The Project Does

### Preprocessing layer

The preprocessing pipeline is designed for noisy Russian ASR transcripts. It:

- removes disfluencies and filler words
- removes punctuation while preserving protected separators
- preserves punctuation inside numeric patterns, domains, URLs, and emails
- normalizes spoken numbers into written numeric form
- returns audit information for removed and protected spans

Typical examples:

- `телефон восемь девять один три один два три четыре пять шесть семь` -> `телефон 89131234567`
- `справка мсэ номер ноль восемь семь четыре два три дробь две тысячи двадцать один` -> `справка мсэ номер 087423 дробь 2021`

### Anonymization layer

The anonymization layer runs on normalized text and detects numeric PII using contextual rules.

Currently supported numeric entity types:

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

This layer can:

- return structured matches
- normalize matched values into canonical form
- mask detected spans in text
- run as a full end-to-end pipeline together with preprocessing

## Package Layout

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
        tokenization.py
        types.py

scripts/
  evaluate_numeric_pii_metrics.py
```

## Installation

For development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

For API usage:

```bash
pip install -e .[api]
```

## Main Public APIs

### 1. Run preprocessing only

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

Returned object includes:

- `original_text`
- `disfluency_cleaned_text`
- `punctuation_cleaned_text`
- `cleaned_text`
- `normalized_text`
- `removed_spans`
- `punctuation_removed_spans`
- `punctuation_protected_spans`
- `integer_spans`

### 2. Run numeric PII on normalized text

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

### 3. Run the full end-to-end pipeline

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

Returned object includes:

- `original_text`
- `preprocessing_result`
- `matches`
- `masked_text`

## CLI

Install in editable mode and use:

```bash
anonmed-preprocess "ну эм номер один два" --run
python -m anonmed.cli "ну эм номер один два" --run
```

Supported CLI modes:

- default: print extracted numeric spans as JSON
- `--replace`: replace numeric spans only
- `--clean`: remove disfluencies only
- `--punctuation-clean`: remove punctuation only
- `--run`: run the full preprocessing pipeline and print normalized text
- `--run-json`: run the full preprocessing pipeline and print structured JSON
- `--keep-punctuation`: disable punctuation removal in `--run` and `--run-json`

Examples:

```bash
anonmed-preprocess "ну, эм, номер один два три" --run
# номер 123

anonmed-preprocess "сайт test.com, код один два" --punctuation-clean
# сайт test.com код один два
```

## HTTP API

Start the API with:

```bash
uvicorn anonmed.api:create_app --factory
```

Current HTTP API exposes preprocessing endpoints:

- `/v1/asr-integer/parse`
- `/v1/asr-integer/punctuation-clean`
- `/v1/asr-integer/run`
- `/v1/preprocessing/asr/parse`
- `/v1/preprocessing/asr/punctuation-clean`
- `/v1/preprocessing/asr/run`

At the moment the HTTP API covers preprocessing and numeric span extraction from that layer. Numeric PII anonymization is currently exposed through Python APIs and scripts, not through separate HTTP endpoints.

## Evaluation Script

The repository contains an evaluation script for numeric PII on JSONL datasets:

```bash
.venv/bin/python scripts/evaluate_numeric_pii_metrics.py gt_asr.jsonl
.venv/bin/python scripts/evaluate_numeric_pii_metrics.py gt_asr.jsonl --json
```

The script:

- runs the full pipeline `text -> preprocessing -> numeric PII`
- evaluates only numeric PII types
- computes `precision`, `recall`, `f1`, `accuracy`
- computes `hard` and `soft` metrics
- evaluates `hard_negatives` separately

### Hard vs soft metrics

- `hard`: strict mention-level matching by type and canonical value
- `soft`: deduplicated and more forgiving matching for fragmented numeric entities inside a record

### Artifacts

Each run creates a directory:

```text
artifacts/YYYY-MM-DD/HH-MM-SS/
```

Inside it:

- `dataset_after_preprocessing.jsonl`
- `dataset_after_model.jsonl`
- `metrics.json`
- `run_metadata.json`

`dataset_after_preprocessing.jsonl` stores the original record text together with `preprocessed_text`.

`dataset_after_model.jsonl` stores:

- `preprocessed_text`
- `masked_text`
- model matches with type, span, raw value, and canonical value

You can override the root directory:

```bash
.venv/bin/python scripts/evaluate_numeric_pii_metrics.py gt_asr.jsonl --artifacts-root artifacts
```

## Design Notes

The current implementation is deterministic and rule-based. It is not a general-purpose Russian ITN system and not a learned NER model.

This is intentional:

- preprocessing stays focused on text cleanup and numeric normalization
- anonymization stays focused on contextual numeric PII detection
- evaluation can measure the whole pipeline while keeping layers inspectable

## Current Limitations

- Numeric PII quality depends on preprocessing quality. If spoken numbers are normalized incorrectly, downstream anonymization will miss them.
- `DATE_BIRTH` remains the hardest class because spoken date forms are much more varied than plain digit sequences.
- Some document-like identifiers are still context-sensitive and may require more lexical coverage for new ASR styles.
- The HTTP API does not yet expose a dedicated anonymization endpoint.

## Suggested Workflow

For local debugging:

1. Run preprocessing and inspect `normalized_text`.
2. Run `run_numeric_pii_pipeline(...)` on the same text.
3. If evaluating on a dataset, inspect `artifacts/.../dataset_after_preprocessing.jsonl` first.
4. Then inspect `artifacts/.../dataset_after_model.jsonl` to see what was actually matched and masked.

That usually makes it clear whether an error comes from spoken-number normalization or from the numeric PII rules themselves.

## Url to datasets

https://disk.360.yandex.ru/d/m4rh5c9qIxh3rg
