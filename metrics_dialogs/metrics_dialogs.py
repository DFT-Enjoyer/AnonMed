import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Возможные ключи ПДн (те же, что в промпте)
ALL_KEYS = {
    'full_address', 'telegram_nicks', 'email', 'name', 'phone_mobile',
    'phone_landline', 'snils', 'passport', 'birthdate', 'inn', 'oms',
    'age', 'mse', 'birth_certificate', 'driver_license', 'full_company_name'
}

def load_dialogues_from_jsonl(file_path: str) -> List[Dict]:
    """Читает JSONL-файл и возвращает список dict'ов с полями text и spans."""
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Убедимся, что есть поле text и spans (может отсутствовать)
                dialogues.append({
                    'text': obj.get('text', ''),
                    'spans': obj.get('spans', [])
                })
            except json.JSONDecodeError as e:
                print(f"Пропущена строка из-за ошибки JSON: {e}")
    return dialogues

def get_replicas(dialogue_text: str) -> List[str]:
    """Извлекает реплики из текста диалога (строки, начинающиеся с '—')."""
    replicas = []
    for line in dialogue_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('—'):
            replicas.append(line[1:].strip())
        else:
            replicas.append(line)
    return replicas

def distinct_n(replicas: List[str], n: int) -> float:
    all_ngrams = []
    unique_ngrams = set()
    for rep in replicas:
        tokens = rep.split()
        ng = list(ngrams(tokens, n))
        all_ngrams.extend(ng)
        unique_ngrams.update(ng)
    if not all_ngrams:
        return 0.0
    return len(unique_ngrams) / len(all_ngrams)

def interjection_ratio(replicas: List[str]) -> float:
    markers = {'ага', 'угу', 'э-э', 'ммм', 'ну', 'так', 'ой', 'ой-ой', 'вот', 'да', 'нет'}
    count = 0
    for rep in replicas:
        if set(rep.lower().split()).intersection(markers):
            count += 1
    return count / len(replicas) if replicas else 0

def avg_replica_length(replicas: List[str]) -> float:
    lengths = [len(rep.split()) for rep in replicas]
    return sum(lengths) / len(lengths) if lengths else 0

def self_bleu(dialogues_texts: List[str]) -> float:
    if len(dialogues_texts) < 2:
        return 1.0
    smoothie = SmoothingFunction().method1
    scores = []
    for i, dialog in enumerate(dialogues_texts):
        refs = [d.split() for j, d in enumerate(dialogues_texts) if j != i]
        hyp = dialog.split()
        score = sentence_bleu(refs, hyp, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores)

def msttr(replicas: List[str], segment_size: int = 20) -> float:
    all_words = []
    for rep in replicas:
        all_words.extend(rep.split())
    if len(all_words) < segment_size:
        return len(set(all_words)) / len(all_words) if all_words else 0.0
    ttrs = []
    for i in range(0, len(all_words) - segment_size + 1, segment_size):
        segment = all_words[i:i+segment_size]
        ttrs.append(len(set(segment)) / segment_size)
    return sum(ttrs) / len(ttrs)

def extract_slot_contexts_from_spans(dialogues: List[Dict]) -> Dict[str, List[str]]:
    """Извлекает контекст запроса для каждого персонального данного, используя spans.
    Возвращает словарь: label -> список строк-вопросов."""
    slot_contexts = defaultdict(list)
    for d in dialogues:
        text = d['text']
        spans = d.get('spans', [])
        if not spans:
            continue

        # Для каждого span находим строку и левый контекст
        lines = text.split('\n')
        # Для быстрого поиска вычислим начальные позиции строк
        line_starts = [0]
        for line in lines:
            # +1 для символа \n
            line_starts.append(line_starts[-1] + len(line) + 1)

        for span in spans:
            begin = span['begin']
            # Найдём, в какой строке находится begin
            line_idx = 0
            for i in range(len(line_starts)-1):
                if line_starts[i] <= begin < line_starts[i+1]:
                    line_idx = i
                    break
            # Начало строки
            line_start = line_starts[line_idx]
            # Контекст: от начала строки до begin
            left_context = text[line_start:begin].strip()
            # Убираем тире и пробел, если есть
            if left_context.startswith('—'):
                left_context = left_context[1:].strip()
            if left_context:
                label = span['label']
                slot_contexts[label].append(left_context)
    return slot_contexts

def slot_diversity(dialogues: List[Dict]) -> Dict:
    """Вычисляет вариативность формулировок для каждого слота."""
    slot_ctx = extract_slot_contexts_from_spans(dialogues)
    diversities = {}
    total_unique = 0
    covered_slots = 0
    for key, contexts in slot_ctx.items():
        unique_count = len(set(contexts))
        diversities[f'slot_{key}_uniq'] = unique_count
        total_unique += unique_count
        covered_slots += 1
    if covered_slots > 0:
        diversities['slot_avg_unique'] = total_unique / covered_slots
    else:
        diversities['slot_avg_unique'] = 0.0
    diversities['slot_coverage'] = covered_slots / len(ALL_KEYS) if ALL_KEYS else 0
    return diversities

def evaluate_dialogues_from_jsonl(file_path: str) -> Dict:
    raw_dialogues = load_dialogues_from_jsonl(file_path)

    # Собираем все тексты (для self_bleu и реплик)
    texts_only = [d['text'] for d in raw_dialogues]
    all_replicas = []
    for t in texts_only:
        all_replicas.extend(get_replicas(t))

    if not all_replicas:
        return {'error': 'Не найдено реплик'}

    metrics = {
        'num_dialogues': len(raw_dialogues),
        'total_replicas': len(all_replicas),
        'distinct_1': distinct_n(all_replicas, 1),
        'distinct_2': distinct_n(all_replicas, 2),
        'interjection_ratio': interjection_ratio(all_replicas),
        'avg_replica_length': avg_replica_length(all_replicas),
        'msttr': msttr(all_replicas),
    }

    if len(raw_dialogues) > 1:
        metrics['self_bleu'] = self_bleu(texts_only)

    # Слотовое разнообразие на основе spans
    slot_metrics = slot_diversity(raw_dialogues)
    metrics.update(slot_metrics)

    return metrics

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Использование: python eval_dialogues.py path/to/file.jsonl")
        sys.exit(1)

    path = sys.argv[1]
    res = evaluate_dialogues_from_jsonl(path)
    for k, v in res.items():
        print(f"{k}: {v}")