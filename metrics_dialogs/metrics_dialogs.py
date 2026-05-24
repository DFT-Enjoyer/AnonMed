import json
import re
from collections import Counter
from typing import List, Dict
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# При необходимости установите: pip install nltk
# и загрузите ресурсы: nltk.download('punkt')

def load_dialogues_from_jsonl(file_path: str) -> List[str]:
    """Читает JSONL-файл и возвращает список текстов диалогов."""
    dialogues = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                dialogues.append(obj['text'])
            except json.JSONDecodeError as e:
                print(f"Пропущена строка из-за ошибки JSON: {e}")
    return dialogues

def clean_text(text: str) -> str:
    """Удаляет теги [S:...], если есть."""
    return re.sub(r'\[S:.*?\]', '', text)

def get_replicas(dialogue: str) -> List[str]:
    """Извлекает реплики из диалога (строки, начинающиеся с '—'). 
    Если таких нет, разбивает по строкам."""
    replicas = []
    for line in dialogue.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Если строка начинается с '—', считаем её репликой
        if line.startswith('—'):
            replicas.append(line[1:].strip())
        else:
            # Если диалог без тире, добавляем строку как есть
            replicas.append(line)
    return replicas

def distinct_n(replicas: List[str], n: int) -> float:
    """Вычисляет Distinct-n для списка реплик."""
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

def ttr(replicas: List[str]) -> float:
    """Type-Token Ratio."""
    all_words = []
    for rep in replicas:
        all_words.extend(rep.split())
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)

def interjection_ratio(replicas: List[str]) -> float:
    """Доля реплик, содержащих хотя бы одно междометие/разговорный маркер."""
    markers = {'ага', 'угу', 'э-э', 'ммм', 'ну', 'так', 'ой', 'ой-ой', 'вот', 'да', 'нет'}
    count = 0
    for rep in replicas:
        words = set(rep.lower().split())
        if words.intersection(markers):
            count += 1
    return count / len(replicas) if replicas else 0

def avg_replica_length(replicas: List[str]) -> float:
    lengths = [len(rep.split()) for rep in replicas]
    return sum(lengths) / len(lengths) if lengths else 0

def self_bleu(dialogues: List[str]) -> float:
    """Средний BLEU каждого диалога относительно остальных (низкое значение = высокое разнообразие)."""
    if len(dialogues) < 2:
        return 1.0
    smoothie = SmoothingFunction().method1
    scores = []
    for i, dialog in enumerate(dialogues):
        refs = [d.split() for j, d in enumerate(dialogues) if j != i]
        hyp = dialog.split()
        score = sentence_bleu(refs, hyp, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores)

def evaluate_dialogues_from_jsonl(file_path: str) -> Dict:
    # Загружаем и чистим тексты
    raw_dialogues = load_dialogues_from_jsonl(file_path)
    cleaned = [clean_text(d) for d in raw_dialogues]

    # Собираем все реплики
    all_replicas = []
    for d in cleaned:
        all_replicas.extend(get_replicas(d))

    if not all_replicas:
        return {'error': 'Не найдено реплик'}

    metrics = {
        'num_dialogues': len(cleaned),
        'total_replicas': len(all_replicas),
        'distinct_1': distinct_n(all_replicas, 1),
        'distinct_2': distinct_n(all_replicas, 2),
        'interjection_ratio': interjection_ratio(all_replicas),
        'avg_replica_length': avg_replica_length(all_replicas),
    }

    if len(cleaned) > 1:
        metrics['self_bleu'] = self_bleu(cleaned)

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