from __future__ import annotations

from asr_integer_extractor.lexicon import FUZZY_VOCABULARY
from asr_integer_extractor.models import LexicalMatch


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row: list[int] = list(range(len(right) + 1))
    current_row: list[int] = [0] * (len(right) + 1)

    for left_index, left_char in enumerate(left, start=1):
        current_row[0] = left_index
        for right_index, right_char in enumerate(right, start=1):
            insertion_cost: int = current_row[right_index - 1] + 1
            deletion_cost: int = previous_row[right_index] + 1
            replacement_cost: int = previous_row[right_index - 1] + int(left_char != right_char)
            current_row[right_index] = min(insertion_cost, deletion_cost, replacement_cost)
        previous_row, current_row = current_row, previous_row

    distance: int = previous_row[-1]
    return distance


def normalized_levenshtein_similarity(left: str, right: str) -> float:
    max_length: int = max(len(left), len(right))
    if max_length == 0:
        return 1.0
    distance: int = levenshtein_distance(left, right)
    similarity: float = 1.0 - float(distance) / float(max_length)
    return max(0.0, similarity)


def character_ngrams(text: str, n: int = 3) -> frozenset[str]:
    padded: str = f"^{text}$"
    if len(padded) <= n:
        return frozenset({padded})
    ngrams: frozenset[str] = frozenset(padded[index : index + n] for index in range(len(padded) - n + 1))
    return ngrams


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union_size: int = len(left | right)
    if union_size == 0:
        return 0.0
    intersection_size: int = len(left & right)
    similarity: float = float(intersection_size) / float(union_size)
    return similarity


def ru_phonetic_key(text: str) -> str:
    replacements: dict[str, str] = {
        "а": "а",
        "о": "а",
        "е": "и",
        "ё": "и",
        "и": "и",
        "ы": "и",
        "э": "и",
        "ю": "у",
        "я": "а",
        "б": "п",
        "в": "ф",
        "г": "к",
        "д": "т",
        "ж": "ш",
        "з": "с",
        "ь": "",
        "ъ": "",
    }
    characters: list[str] = []
    previous: str = ""
    for character in text:
        mapped: str = replacements.get(character, character)
        if mapped and mapped != previous:
            characters.append(mapped)
        previous = mapped
    key: str = "".join(characters)
    return key


def lexical_similarity(source: str, candidate: str) -> float:
    levenshtein_score: float = normalized_levenshtein_similarity(source, candidate)
    ngram_score: float = jaccard_similarity(character_ngrams(source), character_ngrams(candidate))
    phonetic_score: float = 1.0 if ru_phonetic_key(source) == ru_phonetic_key(candidate) else 0.0
    score: float = 0.58 * levenshtein_score + 0.30 * ngram_score + 0.12 * phonetic_score
    return score


def best_fuzzy_match(
    token: str,
    *,
    threshold: float,
    min_length: int,
    vocabulary: tuple[str, ...] = FUZZY_VOCABULARY,
) -> LexicalMatch | None:
    if len(token) < min_length:
        return None

    best_word: str = ""
    best_score: float = 0.0
    for candidate in vocabulary:
        score: float = lexical_similarity(token, candidate)
        if score > best_score:
            best_word = candidate
            best_score = score

    if not best_word or best_score < threshold:
        return None

    match: LexicalMatch = LexicalMatch(
        source=token,
        normalized=token,
        canonical=best_word,
        score=best_score,
        is_fuzzy=True,
    )
    return match


__all__: list[str] = [
    "best_fuzzy_match",
    "character_ngrams",
    "jaccard_similarity",
    "levenshtein_distance",
    "lexical_similarity",
    "normalized_levenshtein_similarity",
    "ru_phonetic_key",
]
