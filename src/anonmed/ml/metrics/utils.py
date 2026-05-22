from dataclasses import dataclass

from anonmed.ml.core.types import AnnotationSet, Case, Span, TextDocument


@dataclass(frozen=True, slots=True)
class EntityUnit:
    line_idx: int
    begin: int
    end: int
    label: str


@dataclass(frozen=True, slots=True)
class Counts:
    tp: int
    fp: int
    fn: int


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def precision(counts: Counts) -> float:
    return safe_div(counts.tp, counts.tp + counts.fp)


def recall(counts: Counts) -> float:
    return safe_div(counts.tp, counts.tp + counts.fn)


def f1(counts: Counts) -> float:
    p_value = precision(counts)
    r_value = recall(counts)
    return safe_div(2.0 * p_value * r_value, p_value + r_value)


def accuracy_without_tn(counts: Counts) -> float:
    return safe_div(counts.tp, counts.tp + counts.fp + counts.fn)


def entities_from_annotation(annotation: AnnotationSet) -> list[EntityUnit]:
    entities: list[EntityUnit] = []
    for line in annotation.lines:
        for span in line.spans:
            entities.append(
                EntityUnit(
                    line_idx=span.line_idx,
                    begin=span.begin,
                    end=span.end,
                    label=span.label,
                )
            )
    return entities


def hard_entity_counts(prediction: AnnotationSet, target: AnnotationSet) -> Counts:
    pred_set = set(entities_from_annotation(prediction))
    target_set = set(entities_from_annotation(target))
    tp = len(pred_set & target_set)
    fp = len(pred_set - target_set)
    fn = len(target_set - pred_set)
    return Counts(tp=tp, fp=fp, fn=fn)


def _entities_overlap(pred: EntityUnit, target: EntityUnit) -> bool:
    if pred.line_idx != target.line_idx:
        return False
    if pred.label != target.label:
        return False
    return max(pred.begin, target.begin) < min(pred.end, target.end)


def _build_soft_edges(pred_entities: list[EntityUnit], target_entities: list[EntityUnit]) -> list[list[int]]:
    graph: list[list[int]] = []
    for pred in pred_entities:
        neighbors: list[int] = []
        for target_idx, target in enumerate(target_entities):
            if _entities_overlap(pred, target):
                neighbors.append(target_idx)
        graph.append(neighbors)
    return graph


def _max_bipartite_match(graph: list[list[int]], right_size: int) -> int:
    matched_right = [-1] * right_size

    def try_augment(left_idx: int, visited: list[bool]) -> bool:
        for right_idx in graph[left_idx]:
            if visited[right_idx]:
                continue
            visited[right_idx] = True
            if matched_right[right_idx] == -1 or try_augment(matched_right[right_idx], visited):
                matched_right[right_idx] = left_idx
                return True
        return False

    matches = 0
    for left_idx in range(len(graph)):
        visited = [False] * right_size
        if try_augment(left_idx, visited):
            matches += 1
    return matches


def soft_entity_counts(prediction: AnnotationSet, target: AnnotationSet) -> Counts:
    pred_entities = entities_from_annotation(prediction)
    target_entities = entities_from_annotation(target)
    graph = _build_soft_edges(pred_entities, target_entities)
    tp = _max_bipartite_match(graph, len(target_entities))
    fp = len(pred_entities) - tp
    fn = len(target_entities) - tp
    return Counts(tp=tp, fp=fp, fn=fn)


def labeled_char_units(annotation: AnnotationSet) -> set[tuple[int, int, str]]:
    units: set[tuple[int, int, str]] = set()
    for entity in entities_from_annotation(annotation):
        for char_idx in range(entity.begin, entity.end):
            units.add((entity.line_idx, char_idx, entity.label))
    return units


def unlabeled_char_units(annotation: AnnotationSet) -> set[tuple[int, int]]:
    units: set[tuple[int, int]] = set()
    for entity in entities_from_annotation(annotation):
        for char_idx in range(entity.begin, entity.end):
            units.add((entity.line_idx, char_idx))
    return units


def hard_char_counts(prediction: AnnotationSet, target: AnnotationSet) -> Counts:
    pred_units = labeled_char_units(prediction)
    target_units = labeled_char_units(target)
    tp = len(pred_units & target_units)
    fp = len(pred_units - target_units)
    fn = len(target_units - pred_units)
    return Counts(tp=tp, fp=fp, fn=fn)


def soft_char_counts(prediction: AnnotationSet, target: AnnotationSet) -> Counts:
    pred_units = unlabeled_char_units(prediction)
    target_units = unlabeled_char_units(target)
    tp = len(pred_units & target_units)
    fp = len(pred_units - target_units)
    fn = len(target_units - pred_units)
    return Counts(tp=tp, fp=fp, fn=fn)


def document_total_chars(document: TextDocument) -> int:
    return sum(len(line.text) for line in document.lines)


def aggregate_counts(cases: tuple[Case, ...], predictions: tuple[AnnotationSet, ...], mode: str) -> Counts:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for case, prediction in zip(cases, predictions, strict=True):
        if mode == "entity_hard":
            current = hard_entity_counts(prediction=prediction, target=case.target)
        elif mode == "entity_soft":
            current = soft_entity_counts(prediction=prediction, target=case.target)
        elif mode == "char_hard":
            current = hard_char_counts(prediction=prediction, target=case.target)
        elif mode == "char_soft":
            current = soft_char_counts(prediction=prediction, target=case.target)
        else:
            raise ValueError(f"Unknown aggregate mode: {mode}")

        total_tp += current.tp
        total_fp += current.fp
        total_fn += current.fn

    return Counts(tp=total_tp, fp=total_fp, fn=total_fn)


def coverage_percent(cases: tuple[Case, ...], predictions: tuple[AnnotationSet, ...]) -> tuple[float, float]:
    covered_chars = 0
    gt_chars = 0
    extra_predicted_chars = 0

    for case, prediction in zip(cases, predictions, strict=True):
        target_chars = unlabeled_char_units(case.target)
        predicted_chars = unlabeled_char_units(prediction)
        covered_chars += len(target_chars & predicted_chars)
        gt_chars += len(target_chars)
        extra_predicted_chars += len(predicted_chars - target_chars)

    coverage = safe_div(covered_chars * 100.0, gt_chars)
    over_coverage = safe_div(extra_predicted_chars * 100.0, gt_chars)
    return coverage, over_coverage
