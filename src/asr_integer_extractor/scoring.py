from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from asr_integer_extractor.models import SpanKind

_KIND_ORDER: Final[tuple[SpanKind, ...]] = ("digits", "digit_sequence", "cardinal", "mixed")


def build_feature_vector(
    *,
    token_count: int,
    fuzzy_count: int,
    mean_lexical_score: float,
    has_fraction_tail: bool,
    kind: SpanKind,
) -> NDArray[np.float64]:
    safe_token_count: int = max(token_count, 1)
    fuzzy_ratio: float = float(fuzzy_count) / float(safe_token_count)
    fraction_penalty: float = 1.0 if has_fraction_tail else 0.0
    kind_index: int = _KIND_ORDER.index(kind) if kind in _KIND_ORDER else len(_KIND_ORDER)
    kind_feature: float = float(kind_index) / float(max(len(_KIND_ORDER) - 1, 1))
    features: NDArray[np.float64] = np.array(
        [
            float(safe_token_count),
            fuzzy_ratio,
            float(mean_lexical_score),
            fraction_penalty,
            kind_feature,
        ],
        dtype=np.float64,
    )
    return features


def score_confidence(features: NDArray[np.float64]) -> float:
    if features.shape != (5,):
        raise ValueError(f"Expected feature vector with shape (5,), got {features.shape}.")

    fuzzy_ratio: float = float(features[1])
    mean_lexical_score: float = float(features[2])
    fraction_penalty: float = float(features[3])
    raw_score: float = 0.98
    raw_score -= 0.22 * fuzzy_ratio
    raw_score -= 0.05 * fraction_penalty
    raw_score *= max(0.0, min(1.0, mean_lexical_score))
    confidence: float = max(0.0, min(1.0, raw_score))
    return confidence


__all__: list[str] = ["build_feature_vector", "score_confidence"]
