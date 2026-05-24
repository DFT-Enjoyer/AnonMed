from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence

TextLayer = Literal["original", "normalized"]

__all__: tuple[str, ...] = (
    "OriginalTextRestorer",
    "RestorableMention",
    "RestoredTextResult",
    "TextLayer",
    "patch_text",
    "restore_safe_original_text",
)


class RestorableMention(Protocol):
    original_start: int
    original_end: int
    normalized_start: int
    normalized_end: int
    replacement: str
    projection_status: str
    mention_id: str


@dataclass(frozen=True, slots=True)
class RestoredTextResult:
    safe_text: str
    masked_original_text: str
    masked_normalized_text: str
    used_mentions: tuple[RestorableMention, ...]
    skipped_mentions: tuple[RestorableMention, ...]
    is_safe: bool
    audit: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class OriginalTextRestorer:
    original_layer_name: str = "original"
    normalized_layer_name: str = "normalized"

    def restore(
        self,
        *,
        original_text: str,
        normalized_text: str,
        mentions: Sequence[RestorableMention],
    ) -> RestoredTextResult:
        used_mentions, skipped_mentions = _split_restorable_mentions(
            original_text=original_text,
            normalized_text=normalized_text,
            mentions=mentions,
        )
        masked_original_text: str = patch_text(
            original_text,
            used_mentions,
            layer="original",
        )
        masked_normalized_text: str = patch_text(
            normalized_text,
            used_mentions,
            layer="normalized",
        )
        skipped_status_counts: dict[str, int] = _projection_status_counts(skipped_mentions)
        audit: dict[str, object] = {
            "original_layer": self.original_layer_name,
            "normalized_layer": self.normalized_layer_name,
            "input_mention_count": len(mentions),
            "used_mention_count": len(used_mentions),
            "skipped_mention_count": len(skipped_mentions),
            "skipped_projection_status_counts": skipped_status_counts,
            "safe_text_source": "masked_original_text",
        }
        return RestoredTextResult(
            safe_text=masked_original_text,
            masked_original_text=masked_original_text,
            masked_normalized_text=masked_normalized_text,
            used_mentions=used_mentions,
            skipped_mentions=skipped_mentions,
            is_safe=len(skipped_mentions) == 0,
            audit=audit,
        )


def restore_safe_original_text(
    *,
    original_text: str,
    normalized_text: str,
    mentions: Sequence[RestorableMention],
) -> RestoredTextResult:
    restorer = OriginalTextRestorer()
    return restorer.restore(
        original_text=original_text,
        normalized_text=normalized_text,
        mentions=mentions,
    )


def patch_text(
    text: str,
    mentions: Sequence[RestorableMention],
    *,
    layer: TextLayer = "original",
) -> str:
    parts: list[str] = []
    cursor: int = 0
    sorted_mentions: list[RestorableMention] = sorted(
        mentions,
        key=lambda item: _mention_span(item, layer),
    )
    for mention in sorted_mentions:
        start, end = _mention_span(mention, layer)
        if start < cursor or start >= end:
            continue
        bounded_start: int = max(0, min(len(text), start))
        bounded_end: int = max(0, min(len(text), end))
        if bounded_start < cursor or bounded_start >= bounded_end:
            continue
        parts.append(text[cursor:bounded_start])
        parts.append(mention.replacement)
        cursor = bounded_end
    parts.append(text[cursor:])
    return "".join(parts)


def _split_restorable_mentions(
    *,
    original_text: str,
    normalized_text: str,
    mentions: Sequence[RestorableMention],
) -> tuple[tuple[RestorableMention, ...], tuple[RestorableMention, ...]]:
    used_mentions: list[RestorableMention] = []
    skipped_mentions: list[RestorableMention] = []
    for mention in mentions:
        if _mention_is_restorable(
            original_text=original_text,
            normalized_text=normalized_text,
            mention=mention,
        ):
            used_mentions.append(mention)
        else:
            skipped_mentions.append(mention)
    return tuple(used_mentions), tuple(skipped_mentions)


def _mention_is_restorable(
    *,
    original_text: str,
    normalized_text: str,
    mention: RestorableMention,
) -> bool:
    if mention.projection_status != "ok":
        return False
    if not _valid_span(mention.original_start, mention.original_end, len(original_text)):
        return False
    return _valid_span(mention.normalized_start, mention.normalized_end, len(normalized_text))


def _valid_span(start: int, end: int, text_length: int) -> bool:
    return 0 <= start < end <= text_length


def _projection_status_counts(mentions: Sequence[RestorableMention]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mention in mentions:
        status: str = mention.projection_status
        counts[status] = counts.get(status, 0) + 1
    return counts


def _mention_span(
    mention: RestorableMention,
    layer: TextLayer,
) -> tuple[int, int]:
    if layer == "original":
        return mention.original_start, mention.original_end
    return mention.normalized_start, mention.normalized_end
