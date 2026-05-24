from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TextAlignment:
    source_text: str
    target_text: str
    target_to_source_spans: tuple[SourceSpan, ...]

    def source_span_for_target_span(self, start: int, end: int) -> SourceSpan:
        bounded_start: int = max(0, start)
        bounded_end: int = min(len(self.target_to_source_spans), end)
        spans: tuple[SourceSpan, ...] = self.target_to_source_spans[bounded_start:bounded_end]
        non_empty_spans: list[SourceSpan] = [span for span in spans if span.start < span.end]
        if not non_empty_spans:
            return SourceSpan(start=0, end=0)

        source_start: int = min(span.start for span in non_empty_spans)
        source_end: int = max(span.end for span in non_empty_spans)
        return SourceSpan(start=source_start, end=source_end)


class ReplacementSpan(Protocol):
    start: int
    end: int
    normalized: str


def align_texts_by_diff(source_text: str, target_text: str) -> TextAlignment:
    source_spans: list[SourceSpan] = []
    matcher: SequenceMatcher = SequenceMatcher(a=source_text, b=target_text, autojunk=False)

    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        target_length: int = target_end - target_start
        if tag == "equal":
            for offset in range(target_length):
                source_index: int = source_start + offset
                source_spans.append(SourceSpan(start=source_index, end=source_index + 1))
            continue

        if tag == "replace":
            replacement_source_span = SourceSpan(start=source_start, end=source_end)
            source_spans.extend(replacement_source_span for _ in range(target_length))
            continue

        if tag == "insert":
            insertion_source_span = SourceSpan(start=source_start, end=source_start)
            source_spans.extend(insertion_source_span for _ in range(target_length))
            continue

        if tag == "delete":
            continue

    return TextAlignment(
        source_text=source_text,
        target_text=target_text,
        target_to_source_spans=tuple(source_spans),
    )


def build_replacement_alignment(
    source_text: str,
    replacement_spans: Sequence[ReplacementSpan],
) -> TextAlignment:
    target_characters: list[str] = []
    source_spans: list[SourceSpan] = []
    cursor: int = 0

    for span in sorted(replacement_spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor:
            continue

        for source_index in range(cursor, span.start):
            target_characters.append(source_text[source_index])
            source_spans.append(SourceSpan(start=source_index, end=source_index + 1))

        replacement_source_span = SourceSpan(start=span.start, end=span.end)
        replacement_text: str = _replacement_text(span)
        for character in replacement_text:
            target_characters.append(character)
            source_spans.append(replacement_source_span)

        cursor = span.end

    for source_index in range(cursor, len(source_text)):
        target_characters.append(source_text[source_index])
        source_spans.append(SourceSpan(start=source_index, end=source_index + 1))

    target_text: str = "".join(target_characters)
    return TextAlignment(
        source_text=source_text,
        target_text=target_text,
        target_to_source_spans=tuple(source_spans),
    )


def _replacement_text(span: ReplacementSpan) -> str:
    span_object: Any = span
    value: object = getattr(span_object, "value", span.normalized)
    return str(value)


def compose_alignments(first: TextAlignment, second: TextAlignment) -> TextAlignment:
    composed_spans: list[SourceSpan] = []
    for intermediate_span in second.target_to_source_spans:
        source_span: SourceSpan = first.source_span_for_target_span(
            intermediate_span.start,
            intermediate_span.end,
        )
        composed_spans.append(source_span)

    return TextAlignment(
        source_text=first.source_text,
        target_text=second.target_text,
        target_to_source_spans=tuple(composed_spans),
    )


__all__: list[str] = [
    "SourceSpan",
    "TextAlignment",
    "align_texts_by_diff",
    "build_replacement_alignment",
    "compose_alignments",
]
