from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Final, Literal

from anonmed.preprocessing.asr.tokenization import tokenize_preserving_spans
from anonmed.preprocessing.asr.types import Token


ContactKind = Literal["email", "telegram"]

_CONTENT_TOKEN_KINDS: Final[frozenset[str]] = frozenset({"word", "digits"})
_EMAIL_LOCAL_CUES: Final[frozenset[str]] = frozenset(
    {
        "email",
        "e-mail",
        "имейл",
        "мейл",
        "мэйл",
        "почта",
        "почту",
        "почты",
        "почте",
        "электронная",
        "электронную",
        "рабочий",
        "рабочая",
        "личный",
        "личная",
        "лично",
    }
)
_TELEGRAM_CUES: Final[frozenset[str]] = frozenset(
    {
        "telegram",
        "телеграм",
        "телеграмм",
        "тг",
        "ник",
        "username",
    }
)
_DOT_WORDS: Final[frozenset[str]] = frozenset({"точка", "dot"})
_HYPHEN_WORDS: Final[frozenset[str]] = frozenset({"дефис", "тире", "минус"})
_UNDERSCORE_WORDS: Final[frozenset[str]] = frozenset(
    {"подчеркивание", "подчеркивания", "underscore"}
)
_CONTACT_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "адрес",
        "алло",
        "если",
        "жду",
        "записал",
        "записываю",
        "место",
        "можно",
        "отлично",
        "паспорт",
        "потребуется",
        "работы",
        "снимки",
        "спасибо",
        "телефон",
        "укажите",
        "фио",
    }
)
_DOMAIN_ALIASES: Final[dict[str, str]] = {
    "gmail": "gmail",
    "google": "google",
    "mail": "mail",
    "yandex": "yandex",
    "бк": "bk",
    "гмейл": "gmail",
    "джимейл": "gmail",
    "джимэйл": "gmail",
    "лист": "list",
    "майл": "mail",
    "меил": "mail",
    "мейл": "mail",
    "мэйл": "mail",
    "рамблер": "rambler",
    "яндекс": "yandex",
}
_TLD_ALIASES: Final[dict[str, str]] = {
    "biz": "biz",
    "com": "com",
    "info": "info",
    "net": "net",
    "org": "org",
    "ru": "ru",
    "ком": "com",
    "нет": "net",
    "орг": "org",
    "рф": "rf",
    "ру": "ru",
}
_CYRILLIC_TRANSLITERATION: Final[dict[str, str]] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ы": "y",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    "ь": "",
    "ъ": "",
}
_CONTACT_ALLOWED_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class ContactSpan:
    start: int
    end: int
    raw: str
    normalized: str
    kind: ContactKind
    reason: str


@dataclass(frozen=True, slots=True)
class ContactNormalizedText:
    original_text: str
    text: str
    spans: tuple[ContactSpan, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ContactNormalizerConfig:
    max_email_local_tokens: int = 8
    max_email_domain_tokens: int = 8
    max_telegram_handle_tokens: int = 7
    telegram_context_window_tokens: int = 8
    require_telegram_context: bool = True


@dataclass(frozen=True, slots=True)
class _ParsedDomain:
    end_index: int
    text: str


@dataclass(frozen=True, slots=True)
class _ParsedContact:
    start_index: int
    end_index: int
    normalized: str
    kind: ContactKind
    reason: str


class ContactNormalizer:
    def __init__(self, config: ContactNormalizerConfig | None = None) -> None:
        self.config: ContactNormalizerConfig = (
            config if config is not None else ContactNormalizerConfig()
        )

    def normalize(self, text: str) -> ContactNormalizedText:
        tokens: list[Token] = tokenize_preserving_spans(text)
        content_tokens: list[Token] = [
            token for token in tokens if token.kind in _CONTENT_TOKEN_KINDS
        ]
        spans: list[ContactSpan] = []
        covered_until: int = -1

        for token_index, token in enumerate(content_tokens):
            if token.start < covered_until or token.normalized != "собака":
                continue

            parsed: _ParsedContact | None = self._parse_at_contact(content_tokens, token_index)
            if parsed is None:
                continue

            start_token: Token = content_tokens[parsed.start_index]
            end_token: Token = content_tokens[parsed.end_index - 1]
            span = ContactSpan(
                start=start_token.start,
                end=end_token.end,
                raw=text[start_token.start : end_token.end],
                normalized=parsed.normalized,
                kind=parsed.kind,
                reason=parsed.reason,
            )
            spans.append(span)
            covered_until = span.end

        normalized_text: str = _replace_contact_spans(text, spans)
        result = ContactNormalizedText(
            original_text=text,
            text=normalized_text,
            spans=tuple(spans),
        )
        return result

    def _parse_at_contact(
        self,
        tokens: list[Token],
        at_index: int,
    ) -> _ParsedContact | None:
        email_contact: _ParsedContact | None = self._parse_email(tokens, at_index)
        if email_contact is not None:
            return email_contact

        telegram_contact: _ParsedContact | None = self._parse_telegram(tokens, at_index)
        return telegram_contact

    def _parse_email(self, tokens: list[Token], at_index: int) -> _ParsedContact | None:
        domain: _ParsedDomain | None = _parse_email_domain(
            tokens,
            at_index + 1,
            max_tokens=self.config.max_email_domain_tokens,
        )
        if domain is None:
            return None

        local_start: int = _find_email_local_start(
            tokens,
            at_index,
            max_tokens=self.config.max_email_local_tokens,
        )
        if local_start >= at_index:
            return None

        local_part: str | None = _build_contact_part(tokens[local_start:at_index], allow_dot=True)
        if local_part is None:
            return None

        normalized: str = f"{local_part}@{domain.text}"
        parsed = _ParsedContact(
            start_index=local_start,
            end_index=domain.end_index,
            normalized=normalized,
            kind="email",
            reason="spoken_email",
        )
        return parsed

    def _parse_telegram(self, tokens: list[Token], at_index: int) -> _ParsedContact | None:
        if self.config.require_telegram_context and not _has_telegram_context(
            tokens,
            at_index,
            window_tokens=self.config.telegram_context_window_tokens,
        ):
            return None

        handle_start: int = at_index + 1
        if handle_start >= len(tokens):
            return None

        handle_end: int = _find_telegram_handle_end(
            tokens,
            handle_start,
            max_tokens=self.config.max_telegram_handle_tokens,
        )
        if handle_end <= handle_start:
            return None

        handle: str | None = _build_contact_part(tokens[handle_start:handle_end], allow_dot=False)
        if handle is None or len(handle) < 3:
            return None

        normalized: str = f"@{handle}"
        parsed = _ParsedContact(
            start_index=at_index,
            end_index=handle_end,
            normalized=normalized,
            kind="telegram",
            reason="spoken_telegram",
        )
        return parsed


def _find_email_local_start(tokens: list[Token], at_index: int, *, max_tokens: int) -> int:
    search_start: int = max(0, at_index - max_tokens)
    cue_index: int | None = None
    for index in range(search_start, at_index):
        if tokens[index].normalized in _EMAIL_LOCAL_CUES:
            cue_index = index

    start_index: int = at_index - 1 if cue_index is None else cue_index + 1
    while start_index < at_index and tokens[start_index].normalized in _EMAIL_LOCAL_CUES:
        start_index += 1
    return start_index


def _parse_email_domain(
    tokens: list[Token],
    start_index: int,
    *,
    max_tokens: int,
) -> _ParsedDomain | None:
    labels: list[str] = []
    current_label_segments: list[str] = []
    saw_dot: bool = False
    max_index: int = min(len(tokens), start_index + max_tokens)
    index: int = start_index

    while index < max_index:
        token: Token = tokens[index]
        token_text: str = token.normalized
        if token_text in _DOT_WORDS:
            if not current_label_segments:
                return None
            label: str = "".join(current_label_segments).strip("-")
            if not label:
                return None
            labels.append(label)
            current_label_segments = []
            saw_dot = True
            index += 1
            continue

        if saw_dot and token_text in _CONTACT_STOP_WORDS:
            break

        segment: str | None = _domain_token_segment(token)
        if segment is None:
            break

        current_label_segments.append(segment)
        candidate_label: str = "".join(current_label_segments).strip("-")
        if saw_dot and candidate_label in _TLD_ALIASES.values():
            labels.append(candidate_label)
            text: str = ".".join(labels)
            parsed = _ParsedDomain(end_index=index + 1, text=text)
            return parsed

        index += 1

    return None


def _domain_token_segment(token: Token) -> str | None:
    if token.kind == "digits":
        return token.normalized

    token_text: str = token.normalized
    if token_text in _TLD_ALIASES:
        return _TLD_ALIASES[token_text]
    if token_text in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[token_text]
    if token_text in _HYPHEN_WORDS:
        return "-"
    if token_text in _UNDERSCORE_WORDS or token_text in _DOT_WORDS:
        return None
    segment: str = _transliterate_token(token_text)
    return segment


def _find_telegram_handle_end(tokens: list[Token], start_index: int, *, max_tokens: int) -> int:
    max_index: int = min(len(tokens), start_index + max_tokens)
    index: int = start_index
    consumed_any: bool = False

    while index < max_index:
        token_text: str = tokens[index].normalized
        if consumed_any and token_text in _CONTACT_STOP_WORDS:
            break
        if _is_underscore_phrase(tokens, index):
            consumed_any = True
            index += 2
            continue
        if _contact_token_segment(tokens[index], allow_dot=False) is None:
            break
        consumed_any = True
        index += 1

    return index


def _has_telegram_context(tokens: list[Token], at_index: int, *, window_tokens: int) -> bool:
    context_start: int = max(0, at_index - window_tokens)
    context_tokens: list[Token] = tokens[context_start:at_index]
    result: bool = any(token.normalized in _TELEGRAM_CUES for token in context_tokens)
    return result


def _build_contact_part(tokens: list[Token], *, allow_dot: bool) -> str | None:
    pieces: list[str] = []
    index: int = 0
    while index < len(tokens):
        if _is_underscore_phrase(tokens, index):
            pieces.append("_")
            index += 2
            continue
        segment: str | None = _contact_token_segment(tokens[index], allow_dot=allow_dot)
        if segment is None:
            return None
        pieces.append(segment)
        index += 1

    value: str = "".join(pieces).strip("._-")
    if not value or _CONTACT_ALLOWED_RE.fullmatch(value) is None:
        return None
    if not any(character.isalnum() for character in value):
        return None
    return value


def _contact_token_segment(token: Token, *, allow_dot: bool) -> str | None:
    token_text: str = token.normalized
    if token.kind == "digits":
        return token.normalized
    if token_text in _DOT_WORDS:
        return "." if allow_dot else None
    if token_text in _HYPHEN_WORDS:
        return "-"
    if token_text in _UNDERSCORE_WORDS:
        return "_"
    if token_text in _CONTACT_STOP_WORDS:
        return None
    segment: str = _transliterate_token(token_text)
    return segment


def _is_underscore_phrase(tokens: list[Token], index: int) -> bool:
    next_index: int = index + 1
    if next_index >= len(tokens):
        return False
    left: str = tokens[index].normalized
    right: str = tokens[next_index].normalized
    result: bool = left == "нижнее" and right in _UNDERSCORE_WORDS
    return result


def _transliterate_token(token_text: str) -> str:
    lowered: str = token_text.lower()
    pieces: list[str] = []
    for character in lowered:
        if character.isascii() and character.isalnum():
            pieces.append(character)
            continue
        replacement: str | None = _CYRILLIC_TRANSLITERATION.get(character)
        if replacement is not None:
            pieces.append(replacement)
    result: str = "".join(pieces)
    return result


def _replace_contact_spans(text: str, spans: list[ContactSpan]) -> str:
    pieces: list[str] = []
    cursor: int = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor:
            continue
        pieces.append(text[cursor:span.start])
        pieces.append(span.normalized)
        cursor = span.end
    pieces.append(text[cursor:])
    result: str = "".join(pieces)
    return result


def normalize_spoken_contacts(
    text: str,
    config: ContactNormalizerConfig | None = None,
) -> str:
    normalizer = ContactNormalizer(config=config)
    result: ContactNormalizedText = normalizer.normalize(text)
    return result.text


__all__: list[str] = [
    "ContactKind",
    "ContactNormalizedText",
    "ContactNormalizer",
    "ContactNormalizerConfig",
    "ContactSpan",
    "normalize_spoken_contacts",
]
