"""
PII redaction utilities for Noveum Trace SDK.

This module provides functions to detect and redact personally
identifiable information from trace data.

Detection is regex-based, plus Google's libphonenumber (``phonenumbers``) for
phone numbers. There is no NER / ML model: name detection happens server-side on
the Noveum ingestion endpoints.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple, Optional, Union

try:
    import phonenumbers
except ImportError:  # pragma: no cover - required dependency; regex fallback below
    phonenumbers = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# User-supplied extra patterns: a list of regexes (label ``CUSTOM``) or a
# ``{"LABEL": "regex"}`` mapping. Same shape as ``SecurityConfig.custom_redaction_patterns``.
CustomPatterns = Union[list[str], dict[str, str], None]

# Strip from URL match end only (prose often appends . , ) ] } : ; ! ? ' ")
_URL_TRAILING_PUNCT = frozenset(".,;:!?)]}'\"")

# E.164 caps a full international number at 15 digits.
_PHONE_MAX_DIGITS = 15
_PHONE_MIN_DIGITS = 8

# Regions whose *local* formats (no country code, e.g. UAE ``050 123 4567``) are
# recognised. Numbers written with ``+`` / an international prefix are found
# for every country regardless of this list.
PHONE_REGIONS: tuple[str, ...] = ("AE", "SA", "QA", "EG", "TR", "GB", "IN", "US")

# Cheap pre-check: libphonenumber is only worth running on text with 7+ digits.
_PHONE_PREFILTER_RE = re.compile(r"\d(?:\D{0,3}\d){6}")

# Official IBAN lengths (SWIFT registry). Only these countries are matched: a
# fixed length keeps the mod-97 check from accepting arbitrary uppercase tokens.
_IBAN_LENGTHS: dict[str, int] = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28,
    "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24,
    "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IQ": 23,
    "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32,
    "LI": 21, "LT": 20, "LU": 20, "LV": 21, "LY": 25, "MC": 27, "MD": 24,
    "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15,
    "OM": 23, "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24,
    "RS": 22, "SA": 24, "SC": 31, "SD": 18, "SE": 24, "SI": 19, "SK": 24,
    "SM": 27, "ST": 25, "SV": 28, "TL": 23, "TN": 24, "TR": 26, "UA": 29,
    "VA": 22, "VG": 24, "XK": 20,
}  # fmt: skip


def _url_match_exclusive_end(raw: str) -> int:
    """Exclusive end index into ``raw`` with trailing sentence punctuation removed."""
    end = len(raw)
    while end > 0 and raw[end - 1] in _URL_TRAILING_PUNCT:
        end -= 1
    return end


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_mod97_valid(compact: str) -> bool:
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


# Validators take a match and return the (absolute) exclusive end of the PII
# span, or ``None`` to reject the match.
_Validator = Callable[["re.Match[str]"], Optional[int]]


def _validate_url(m: re.Match[str]) -> Optional[int]:
    end = m.start() + _url_match_exclusive_end(m.group(0))
    return end if end > m.start() else None


def _validate_emirates_id(m: re.Match[str]) -> Optional[int]:
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    # Separated ``784-YYYY-NNNNNNN-C`` is unambiguous; a bare 15-digit run must
    # also pass the Luhn check (otherwise it falls through to BANK_ACCOUNT).
    if digits != raw or _luhn_valid(digits):
        return m.end()
    return None


def _validate_iban(m: re.Match[str]) -> Optional[int]:
    raw = m.group(0)
    # Map each compact (space-free) length to the raw end index it corresponds to.
    ends: dict[int, int] = {}
    count = 0
    for i, ch in enumerate(raw):
        if ch != " ":
            count += 1
            ends[count] = i + 1
    compact = raw.replace(" ", "")
    n = _IBAN_LENGTHS.get(compact[:2])
    if n is None or n > len(compact) or not _iban_mod97_valid(compact[:n]):
        return None
    end = ends[n]
    # Must not stop mid-word (the regex may have run on into a following token).
    if end != len(raw) and raw[end] != " ":
        return None
    return m.start() + end


def _validate_intl_phone(m: re.Match[str]) -> Optional[int]:
    """Trim trailing digit groups so the number stays within E.164's 15 digits."""
    raw = m.group(0)
    prefix = 2 if raw.startswith("00") else 0
    total = 0
    end = None
    for g in re.finditer(r"\d+", raw[prefix:]):
        if total + len(g.group(0)) > _PHONE_MAX_DIGITS:
            break
        total += len(g.group(0))
        end = prefix + g.end()
    if end is None or total < _PHONE_MIN_DIGITS:
        return None
    return m.start() + end


def _find_phones(text: str) -> Iterator[tuple[int, int]]:
    """Valid phone numbers per libphonenumber, local formats tried per region."""
    if not _PHONE_PREFILTER_RE.search(text):
        return
    seen: set[tuple[int, int]] = set()
    for region in PHONE_REGIONS:
        for m in phonenumbers.PhoneNumberMatcher(
            text, region, leniency=phonenumbers.Leniency.VALID
        ):
            if (m.start, m.end) not in seen:
                seen.add((m.start, m.end))
                yield m.start, m.end


class _PiiPattern(NamedTuple):
    label: str  # pseudonym prefix, e.g. ``EMAIL`` -> ``EMAIL_<hash>``
    detect_name: str  # name reported by ``detect_pii_types``
    regex: Optional[re.Pattern[str]]
    validator: Optional[_Validator] = None
    # Alternative to ``regex``: yields (start, end) spans directly.
    finder: Optional[Callable[[str], Iterator[tuple[int, int]]]] = None

    def find(self, text: str) -> Iterator[tuple[int, int]]:
        """Yield the (start, end) of every accepted match in ``text``."""
        if self.finder is not None:
            yield from self.finder(text)
            return
        assert self.regex is not None
        for m in self.regex.finditer(text):
            end = self.validator(m) if self.validator else m.end()
            if end is not None and end > m.start():
                yield m.start(), end


# Regex fallback for international / local mobile numbers, only used when
# ``phonenumbers`` cannot be imported.
_REGEX_PHONE_PATTERNS: tuple[_PiiPattern, ...] = (
    # +<cc> or 00<cc>, any grouping (e.g. +971 50 123 4567, +44 (0)20 7946 0958).
    _PiiPattern(
        "PHONE",
        "phone",
        re.compile(r"(?<![\w+-])(?:\+|00)\d{1,4}(?:[ .-]?\(?\d{1,5}\)?)+"),
        _validate_intl_phone,
    ),
    # Local mobiles: UAE/Saudi 05X XXX XXXX, Turkey 05XX XXX XX XX,
    # Egypt 01X XXXX XXXX, UK 07XXX XXXXXX.
    _PiiPattern(
        "PHONE", "phone", re.compile(r"(?<![\w+-])0[157](?:[ -]?\d){8,9}(?![\w-])")
    ),
)

if phonenumbers is not None:
    _PHONE_PATTERNS: tuple[_PiiPattern, ...] = (
        _PiiPattern("PHONE", "phone", None, finder=_find_phones),
    )
else:  # pragma: no cover
    _PHONE_PATTERNS = _REGEX_PHONE_PATTERNS


# Single source of truth for ``redact_pii``, ``detect_pii_types`` and
# ``PiiPseudonymizer``. Order matters: when two matches have the same span,
# the earlier pattern's label wins (e.g. a bare 15-digit Emirates ID beats
# BANK_ACCOUNT, a 16-digit card beats BANK_ACCOUNT).
_PII_PATTERNS: tuple[_PiiPattern, ...] = (
    _PiiPattern(
        "EMAIL",
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    ),
    _PiiPattern("URL", "url", re.compile(r"https?://[^\s]+"), _validate_url),
    # 784-YYYY-NNNNNNN-C (UAE country code, birth year, sequence, Luhn check digit)
    _PiiPattern(
        "EMIRATES_ID",
        "emirates_id",
        re.compile(r"(?<![\w-])784[- ]?\d{4}[- ]?\d{7}[- ]?\d(?![\w-])"),
        _validate_emirates_id,
    ),
    # Any-country IBAN, optionally space-grouped in fours; validated with mod-97.
    _PiiPattern(
        "IBAN",
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"),
        _validate_iban,
    ),
    _PiiPattern(
        "CARD",
        "credit_card",
        re.compile(r"(?<![\w-])\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}(?![\w-])"),
    ),
    # Any country (``+``/``00`` prefixed) and local formats for PHONE_REGIONS.
    *_PHONE_PATTERNS,
    # Legacy US-style / bare 10-digit formats (kept regardless of validity).
    _PiiPattern("PHONE", "phone", re.compile(r"\b\d{3}-\d{3}-\d{4}\b")),
    _PiiPattern("PHONE", "phone", re.compile(r"\b\(\d{3}\)\s*\d{3}-\d{4}\b")),
    _PiiPattern("PHONE", "phone", re.compile(r"\b\d{3}\.\d{3}\.\d{4}\b")),
    _PiiPattern("PHONE", "phone", re.compile(r"\b\d{10}\b")),
    _PiiPattern("SSN", "ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    _PiiPattern("IP", "ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # Bare 11–16 digit runs (bank account numbers); last so specific types win ties.
    # A trailing "." only blocks the match when a digit follows (a decimal), so a
    # sentence-ending account number is still caught.
    _PiiPattern(
        "BANK_ACCOUNT",
        "bank_account",
        re.compile(r"(?<![\w.-])\d{11,16}(?![\w-]|\.\d)"),
    ),
)


def compile_custom_patterns(patterns: CustomPatterns) -> list[_PiiPattern]:
    """
    Compile user-supplied patterns into ``_PiiPattern`` entries.

    Accepts a list of regex strings (labelled ``CUSTOM``) or a ``{"LABEL": regex}``
    mapping. Raises ``ValueError`` on a non-string or invalid regex.
    """
    if not patterns:
        return []
    if isinstance(patterns, dict):
        items = list(patterns.items())
    elif isinstance(patterns, (list, tuple)):
        items = [("CUSTOM", p) for p in patterns]
    else:
        raise ValueError(
            "custom_redaction_patterns must be a list of regex strings or a "
            f"{{label: regex}} mapping, not {type(patterns).__name__}"
        )
    compiled: list[_PiiPattern] = []
    for label, pattern in items:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid custom redaction label: {label!r}")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"Invalid custom redaction pattern for {label!r}")
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(
                f"Invalid custom redaction regex for {label!r}: {pattern!r} ({e})"
            ) from e
        norm = label.strip().upper().replace(" ", "_")
        compiled.append(_PiiPattern(norm, norm.lower(), regex))
    return compiled


def _find_spans(
    text: str, patterns: Sequence[_PiiPattern]
) -> list[tuple[int, int, str]]:
    """Every (start, end, label) match across ``patterns``, in pattern order."""
    spans: list[tuple[int, int, str]] = []
    for p in patterns:
        for start, end in p.find(text):
            spans.append((start, end, p.label))
    return spans


def _non_overlapping_longest_first(
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Keep non-overlapping spans; when spans overlap, prefer longest (then leftmost).

    Ties on identical spans keep input order (``sorted`` is stable), so earlier
    patterns win.
    """
    ordered = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    accepted: list[tuple[int, int, str]] = []
    for start, end, label in ordered:
        if start >= end:
            continue
        if any(not (end <= a0 or start >= a1) for a0, a1, _ in accepted):
            continue
        accepted.append((start, end, label))
    return accepted


# Dates and times in free text are not PII, but their digit runs can look like
# phone numbers (e.g. ``12:28:58.495185`` contains a valid Qatar number). Built-in
# matches overlapping these are dropped; custom patterns are not affected.
_TZ = r"(?:\s?(?:Z|UTC|GMT|[+-]\d{2}:?\d{2}))?"
_TIME = r"\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d{1,9})?)?(?:\s?[AaPp]\.?[Mm]\.?)?" + _TZ
_PROTECTED_DATETIME_RE = re.compile(
    r"(?<![\d:.])(?:"
    # ISO date, optionally with time: 2026-10-07, 2026-10-07T12:28:58.495185Z
    rf"\d{{4}}-\d{{2}}-\d{{2}}(?:[T ]{_TIME})?"
    # Day/month/year: 07/10/2026, 7-10-2026, 07.10.2026 (optionally with time)
    rf"|\d{{1,2}}([/.-])\d{{1,2}}\1\d{{4}}(?:,?\s{_TIME})?"
    # Time on its own: 12:28, 12:28:58, 12:28:58.495185, 9:30 PM
    rf"|{_TIME}"
    r")(?![\d:])"
)


def _overlaps_any(start: int, end: int, spans: Sequence[tuple[int, int, str]]) -> bool:
    return any(not (end <= s0 or start >= s1) for s0, s1, _ in spans)


def _select_spans(
    text: str, custom: Sequence[_PiiPattern]
) -> list[tuple[int, int, str]]:
    """Non-overlapping spans to replace; custom patterns take precedence.

    Custom matches are resolved first, then built-in matches fill the gaps: a
    built-in match overlapping any kept custom match is dropped, even if longer.
    Built-in matches overlapping a date or time are dropped too.
    """
    kept = _non_overlapping_longest_first(_find_spans(text, custom))
    blocked = kept + [
        (m.start(), m.end(), "DATETIME") for m in _PROTECTED_DATETIME_RE.finditer(text)
    ]
    builtin = [
        (start, end, label)
        for start, end, label in _find_spans(text, _PII_PATTERNS)
        if not _overlaps_any(start, end, blocked)
    ]
    return kept + _non_overlapping_longest_first(builtin)


def _tile_mask_to_length(unit: str, n: int) -> str:
    """Repeat ``unit`` to cover ``n`` characters (single-char fast path)."""
    if n <= 0:
        return ""
    if len(unit) == 1:
        return unit * n
    return (unit * ((n + len(unit) - 1) // len(unit)))[:n]


def redact_pii(
    text: str, redaction_char: str = "*", custom_patterns: CustomPatterns = None
) -> str:
    """
    Redact personally identifiable information from text.

    Args:
        text: Text to redact PII from
        redaction_char: String to tile over each matched span (same length as the match).
            Empty or invalid values fall back to ``"*"``.
        custom_patterns: Extra regexes (list, or ``{label: regex}`` mapping).

    Returns:
        Text with PII redacted
    """
    if not isinstance(text, str):
        text = str(text)

    unit = (
        redaction_char
        if isinstance(redaction_char, str) and redaction_char.strip()
        else "*"
    )

    spans = _select_spans(text, compile_custom_patterns(custom_patterns))
    for start, end, _ in sorted(spans, key=lambda s: s[0], reverse=True):
        text = text[:start] + _tile_mask_to_length(unit, end - start) + text[end:]
    return text


def detect_pii_types(text: str, custom_patterns: CustomPatterns = None) -> list[str]:
    """
    Detect types of PII present in text.

    Args:
        text: Text to analyze
        custom_patterns: Extra regexes (list, or ``{label: regex}`` mapping).

    Returns:
        List of PII types detected
    """
    if not isinstance(text, str):
        text = str(text)

    custom = compile_custom_patterns(custom_patterns)
    found = {label for _, _, label in _select_spans(text, custom)}
    pii_types: list[str] = []
    for p in list(custom) + list(_PII_PATTERNS):
        if p.label in found and p.detect_name not in pii_types:
            pii_types.append(p.detect_name)
    return pii_types


# Keys whose values are never pseudonymized (at any depth): rewriting ids
# breaks span parent/child links, and timestamps / durations are not PII but
# often look like phone numbers (e.g. Qatar's 8-digit numbers). Any key ending
# in ``_id`` is skipped as well (trace_id, span_id, parent_span_id, ...).
PSEUDONYMIZE_SKIP_KEYS = frozenset(
    {"timestamp", "start_time", "end_time", "duration", "duration_ms"}
)
PSEUDONYMIZE_SKIP_KEY_SUFFIX = "_id"


def _is_skipped_key(key: Any) -> bool:
    return isinstance(key, str) and (
        key in PSEUDONYMIZE_SKIP_KEYS or key.endswith(PSEUDONYMIZE_SKIP_KEY_SUFFIX)
    )


# Hex digits from the HMAC-SHA256 digest used after the ``LABEL_`` prefix in
# pseudonyms (tunable; longer suffixes reduce collision rate among distinct values).
TOKEN_SUFFIX_LENGTH = 12


class PiiPseudonymizer:
    """
    Deterministic pseudonymization of PII-like spans using HMAC-SHA256 + salt.

    Uses ``_PII_PATTERNS`` (regex + ``phonenumbers``) plus optional custom
    patterns.
    """

    def __init__(self, salt: str, custom_patterns: CustomPatterns = None) -> None:
        self._salt = salt
        self._salt_bytes = salt.encode("utf-8")
        try:
            custom = compile_custom_patterns(custom_patterns)
        except ValueError as e:
            # Config validation rejects these up front; never crash the host app here.
            logger.error("Ignoring custom redaction patterns: %s", e)
            custom = []
        self._custom_patterns = custom

    def _token(self, label: str, value: str) -> str:
        """NFC-normalize ``value``, HMAC-SHA256(salt, value), return LABEL_ + hex suffix."""
        raw = value if isinstance(value, str) else str(value)
        normalized = unicodedata.normalize("NFC", raw)
        digest_hex = hmac.new(
            self._salt_bytes,
            normalized.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        prefix = label.strip().upper().replace(" ", "_")
        return f"{prefix}_{digest_hex[:TOKEN_SUFFIX_LENGTH]}"

    _non_overlapping_longest_first = staticmethod(_non_overlapping_longest_first)

    def pseudonymize(self, text: str) -> str:
        """
        Replace detected PII spans with deterministic pseudonyms.

        Custom-pattern matches take precedence over built-in ones; among the
        rest, overlapping spans are deduplicated so the longest span wins.
        Replacements are applied right-to-left to preserve indices.
        """
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return text

        kept = _select_spans(text, self._custom_patterns)
        # Right-to-left by start index descending
        for start, end, label in sorted(kept, key=lambda s: s[0], reverse=True):
            fragment = text[start:end]
            text = text[:start] + self._token(label, fragment) + text[end:]
        return text

    def pseudonymize_dict(self, data: Any) -> Any:
        """Recursively walk dicts and lists; pseudonymize every string value.

        Values under ``PSEUDONYMIZE_SKIP_KEYS`` or keys ending in ``_id`` are
        passed through unchanged.
        """
        if isinstance(data, dict):
            return {
                k: v if _is_skipped_key(k) else self.pseudonymize_dict(v)
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [self.pseudonymize_dict(item) for item in data]
        if isinstance(data, str):
            return self.pseudonymize(data)
        return data
