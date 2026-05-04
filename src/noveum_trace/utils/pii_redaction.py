"""
PII redaction utilities for Noveum Trace SDK.

This module provides functions to detect and redact personally
identifiable information from trace data.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import re
import unicodedata
from typing import Any

# Compiled once: shared by ``redact_pii``, ``detect_pii_types``, and ``PiiPseudonymizer``.
_PII_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PII_RE_PHONES = (
    re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
    re.compile(r"\b\(\d{3}\)\s*\d{3}-\d{4}\b"),
    re.compile(r"\b\d{3}\.\d{3}\.\d{4}\b"),
    re.compile(r"\b\d{10}\b"),
)
_PII_RE_CARD = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_PII_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PII_RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PII_RE_URL = re.compile(r"https?://[^\s]+")


def redact_pii(text: str, redaction_char: str = "*") -> str:  # noqa: ARG001
    """
    Redact personally identifiable information from text.

    Args:
        text: Text to redact PII from
        redaction_char: Kept for backward compatibility; redaction uses fixed tags
            (e.g. ``[EMAIL_REDACTED]``), not this character.

    Returns:
        Text with PII redacted
    """
    if not isinstance(text, str):
        text = str(text)

    text = _PII_RE_EMAIL.sub("[EMAIL_REDACTED]", text)
    for cre in _PII_RE_PHONES:
        text = cre.sub("[PHONE_REDACTED]", text)
    text = _PII_RE_CARD.sub("[CARD_REDACTED]", text)
    text = _PII_RE_SSN.sub("[SSN_REDACTED]", text)
    text = _PII_RE_IP.sub("[IP_REDACTED]", text)
    text = _PII_RE_URL.sub("[URL_REDACTED]", text)
    return text


def detect_pii_types(text: str) -> list[str]:
    """
    Detect types of PII present in text.

    Args:
        text: Text to analyze

    Returns:
        List of PII types detected
    """
    if not isinstance(text, str):
        text = str(text)

    pii_types: list[str] = []

    if _PII_RE_EMAIL.search(text):
        pii_types.append("email")

    if any(cre.search(text) for cre in _PII_RE_PHONES):
        pii_types.append("phone")

    if _PII_RE_CARD.search(text):
        pii_types.append("credit_card")

    if _PII_RE_SSN.search(text):
        pii_types.append("ssn")

    if _PII_RE_IP.search(text):
        pii_types.append("ip_address")

    return pii_types


# Hex digits from the HMAC-SHA256 digest used after the ``LABEL_`` prefix in
# pseudonyms (tunable; longer suffixes reduce collision rate among distinct values).
TOKEN_SUFFIX_LENGTH = 12


class PiiPseudonymizer:
    """
    Deterministic pseudonymization of PII-like spans using HMAC-SHA256 + salt.

    Uses spaCy ``en_core_web_sm`` when importable and the model is available;
    otherwise relies on regex spans only. No mandatory dependency beyond the
    standard library.
    """

    _NER_LABELS = frozenset({"PERSON", "GPE", "ORG", "LOC"})

    def __init__(self, salt: str) -> None:
        self._salt = salt
        self._salt_bytes = salt.encode("utf-8")
        self._nlp: Any = None
        try:
            spacy = importlib.import_module("spacy")
            self._nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self._nlp = None

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

    def _regex_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for m in _PII_RE_EMAIL.finditer(text):
            spans.append((m.start(), m.end(), "EMAIL"))
        for cre in _PII_RE_PHONES:
            for m in cre.finditer(text):
                spans.append((m.start(), m.end(), "PHONE"))
        for m in _PII_RE_SSN.finditer(text):
            spans.append((m.start(), m.end(), "SSN"))
        for m in _PII_RE_CARD.finditer(text):
            spans.append((m.start(), m.end(), "CARD"))
        for m in _PII_RE_IP.finditer(text):
            spans.append((m.start(), m.end(), "IP"))
        for m in _PII_RE_URL.finditer(text):
            spans.append((m.start(), m.end(), "URL"))
        return spans

    def _ner_spans(self, text: str) -> list[tuple[int, int, str]]:
        if self._nlp is None:
            return []
        doc = self._nlp(text)
        out: list[tuple[int, int, str]] = []
        for ent in doc.ents:
            if ent.label_ in self._NER_LABELS and ent.start_char < ent.end_char:
                out.append((ent.start_char, ent.end_char, ent.label_))
        return out

    @staticmethod
    def _non_overlapping_longest_first(
        spans: list[tuple[int, int, str]],
    ) -> list[tuple[int, int, str]]:
        """Keep non-overlapping spans; when spans overlap, prefer longest (then leftmost)."""
        ordered = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
        accepted: list[tuple[int, int, str]] = []
        for start, end, label in ordered:
            if start >= end:
                continue
            if any(not (end <= a0 or start >= a1) for a0, a1, _ in accepted):
                continue
            accepted.append((start, end, label))
        return accepted

    def pseudonymize(self, text: str) -> str:
        """
        Replace detected PII spans with deterministic pseudonyms.

        Overlapping spans are deduplicated so the longest span wins; replacements
        are applied right-to-left to preserve indices.
        """
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return text

        spans = self._ner_spans(text) + self._regex_spans(text)
        kept = self._non_overlapping_longest_first(spans)
        # Right-to-left by start index descending
        for start, end, label in sorted(kept, key=lambda s: s[0], reverse=True):
            fragment = text[start:end]
            text = text[:start] + self._token(label, fragment) + text[end:]
        return text

    def pseudonymize_dict(self, data: Any) -> Any:
        """Recursively walk dicts and lists; pseudonymize every string value."""
        if isinstance(data, dict):
            return {k: self.pseudonymize_dict(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.pseudonymize_dict(item) for item in data]
        if isinstance(data, str):
            return self.pseudonymize(data)
        return data
