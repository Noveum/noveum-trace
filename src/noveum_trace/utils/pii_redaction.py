"""
PII redaction utilities for Noveum Trace SDK.

This module provides functions to detect and redact personally
identifiable information from trace data.
"""

import hashlib
import hmac
import re
import unicodedata
from typing import Any, Optional


def redact_pii(text: str, redaction_char: str = "*") -> str:
    """
    Redact personally identifiable information from text.

    Args:
        text: Text to redact PII from
        redaction_char: Character to use for redaction

    Returns:
        Text with PII redacted
    """
    if not isinstance(text, str):
        text = str(text)

    # Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]", text
    )

    # Phone numbers (various formats)
    phone_patterns = [
        r"\b\d{3}-\d{3}-\d{4}\b",  # 123-456-7890
        r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",  # (123) 456-7890
        r"\b\d{3}\.\d{3}\.\d{4}\b",  # 123.456.7890
        r"\b\d{10}\b",  # 1234567890
    ]

    for pattern in phone_patterns:
        text = re.sub(pattern, "[PHONE_REDACTED]", text)

    # Credit card numbers
    text = re.sub(
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD_REDACTED]", text
    )

    # Social Security Numbers
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", text)

    # IP addresses
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]", text)

    # URLs (optional - might be too aggressive)
    text = re.sub(r"https?://[^\s]+", "[URL_REDACTED]", text)

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

    pii_types = []

    # Check for email addresses
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
        pii_types.append("email")

    # Check for phone numbers
    phone_patterns = [
        r"\b\d{3}-\d{3}-\d{4}\b",
        r"\b\(\d{3}\)\s*\d{3}-\d{4}\b",
        r"\b\d{3}\.\d{3}\.\d{4}\b",
        r"\b\d{10}\b",
    ]

    for pattern in phone_patterns:
        if re.search(pattern, text):
            pii_types.append("phone")
            break

    # Check for credit card numbers
    if re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", text):
        pii_types.append("credit_card")

    # Check for SSN
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
        pii_types.append("ssn")

    # Check for IP addresses
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
        pii_types.append("ip_address")

    return pii_types


def redact_dict_values(
    data: dict[str, Any],
    keys_to_redact: Optional[list[str]] = None,
    redact_all_pii: bool = True,
) -> dict[str, Any]:
    """
    Redact PII from dictionary values.

    Args:
        data: Dictionary to redact
        keys_to_redact: Specific keys to redact (if None, redact based on key names)
        redact_all_pii: Whether to redact all detected PII

    Returns:
        Dictionary with PII redacted
    """
    if not isinstance(data, dict):
        return data

    # Default sensitive key patterns
    sensitive_keys = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "key",
        "auth",
        "email",
        "phone",
        "ssn",
        "credit_card",
        "card_number",
        "address",
        "street",
        "zip",
        "postal_code",
    }

    if keys_to_redact:
        sensitive_keys.update(keys_to_redact)

    redacted_data: dict[str, Any] = {}

    for key, value in data.items():
        key_lower = key.lower()

        # Check if key should be redacted
        should_redact_key = any(
            sensitive_key in key_lower for sensitive_key in sensitive_keys
        )

        if isinstance(value, dict):
            # Recursively redact nested dictionaries
            redacted_data[key] = redact_dict_values(
                value, keys_to_redact, redact_all_pii
            )
        elif isinstance(value, list):
            # Handle lists
            processed_list: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    processed_list.append(
                        redact_dict_values(item, keys_to_redact, redact_all_pii)
                    )
                elif should_redact_key or redact_all_pii:
                    processed_list.append(redact_pii(str(item)))
                else:
                    processed_list.append(item)
            redacted_data[key] = processed_list
        elif isinstance(value, str):
            if should_redact_key:
                redacted_data[key] = "[REDACTED]"
            elif redact_all_pii:
                redacted_data[key] = redact_pii(value)
            else:
                redacted_data[key] = value
        else:
            if should_redact_key:
                redacted_data[key] = "[REDACTED]"
            elif redact_all_pii and isinstance(value, (int, float)):
                # Don't redact numbers unless specifically requested
                redacted_data[key] = value
            else:
                redacted_data[key] = value

    return redacted_data


def is_sensitive_key(key: str) -> bool:
    """
    Check if a key name suggests sensitive data.

    Args:
        key: Key name to check

    Returns:
        True if key suggests sensitive data
    """
    sensitive_patterns = [
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "key",
        "auth",
        "email",
        "phone",
        "ssn",
        "credit",
        "card",
        "address",
        "street",
        "zip",
        "postal",
        "personal",
        "private",
        "confidential",
    ]

    key_lower = key.lower()
    return any(pattern in key_lower for pattern in sensitive_patterns)


def create_redaction_summary(original_text: str, redacted_text: str) -> dict[str, Any]:
    """
    Create a summary of redactions performed.

    Args:
        original_text: Original text before redaction
        redacted_text: Text after redaction

    Returns:
        Summary of redactions
    """
    pii_types = detect_pii_types(original_text)

    return {
        "pii_types_detected": pii_types,
        "redactions_made": len(pii_types) > 0,
        "original_length": len(original_text),
        "redacted_length": len(redacted_text),
        "reduction_ratio": (
            1 - (len(redacted_text) / len(original_text)) if original_text else 0
        ),
    }


class PiiPseudonymizer:
    """
    Deterministic pseudonymization of PII-like spans using HMAC-SHA256 + salt.

    Uses spaCy ``en_core_web_sm`` when importable and the model is available;
    otherwise relies on regex spans only. No mandatory dependency beyond the
    standard library.
    """

    _NER_LABELS = frozenset({"PERSON", "GPE", "ORG", "LOC"})

    # Aligned with ``redact_pii`` / ``detect_pii_types`` patterns in this module.
    _RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    _RE_PHONE = [
        re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
        re.compile(r"\b\(\d{3}\)\s*\d{3}-\d{4}\b"),
        re.compile(r"\b\d{3}\.\d{3}\.\d{4}\b"),
        re.compile(r"\b\d{10}\b"),
    ]
    _RE_CARD = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
    _RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    _RE_URL = re.compile(r"https?://[^\s]+")

    def __init__(self, salt: str) -> None:
        self._salt = salt
        self._salt_bytes = salt.encode("utf-8")
        self._nlp: Any = None
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self._nlp = None

    def _token(self, label: str, value: str) -> str:
        """NFC-normalize ``value``, HMAC-SHA256(salt, value), return LABEL_ + first 5 hex."""
        raw = value if isinstance(value, str) else str(value)
        normalized = unicodedata.normalize("NFC", raw)
        digest_hex = hmac.new(
            self._salt_bytes,
            normalized.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        prefix = label.strip().upper().replace(" ", "_")
        return f"{prefix}_{digest_hex[:5]}"

    def _regex_spans(self, text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for m in self._RE_EMAIL.finditer(text):
            spans.append((m.start(), m.end(), "EMAIL"))
        for cre in self._RE_PHONE:
            for m in cre.finditer(text):
                spans.append((m.start(), m.end(), "PHONE"))
        for m in self._RE_SSN.finditer(text):
            spans.append((m.start(), m.end(), "SSN"))
        for m in self._RE_CARD.finditer(text):
            spans.append((m.start(), m.end(), "CARD"))
        for m in self._RE_IP.finditer(text):
            spans.append((m.start(), m.end(), "IP"))
        for m in self._RE_URL.finditer(text):
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
