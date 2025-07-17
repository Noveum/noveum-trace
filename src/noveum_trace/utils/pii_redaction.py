"""
PII redaction utilities for Noveum Trace SDK.

This module provides functions to detect and redact personally
identifiable information from trace data.
"""

import re
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
