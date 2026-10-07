"""Tests for PiiPseudonymizer."""

import re

import pytest

from noveum_trace.utils.pii_redaction import (
    TOKEN_SUFFIX_LENGTH,
    PiiPseudonymizer,
    compile_custom_patterns,
    detect_pii_types,
    redact_pii,
)


class TestPiiPseudonymizerToken:
    def test_token_deterministic(self) -> None:
        p = PiiPseudonymizer("fixed-salt")
        t1 = p._token("EMAIL", "user@example.com")
        t2 = p._token("EMAIL", "user@example.com")
        assert t1 == t2
        assert t1.startswith("EMAIL_")
        assert len(t1.split("_", 1)[1]) == TOKEN_SUFFIX_LENGTH

    def test_token_salt_changes_output(self) -> None:
        a = PiiPseudonymizer("salt-a")._token("EMAIL", "user@example.com")
        b = PiiPseudonymizer("salt-b")._token("EMAIL", "user@example.com")
        assert a != b

    def test_token_nfc_equivalence(self) -> None:
        # Precomposed vs decomposed e — same NFC, same token
        p = PiiPseudonymizer("s")
        nfc = "caf\u00e9@x.com"
        nfd = "cafe\u0301@x.com"
        assert p._token("EMAIL", nfc) == p._token("EMAIL", nfd)


class TestPiiPseudonymizerSpans:
    def test_non_overlapping_longest_wins(self) -> None:
        # Inner span shorter; only longer kept
        spans = [(0, 5, "A"), (2, 8, "B")]
        got = PiiPseudonymizer._non_overlapping_longest_first(spans)
        assert got == [(2, 8, "B")]

    def test_non_overlapping_both_kept(self) -> None:
        spans = [(0, 3, "A"), (5, 8, "B")]
        got = PiiPseudonymizer._non_overlapping_longest_first(spans)
        assert len(got) == 2

    def test_tie_length_prefers_left(self) -> None:
        spans = [
            (2, 5, "A"),
            (0, 3, "B"),
        ]  # both length 3; process order by (-len, start)
        got = PiiPseudonymizer._non_overlapping_longest_first(spans)
        assert len(got) == 1
        assert got[0][0] == 0


class TestPiiPseudonymizeRegex:
    def test_email_replaced(self) -> None:
        p = PiiPseudonymizer("salt")
        out = p.pseudonymize("mail user@example.com end")
        assert "user@example.com" not in out
        assert "mail " in out and " end" in out
        assert "EMAIL_" in out

    def test_phone_ssn_card_ip_url(self) -> None:
        p = PiiPseudonymizer("salt")
        raw = (
            "p 123-456-7890 s 123-45-6789 "
            "c 4111 1111 1111 1111 i 192.168.1.1 u https://a.com/x"
        )
        out = p.pseudonymize(raw)
        assert "123-456-7890" not in out
        assert "123-45-6789" not in out
        assert "4111 1111 1111 1111" not in out
        assert "192.168.1.1" not in out
        assert "https://a.com/x" not in out
        for label in ("PHONE_", "SSN_", "CARD_", "IP_", "URL_"):
            assert label in out

    def test_url_trailing_punct_same_pseudonym(self) -> None:
        """Canonical URL (no trailing sentence punctuation) used for hashing."""
        p = PiiPseudonymizer("salt")
        with_dot = p.pseudonymize("u https://ex.com/a.")
        no_dot = p.pseudonymize("u https://ex.com/a")
        tok = re.compile(r"URL_[a-f0-9]+")
        assert tok.search(with_dot) and tok.search(no_dot)
        assert tok.search(with_dot).group(0) == tok.search(no_dot).group(0)
        assert with_dot.endswith(".")

    def test_right_to_left_indices(self) -> None:
        p = PiiPseudonymizer("salt")
        out = p.pseudonymize("a@b.co c@d.co")
        assert "a@b.co" not in out and "c@d.co" not in out
        assert out.count("EMAIL_") == 2


class TestPiiPseudonymizeDict:
    def test_nested(self) -> None:
        p = PiiPseudonymizer("salt")
        data = {"x": "a@b.co", "y": [{"z": "c@d.co"}], "n": 1}
        out = p.pseudonymize_dict(data)
        assert out["n"] == 1
        assert "a@b.co" not in out["x"]
        assert "c@d.co" not in out["y"][0]["z"]


def _labels(text: str, **kw) -> list[str]:
    """Labels of the pseudonym tokens produced for ``text``, left to right."""
    out = PiiPseudonymizer("salt", **kw).pseudonymize(text)
    return re.findall(rf"([A-Z_]+?)_[a-f0-9]{{{TOKEN_SUFFIX_LENGTH}}}", out)


class TestEmiratesId:
    @pytest.mark.parametrize(
        "eid",
        [
            "784-1990-0000001-0",
            "784199000000010",
            "784-1985-0000012-4",
            "784198500000124",
            "784-2001-0000123-7",
            "784200100001237",
        ],
    )
    def test_valid_ids(self, eid: str) -> None:
        assert _labels(f"EID {eid} ok") == ["EMIRATES_ID"]

    def test_bare_bad_check_digit_is_not_emirates_id(self) -> None:
        # Falls through to the generic account-number rule instead.
        assert _labels("784199000000011") == ["BANK_ACCOUNT"]

    def test_hyphenated_bad_check_digit_still_redacted(self) -> None:
        assert _labels("784-1990-0000001-1") == ["EMIRATES_ID"]


class TestIban:
    @pytest.mark.parametrize(
        "iban",
        [
            "AE07 0331 2345 6789 0123 456",
            "AE070331234567890123456",
            "SA03 8000 0000 6080 1016 7519",
            "GB29 NWBK 6016 1331 9268 19",
            "TR33 0006 1005 1978 6457 8413 26",
        ],
    )
    def test_valid(self, iban: str) -> None:
        out = PiiPseudonymizer("salt").pseudonymize(f"iban {iban} THEN more")
        assert iban not in out
        assert out.startswith("iban IBAN_") and out.endswith(" THEN more")

    def test_bad_checksum_not_iban(self) -> None:
        assert "IBAN" not in _labels("AE08 0331 2345 6789 0123 456")


class TestPhones:
    @pytest.mark.parametrize(
        "phone",
        [
            "+971 50 123 4567",
            "+971501234567",
            "00971 50 123 4567",
            "050 123 4567",
            "0501234567",
            "+966 55 123 4567",
            "+974 3312 3456",
            "+20 10 1234 5678",
            "+90 532 123 45 67",
            "0532 123 45 67",
            "+44 20 7946 0958",
            "+44 (0)20 7946 0958",
            "07946 123456",
            "+1-415-555-0199",
            "+91 98765 43210",
            "123-456-7890",
        ],
    )
    def test_detected(self, phone: str) -> None:
        out = PiiPseudonymizer("salt").pseudonymize(f"call {phone} now")
        assert phone not in out
        assert re.fullmatch(r"call PHONE_[a-f0-9]+ now", out), out

    def test_trailing_number_not_swallowed(self) -> None:
        out = PiiPseudonymizer("salt").pseudonymize("+971 50 123 4567, 2024 x")
        assert re.fullmatch(r"PHONE_[a-f0-9]+, 2024 x", out), out

    def test_invalid_numbers_ignored(self) -> None:
        # libphonenumber validity check: wrong lengths / unassigned prefixes
        assert _labels("order 1234567 qty +971 12") == []

    def test_short_plus_number_ignored(self) -> None:
        assert _labels("score +12 34") == []


class TestBankAccount:
    @pytest.mark.parametrize("n", ["98765432109", "1234567890123", "1234567890123456"])
    def test_11_to_16_digits(self, n: str) -> None:
        assert _labels(f"acct {n}") == (["CARD"] if len(n) == 16 else ["BANK_ACCOUNT"])

    def test_17_digits_and_decimals_ignored(self) -> None:
        assert _labels("12345678901234567") == []
        assert _labels("3.14159265358979") == []


class TestCustomPatterns:
    def test_list_uses_custom_label(self) -> None:
        assert _labels("emp EMP-12345", custom_patterns=[r"EMP-\d+"]) == ["CUSTOM"]

    def test_dict_labels(self) -> None:
        assert _labels(
            "emp EMP-12345 order ORD/99",
            custom_patterns={"employee id": r"EMP-\d+", "ORDER": r"ORD/\d+"},
        ) == ["EMPLOYEE_ID", "ORDER"]

    def test_custom_wins_tie_with_builtin(self) -> None:
        assert _labels("12345678901", custom_patterns={"MEMBER": r"\d{11}"}) == [
            "MEMBER"
        ]

    def test_shorter_custom_beats_longer_builtin(self) -> None:
        # Built-in EMAIL would cover the whole address; the custom match wins.
        out = PiiPseudonymizer("salt", custom_patterns={"USER": r"jdoe"}).pseudonymize(
            "mail jdoe@corp.com"
        )
        assert re.fullmatch(r"mail USER_[a-f0-9]+@corp\.com", out), out

    def test_redact_pii_custom_precedence(self) -> None:
        assert redact_pii("x 784-1990-0000001-0", custom_patterns=[r"1990"]) == (
            "x 784-****-0000001-0"
        )

    def test_invalid_regex_raises_on_compile(self) -> None:
        with pytest.raises(ValueError, match="Invalid custom redaction regex"):
            compile_custom_patterns(["("])

    def test_invalid_regex_ignored_by_pseudonymizer(self) -> None:
        # Never crash the host app: bad custom patterns are dropped, built-ins still run.
        assert _labels("a@b.co", custom_patterns=["("]) == ["EMAIL"]


class TestSharedPatterns:
    def test_redact_pii_masks_new_types(self) -> None:
        text = "id 784-1990-0000001-0 tel +971 50 123 4567 end."
        out = redact_pii(text)
        assert out == "id " + "*" * 18 + " tel " + "*" * 16 + " end."

    def test_redact_pii_custom(self) -> None:
        assert redact_pii("EMP-1", custom_patterns=[r"EMP-\d"]) == "*****"

    def test_detect_pii_types(self) -> None:
        got = detect_pii_types(
            "x@y.co 784199000000010 AE070331234567890123456 +971501234567 "
            "12345678901"
        )
        assert set(got) >= {"email", "emirates_id", "iban", "phone", "bank_account"}

    def test_uuid_with_all_digit_segment_untouched(self) -> None:
        # trace/span ids must survive pseudonymization or traces lose linkage
        uid = "550e8400-e29b-41d4-0512-446655440000"
        assert PiiPseudonymizer("salt").pseudonymize(uid) == uid


def test_config_rejects_invalid_custom_pattern() -> None:
    from noveum_trace.core.config import Config, SecurityConfig
    from noveum_trace.utils.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="custom_redaction_patterns"):
        Config(security=SecurityConfig(custom_redaction_patterns={"BAD": "("}))
