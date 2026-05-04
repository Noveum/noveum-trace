"""Tests for PiiPseudonymizer."""

import sys
import types

import pytest

from noveum_trace.utils.pii_redaction import PiiPseudonymizer


@pytest.fixture(autouse=True)
def stub_spacy_no_heavy_model(monkeypatch) -> None:
    """Avoid loading ``en_core_web_sm`` during tests (slow / may be absent)."""

    class FakeDoc:
        __slots__ = ("text", "ents")

        def __init__(self, text: str) -> None:
            self.text = text
            self.ents: list = []

    class FakeNlp:
        def __call__(self, text: str) -> FakeDoc:
            return FakeDoc(text)

    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = lambda *_a, **_k: FakeNlp()
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)


class TestPiiPseudonymizerToken:
    def test_token_deterministic(self) -> None:
        p = PiiPseudonymizer("fixed-salt")
        t1 = p._token("EMAIL", "user@example.com")
        t2 = p._token("EMAIL", "user@example.com")
        assert t1 == t2
        assert t1.startswith("EMAIL_")
        assert len(t1.split("_", 1)[1]) == 5

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


class TestPiiPseudonymizeNer:
    def test_spacy_person_span_replaced(self, monkeypatch) -> None:
        class FakeEnt:
            __slots__ = ("start_char", "end_char", "label_")

            def __init__(self, start: int, end: int, label: str) -> None:
                self.start_char = start
                self.end_char = end
                self.label_ = label

        class FakeDoc:
            __slots__ = ("ents",)

            def __init__(self) -> None:
                self.ents = [FakeEnt(0, 3, "PERSON")]

        class FakeNlp:
            def __call__(self, text: str) -> FakeDoc:
                return FakeDoc()

        fake_spacy = types.ModuleType("spacy")
        fake_spacy.load = lambda *_a, **_k: FakeNlp()
        monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
        p = PiiPseudonymizer("salt")
        assert p._nlp is not None
        out = p.pseudonymize("Bob went home")
        assert "Bob" not in out
        assert out.startswith("PERSON_")


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


def test_init_falls_back_when_spacy_load_fails(monkeypatch) -> None:
    """OSError from spacy.load leaves _nlp None; regex pseudonymization still works."""

    def boom(*_a, **_k):
        raise OSError("no model")

    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = boom
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    p = PiiPseudonymizer("salt")
    assert p._nlp is None
    assert "a@b.co" not in p.pseudonymize("a@b.co")
