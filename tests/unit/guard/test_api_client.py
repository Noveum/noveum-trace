"""Unit tests for GuardAPIClient — in-memory stub correctness and thread safety."""

import threading
import uuid

import httpx
import pytest

from noveum_trace.guard.api_client import _MAX_BLOCKED_EVENTS, GuardAPIClient
from noveum_trace.guard.exceptions import GuardBackendUnavailable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# reserve()
# ---------------------------------------------------------------------------


class TestReserve:
    def test_first_reservation_succeeds_when_under_cap(self):
        api = GuardAPIClient()
        result = api.reserve(_call_id(), "proj", reserved_usd=50.0, max_usd=100.0)
        assert result.admitted is True
        assert result.current_spend_usd == 50.0

    def test_second_reservation_fails_when_over_cap(self):
        """reserve(50) succeeds; reserve(60) on same project fails — 50+60 > 100."""
        api = GuardAPIClient()
        r1 = api.reserve(_call_id(), "proj", reserved_usd=50.0, max_usd=100.0)
        r2 = api.reserve(_call_id(), "proj", reserved_usd=60.0, max_usd=100.0)
        assert r1.admitted is True
        assert r2.admitted is False

    def test_second_reservation_fails_reports_current_spend(self):
        api = GuardAPIClient()
        api.reserve(_call_id(), "proj", reserved_usd=50.0, max_usd=100.0)
        r2 = api.reserve(_call_id(), "proj", reserved_usd=60.0, max_usd=100.0)
        assert r2.current_spend_usd == 50.0  # spend as seen at rejection

    def test_exact_cap_is_admitted(self):
        """spend + reserved == max_usd should admit (boundary: >, not >=)."""
        api = GuardAPIClient()
        result = api.reserve(_call_id(), "proj", reserved_usd=100.0, max_usd=100.0)
        assert result.admitted is True

    def test_one_over_cap_is_rejected(self):
        api = GuardAPIClient()
        result = api.reserve(_call_id(), "proj", reserved_usd=100.01, max_usd=100.0)
        assert result.admitted is False

    def test_separate_projects_are_independent(self):
        api = GuardAPIClient()
        r1 = api.reserve(_call_id(), "proj-a", reserved_usd=90.0, max_usd=100.0)
        r2 = api.reserve(_call_id(), "proj-b", reserved_usd=90.0, max_usd=100.0)
        assert r1.admitted is True
        assert r2.admitted is True

    def test_spend_accumulates_across_reservations(self):
        api = GuardAPIClient()
        api.reserve(_call_id(), "proj", reserved_usd=30.0, max_usd=100.0)
        api.reserve(_call_id(), "proj", reserved_usd=30.0, max_usd=100.0)
        assert api.current_spend("proj") == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# reconcile() — reserve then release restores budget
# ---------------------------------------------------------------------------


class TestReconcile:
    def test_release_full_reservation_restores_budget(self):
        """reserve(50) then reconcile(50) → spend back to 0."""
        api = GuardAPIClient()
        call_id = _call_id()
        api.reserve(call_id, "proj", reserved_usd=50.0, max_usd=100.0)
        api.reconcile(call_id, "proj", unconsumed_usd=50.0)
        assert api.current_spend("proj") == pytest.approx(0.0)

    def test_release_partial_returns_only_unconsumed(self):
        """reserve(50), actual cost 30 → reconcile(20) → spend = 30."""
        api = GuardAPIClient()
        call_id = _call_id()
        api.reserve(call_id, "proj", reserved_usd=50.0, max_usd=100.0)
        api.reconcile(call_id, "proj", unconsumed_usd=20.0)
        assert api.current_spend("proj") == pytest.approx(30.0)

    def test_reconcile_clears_inflight_entry(self):
        api = GuardAPIClient()
        call_id = _call_id()
        api.reserve(call_id, "proj", reserved_usd=50.0, max_usd=100.0)
        assert api.inflight_count() == 1
        api.reconcile(call_id, "proj", unconsumed_usd=50.0)
        assert api.inflight_count() == 0

    def test_reconcile_does_not_go_below_zero(self):
        api = GuardAPIClient()
        call_id = _call_id()
        api.reserve(call_id, "proj", reserved_usd=10.0, max_usd=100.0)
        api.reconcile(call_id, "proj", unconsumed_usd=9999.0)  # absurdly large
        assert api.current_spend("proj") == 0.0

    def test_budget_available_again_after_release(self):
        """After release, a previously rejected reservation now admits."""
        api = GuardAPIClient()
        call_id = _call_id()
        api.reserve(call_id, "proj", reserved_usd=80.0, max_usd=100.0)
        rejected = api.reserve(_call_id(), "proj", reserved_usd=30.0, max_usd=100.0)
        assert rejected.admitted is False

        api.reconcile(call_id, "proj", unconsumed_usd=80.0)  # release
        admitted = api.reserve(_call_id(), "proj", reserved_usd=30.0, max_usd=100.0)
        assert admitted.admitted is True


# ---------------------------------------------------------------------------
# report_usage() — non-strict path
# ---------------------------------------------------------------------------


class TestReportUsage:
    def test_report_usage_accumulates_spend(self):
        api = GuardAPIClient()
        api.report_usage(_call_id(), "proj", actual_usd=5.0, model="gpt-4o")
        api.report_usage(_call_id(), "proj", actual_usd=3.0, model="gpt-4o")
        assert api.current_spend("proj") == pytest.approx(8.0)

    def test_report_usage_does_not_touch_inflight(self):
        api = GuardAPIClient()
        api.report_usage(_call_id(), "proj", actual_usd=5.0, model="gpt-4o")
        assert api.inflight_count() == 0

    def test_report_usage_bumps_rate_counters(self):
        api = GuardAPIClient()
        api.report_usage(
            _call_id(),
            "proj",
            actual_usd=1.0,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        rate = api.current_rate("proj")
        assert rate["requests_1m"] == 1
        assert rate["requests_1h"] == 1
        assert rate["requests_1d"] == 1
        assert rate["tokens_1m"] == 150
        assert rate["tokens_1h"] == 150
        assert rate["tokens_1d"] == 150

    def test_report_usage_accumulates_rate_counters_across_calls(self):
        api = GuardAPIClient()
        api.report_usage(
            _call_id(),
            "proj",
            actual_usd=1.0,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        api.report_usage(
            _call_id(),
            "proj",
            actual_usd=1.0,
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
        )
        rate = api.current_rate("proj")
        assert rate["requests_1m"] == 2
        assert rate["tokens_1m"] == 165

    def test_report_usage_dedups_by_call_id(self):
        """Two policies reporting the same call must not double count."""
        api = GuardAPIClient()
        call_id = _call_id()
        api.report_usage(
            call_id,
            "proj",
            actual_usd=1.0,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        api.report_usage(
            call_id,
            "proj",
            actual_usd=1.0,
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        assert api.current_spend("proj") == pytest.approx(1.0)
        assert api.current_rate("proj")["requests_1m"] == 1


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------


class TestGetState:
    def test_get_state_returns_zero_for_new_project(self):
        api = GuardAPIClient()
        state = api.get_state("new-proj")
        assert state == {"spend": 0.0, "rate": {}}

    def test_get_state_reflects_current_spend(self):
        api = GuardAPIClient()
        api.reserve(_call_id(), "proj", reserved_usd=42.0, max_usd=200.0)
        state = api.get_state("proj")
        assert state["spend"] == pytest.approx(42.0)

    def test_get_state_returns_copy(self):
        api = GuardAPIClient()
        state = api.get_state("proj")
        state["spend"] = 9999.0  # mutate the returned dict
        assert api.current_spend("proj") == 0.0  # original unaffected


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_state(self):
        api = GuardAPIClient()
        api.reserve(_call_id(), "proj", reserved_usd=50.0, max_usd=100.0)
        api.set_policy_config("proj", {"cost_cap_usd": 200.0})
        api.reset()
        assert api.current_spend("proj") == 0.0
        assert api.inflight_count() == 0
        assert api.get_policy_config("proj") == {}


# ---------------------------------------------------------------------------
# Thread safety — 100 concurrent reservations
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


class _FakeHTTPClient:
    """Stand-in for httpx.Client, injected via monkeypatch."""

    def __init__(self, *, response=None, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url, headers=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


# ---------------------------------------------------------------------------
# fetch_remote_policies() — stub mode vs. real-backend failure handling
# ---------------------------------------------------------------------------


class TestFetchRemotePolicies:
    def test_no_api_key_returns_empty_list(self):
        """Stub mode (no api_key) with nothing stored — legitimate silent case."""
        api = GuardAPIClient()
        assert api.fetch_remote_policies("proj") == []

    def test_no_api_key_returns_stored_config(self):
        api = GuardAPIClient()
        api.set_policy_config("proj", {"type": "cost_cap", "max_usd": 10.0})
        result = api.fetch_remote_policies("proj")
        assert result == [{"type": "cost_cap", "max_usd": 10.0}]

    def test_real_backend_success_returns_policies(self, monkeypatch):
        api = GuardAPIClient(api_key="secret", base_url="https://api.noveum.ai")
        fake_response = _FakeHTTPResponse(
            200, {"policies": [{"type": "cost_cap", "max_usd": 5.0}]}
        )
        monkeypatch.setattr(
            httpx, "Client", lambda **kw: _FakeHTTPClient(response=fake_response)
        )
        result = api.fetch_remote_policies("proj")
        assert result == [{"type": "cost_cap", "max_usd": 5.0}]

    def test_real_backend_non_200_raises_backend_unavailable(self, monkeypatch):
        api = GuardAPIClient(api_key="secret", base_url="https://api.noveum.ai")
        fake_response = _FakeHTTPResponse(500)
        monkeypatch.setattr(
            httpx, "Client", lambda **kw: _FakeHTTPClient(response=fake_response)
        )
        with pytest.raises(GuardBackendUnavailable):
            api.fetch_remote_policies("proj")

    def test_real_backend_network_error_raises_backend_unavailable(self, monkeypatch):
        api = GuardAPIClient(api_key="secret", base_url="https://api.noveum.ai")
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kw: _FakeHTTPClient(raise_exc=httpx.ConnectError("boom")),
        )
        with pytest.raises(GuardBackendUnavailable):
            api.fetch_remote_policies("proj")


class TestConcurrency:
    def test_only_one_of_100_concurrent_reservations_admitted(self):
        """DarGlobal scenario: $100 cap, 100 callers each reserving $99."""
        api = GuardAPIClient()
        admitted = []
        errors = []

        def attempt():
            try:
                result = api.reserve(
                    _call_id(), "proj", reserved_usd=99.0, max_usd=100.0
                )
                if result.admitted:
                    admitted.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=attempt) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected exceptions: {errors}"
        assert len(admitted) == 1, f"Expected 1 admit, got {len(admitted)}"
        assert api.current_spend("proj") <= 100.0

    def test_spend_never_exceeds_cap_under_concurrency(self):
        """Mixed small reservations — spend must never exceed cap."""
        api = GuardAPIClient()
        cap = 100.0

        def attempt():
            api.reserve(_call_id(), "proj", reserved_usd=11.0, max_usd=cap)

        threads = [threading.Thread(target=attempt) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert api.current_spend("proj") <= cap


# ---------------------------------------------------------------------------
# report_blocked()
# ---------------------------------------------------------------------------


class TestReportBlocked:
    def test_records_the_event(self):
        api = GuardAPIClient()
        cid = _call_id()
        api.report_blocked(
            cid, "proj", "gpt-4o", "COST_CAP", policy_id="pol_1", reason="cap hit"
        )
        events = api.blocked_events("proj")
        assert len(events) == 1
        assert events[0] == {
            "call_id": cid,
            "model": "gpt-4o",
            "blocked_by": "COST_CAP",
            "policy_id": "pol_1",
            "reason": "cap hit",
        }

    def test_invalid_blocked_by_is_ignored(self):
        api = GuardAPIClient()
        api.report_blocked(_call_id(), "proj", "gpt-4o", "NOT_A_LIMIT")
        assert api.blocked_events("proj") == []

    def test_blocked_call_is_not_metered(self):
        """The call never ran, so it must not count toward spend or rate."""
        api = GuardAPIClient()
        api.report_blocked(_call_id(), "proj", "gpt-4o", "RATE_LIMIT")
        assert api.current_spend("proj") == 0.0
        assert api.current_rate("proj") == {}

    def test_buffer_is_bounded(self):
        """A long-running process under sustained blocking must not grow the
        inspection buffer without bound."""
        api = GuardAPIClient()
        for _ in range(_MAX_BLOCKED_EVENTS + 250):
            api.report_blocked(_call_id(), "proj", "gpt-4o", "RATE_LIMIT")
        assert len(api.blocked_events("proj")) == _MAX_BLOCKED_EVENTS

    def test_bounded_buffer_drops_the_oldest(self):
        api = GuardAPIClient()
        for i in range(_MAX_BLOCKED_EVENTS + 1):
            api.report_blocked(str(i), "proj", "gpt-4o", "RATE_LIMIT")
        events = api.blocked_events("proj")
        assert events[0]["call_id"] == "1"  # "0" was evicted
        assert events[-1]["call_id"] == str(_MAX_BLOCKED_EVENTS)

    def test_reset_clears_blocked_events(self):
        api = GuardAPIClient()
        api.report_blocked(_call_id(), "proj", "gpt-4o", "COST_CAP")
        api.reset()
        assert api.blocked_events("proj") == []
