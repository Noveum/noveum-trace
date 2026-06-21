"""Unit tests for GuardAPIClient — in-memory stub correctness and thread safety."""

import threading
import uuid

import pytest

from noveum_trace.guard.api_client import GuardAPIClient

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


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------


class TestGetState:
    def test_get_state_returns_zero_for_new_project(self):
        api = GuardAPIClient()
        state = api.get_state("new-proj")
        assert state == {"spend": 0.0}

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
