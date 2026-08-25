"""Unit tests for the client-initiated NovaSynth call loop.

Driven through a fake ``httpx.Client`` and a fake clock, so the claim / arm /
ready / expired paths are exercised with no network and no real sleeping —
the tests assert on ``clock.sleeps`` rather than wall time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from noveum_trace import novasynth
from noveum_trace.novasynth import CallQueue
from noveum_trace.utils.exceptions import ConfigurationError

BASE = "https://api.noveum.ai/api"
INBOUND = f"{BASE}/v1/novasynth/inbound"
NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)


class _Resp:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class _FakeClient:
    """Records every get/post. Queued responses are consumed in order and the
    last one repeats, so a loop keeps seeing the same state."""

    calls: list = []
    get_resps: list = []
    post_resps: list = []
    get_exc = None
    post_exc = None

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @staticmethod
    def _next(queue, default):
        if not queue:
            return default
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def get(self, url, headers=None, params=None):
        _FakeClient.calls.append(
            {"method": "GET", "url": url, "headers": headers, "params": params}
        )
        if _FakeClient.get_exc is not None:
            exc, _FakeClient.get_exc = _FakeClient.get_exc, None  # raise once
            raise exc
        return _FakeClient._next(_FakeClient.get_resps, _Resp(200, {}))

    def post(self, url, headers=None, params=None, json=None):
        _FakeClient.calls.append(
            {
                "method": "POST",
                "url": url,
                "headers": headers,
                "params": params,
                "json": json,
            }
        )
        if _FakeClient.post_exc is not None:
            raise _FakeClient.post_exc
        return _FakeClient._next(_FakeClient.post_resps, _Resp(200, {}))


class _Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(novasynth, "time", c)
    # Wall clock for dial-window arithmetic advances with the fake clock.
    monkeypatch.setattr(
        novasynth, "_utcnow", lambda: NOW + timedelta(seconds=c.now - 1000.0)
    )
    return c


@pytest.fixture
def logs(caplog, monkeypatch):
    """The SDK sets ``propagate = False`` on the ``noveum_trace`` logger, which
    stops records before they reach caplog's root-attached handler."""
    monkeypatch.setattr(logging.getLogger("noveum_trace"), "propagate", True)
    caplog.set_level(logging.INFO, logger=novasynth._log.name)
    return caplog


def _patch(monkeypatch, get_resps=None, post_resps=None, get_exc=None, post_exc=None):
    # A bare dict is shorthand for a 200 carrying that JSON.
    def _wrap(r):
        return _Resp(200, r) if isinstance(r, dict) else r

    _FakeClient.calls = []
    _FakeClient.get_resps = [_wrap(r) for r in get_resps or []]
    _FakeClient.post_resps = [_wrap(r) for r in post_resps or []]
    _FakeClient.get_exc = get_exc
    _FakeClient.post_exc = post_exc
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(**kw))


def _gets():
    return [c for c in _FakeClient.calls if c["method"] == "GET"]


def _posts():
    return [c for c in _FakeClient.calls if c["method"] == "POST"]


def _view(run_id, status, **extra):
    """What the claim and poll endpoints both return."""
    view = {
        "success": True,
        "status": status,
        "runId": run_id,
        "phoneNumber": None,
        "dialWindowClosesAt": None,
        "profile": {"user_id": "U-42"},
        "personaName": "Asha Menon",
        "scenarioName": "Delayed refund",
    }
    view.update(extra)
    return view


def _ready(run_id, number="+918065481242", seconds=300, **extra):
    closes = (NOW + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    extra.setdefault("dialWindowClosesAt", closes)
    return _view(run_id, "ready", phoneNumber=number, **extra)


def _waiting(reason):
    return {
        "success": True,
        "status": "waiting",
        "retryAfterMs": 3000,
        "reason": reason,
    }


def _queue(run_ids, **kwargs):
    kwargs.setdefault("api_key", "k")
    kwargs.setdefault("base_url", BASE)
    return CallQueue(run_ids, **kwargs)


# --- construction -----------------------------------------------------------


def test_missing_api_key_raises_rather_than_faking_a_batch():
    with pytest.raises(ConfigurationError):
        CallQueue(["a"], api_key="", base_url=BASE)


@pytest.mark.parametrize("bad", ["", "   ", "api.noveum.ai/api", "https://", "ftp://x"])
def test_unusable_base_url_raises_rather_than_looping(bad):
    # An unusable URL would otherwise make every request throw, be swallowed,
    # and leave iter_calls() spinning forever.
    with pytest.raises(ConfigurationError):
        CallQueue(["a"], api_key="k", base_url=bad)


def test_credentials_fall_back_to_sdk_config(monkeypatch):
    class _Cfg:
        api_key = "from-config"
        endpoint = "https://example.test/api/"

    monkeypatch.setattr(novasynth, "get_config", lambda: _Cfg())
    q = CallQueue(["a"])
    assert q.api_key == "from-config"
    assert q.base_url == "https://example.test/api"  # trailing slash trimmed


# --- claim then poll, one run at a time ------------------------------------


def test_claim_then_poll_one_run_at_a_time(monkeypatch, clock):
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming"), _view("b", "arming")],
        get_resps=[_ready("a"), _ready("b")],
    )
    q = _queue(["a", "b"], organization_slug="acme")
    seen = [call.run_id for call in q.iter_calls()]
    assert seen == ["a", "b"]

    posts = _posts()
    assert [p["url"] for p in posts] == [
        f"{INBOUND}/runs/a/claim",
        f"{INBOUND}/runs/b/claim",
    ]
    assert posts[0]["headers"] == {"Authorization": "Bearer k"}
    assert posts[0]["params"] == {"organizationSlug": "acme"}
    assert posts[0]["json"]["idempotencyKey"]
    # b is not claimed until a has been handed out: one number at a time.
    assert [c["url"] for c in _FakeClient.calls] == [
        f"{INBOUND}/runs/a/claim",
        f"{INBOUND}/runs/a",
        f"{INBOUND}/runs/b/claim",
        f"{INBOUND}/runs/b",
    ]


def test_ready_run_carries_the_platform_profile(monkeypatch, clock):
    _patch(monkeypatch, post_resps=[_view("a", "arming")], get_resps=[_ready("a")])
    call = next(_queue(["a"]).iter_calls())
    assert call.dial_number == "+918065481242"
    assert call.profile == {"user_id": "U-42"}
    assert call.persona_name == "Asha Menon"
    assert call.scenario_name == "Delayed refund"


def test_claim_that_comes_back_ready_is_yielded_without_a_poll(monkeypatch, clock):
    # An idempotent replay of an earlier claim returns the current view.
    _patch(monkeypatch, post_resps=[_ready("a")])
    q = _queue(["a"])
    assert [c.run_id for c in q.iter_calls()] == ["a"]
    assert _gets() == []


def test_claim_is_retried_while_no_number_is_free(monkeypatch, clock, logs):
    _patch(
        monkeypatch,
        post_resps=[
            _waiting("no_number_available"),
            _waiting("no_number_available"),
            _view("a", "arming"),
        ],
        get_resps=[_ready("a")],
    )
    q = _queue(["a"])
    assert [c.run_id for c in q.iter_calls()] == ["a"]
    assert len(_posts()) == 3
    # Honours the platform's retryAfterMs, and the same key on every attempt.
    assert clock.sleeps[:2] == [3.0, 3.0]
    keys = {p["json"]["idempotencyKey"] for p in _posts()}
    assert len(keys) == 1
    assert "no_number_available" in logs.text


def test_run_reset_to_dormant_is_claimed_again(monkeypatch, clock):
    # A poll answering `waiting` means the claim no longer holds.
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming"), _view("a", "arming")],
        get_resps=[_waiting("not_claimed"), _ready("a")],
    )
    q = _queue(["a"])
    assert [c.run_id for c in q.iter_calls()] == ["a"]
    assert len(_posts()) == 2


def test_run_that_finishes_before_it_is_ready_is_skipped(monkeypatch, clock, logs):
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming"), _view("b", "arming")],
        get_resps=[_view("a", "expired"), _ready("b")],
    )
    q = _queue(["a", "b"])
    assert [c.run_id for c in q.iter_calls()] == ["b"]
    assert "run a expired undialled" in logs.text
    assert "run a skipped" in logs.text
    assert q.statuses["a"] == "expired"


def test_unknown_status_is_not_terminal(monkeypatch, clock):
    """An older SDK must keep polling a platform state it has never heard of."""
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming")],
        get_resps=[_view("a", "warming_up_v2"), _view("a", "completed")],
    )
    q = _queue(["a"])
    assert list(q.iter_calls()) == []
    assert len(_gets()) == 2  # it did not give up on the unknown status
    assert q.statuses == {"a": "completed"}


@pytest.mark.parametrize("number", [None, "", 918065481242, {"e164": "+91806"}])
def test_ready_run_without_a_usable_number_is_never_handed_over(
    monkeypatch, clock, logs, number
):
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming")],
        get_resps=[_ready("a", number=number)],
    )
    assert list(_queue(["a"]).iter_calls()) == []
    assert "no usable" in logs.text


# --- waiting for the call to finish ----------------------------------------


def test_wait_until_finished_returns_the_terminal_status(monkeypatch, clock):
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming")],
        get_resps=[_ready("a"), _view("a", "in_progress"), _view("a", "completed")],
    )
    q = _queue(["a"])
    call = next(q.iter_calls())
    assert call.wait_until_finished() == "completed"
    assert q.summary() == {"completed": 1}


def test_poll_backs_off_once_the_call_is_up(monkeypatch, clock):
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming")],
        get_resps=[
            _ready("a"),
            _ready("a"),  # dialler still placing the call
            _view("a", "in_progress"),
            _view("a", "completed"),
        ],
    )
    q = _queue(["a"])
    call = next(q.iter_calls())
    call.wait_until_finished()
    # arming poll, ready-but-not-yet-connected poll, then the in-call back-off
    assert clock.sleeps == [3.0, 3.0, 10.0]


# --- the dial window --------------------------------------------------------


def test_seconds_remaining_comes_from_dial_window_closes_at(monkeypatch, clock):
    _patch(monkeypatch, post_resps=[_ready("a", seconds=180)])
    call = next(_queue(["a"]).iter_calls())
    assert call.dial_window_closes_at == NOW + timedelta(seconds=180)
    assert call.seconds_remaining == 180.0
    clock.sleep(30)
    assert call.seconds_remaining == 150.0
    clock.sleep(1000)
    assert call.seconds_remaining == 0.0


@pytest.mark.parametrize("raw", [None, "", "not a date", 1234])
def test_missing_or_bad_deadline_is_unbounded_not_zero(monkeypatch, clock, raw):
    # Zero would tell the caller not to bother dialling a run that is ready.
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming")],
        get_resps=[_ready("a", dialWindowClosesAt=raw)],
    )
    call = next(_queue(["a"]).iter_calls())
    assert call.dial_window_closes_at is None
    assert call.seconds_remaining == float("inf")


# --- failures ---------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_rejection_stops_the_loop(monkeypatch, clock, status):
    # Polling cannot fix a bad key or a wrong endpoint, so the loop must fail
    # loudly instead of spinning on it forever.
    _patch(monkeypatch, post_resps=[_Resp(status)])
    with pytest.raises(ConfigurationError):
        list(_queue(["a"]).iter_calls())
    assert len(_FakeClient.calls) == 1  # it did not try again


def test_400_on_claim_surfaces_the_platform_message(monkeypatch, clock):
    _patch(
        monkeypatch,
        post_resps=[_Resp(400, {"message": "Run is not an inbound run"})],
    )
    with pytest.raises(ConfigurationError, match="not an inbound run"):
        list(_queue(["a"]).iter_calls())


def test_transient_failures_are_not_fatal(monkeypatch, clock, logs):
    _patch(
        monkeypatch,
        post_resps=[_Resp(503), _Resp(429)],
        get_resps=[
            _waiting("not_claimed"),  # the failed claim never took: claim again
            _Resp(200, ValueError("bad json")),
            _Resp(200, ["not", "a", "dict"]),
            _view("a", "completed"),
        ],
    )
    q = _queue(["a"])
    assert list(q.iter_calls()) == []
    assert q.statuses == {"a": "completed"}
    assert "HTTP 503" in logs.text
    assert "HTTP 429" in logs.text
    assert "bad JSON" in logs.text


def test_wait_gives_up_after_max_wait_seconds(monkeypatch, clock):
    # A run stuck in a non-terminal status must not poll forever.
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming")],
        get_resps=[_ready("a"), _view("a", "in_progress")],
    )
    q = _queue(["a"], max_wait_seconds=30.0)
    call = next(q.iter_calls())
    with pytest.raises(TimeoutError, match="max_wait_seconds"):
        call.wait_until_finished()
    assert clock.sleeps  # it did wait before giving up


def test_server_retry_after_is_clamped(monkeypatch, clock):
    # A bad retryAfterMs (an hour, say) must not park the loop.
    _patch(
        monkeypatch,
        post_resps=[
            dict(_waiting("no_number_available"), retryAfterMs=3_600_000),
            _view("a", "arming"),
        ],
        get_resps=[_ready("a")],
    )
    assert [c.run_id for c in _queue(["a"]).iter_calls()] == ["a"]
    assert clock.sleeps[0] == 60.0


def test_network_error_is_not_fatal(monkeypatch, clock, logs):
    _patch(
        monkeypatch,
        post_resps=[_view("a", "arming")],
        get_resps=[_view("a", "completed")],
        get_exc=RuntimeError("connection reset"),  # first poll only
    )
    q = _queue(["a"])
    assert list(q.iter_calls()) == []
    assert len(_gets()) == 2  # the failed poll was retried on the next tick
    assert q.statuses == {"a": "completed"}
    assert "connection reset" in logs.text


# --- accounting -------------------------------------------------------------


def test_summary_accounts_for_every_run_in_the_batch(monkeypatch, clock):
    _patch(
        monkeypatch,
        post_resps=[_view("a", "completed"), _view("b", "failed"), _ready("c")],
    )
    q = _queue(["a", "b", "c", "d"])
    gen = q.iter_calls()
    assert next(gen).run_id == "c"  # a and b finished before being handed out
    gen.close()  # d is never reached: it must still be in the count
    assert q.summary() == {"completed": 1, "failed": 1, "ready": 1, "waiting": 1}
    assert sum(q.summary().values()) == len(q.run_ids)
