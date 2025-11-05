"""Functional tests validating LangChain trace emission."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Generator
from uuid import uuid4

import pytest

import noveum_trace
from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler
from noveum_trace.transport.http_transport import HttpTransport


def _ensure_env() -> None:
    missing_vars = [
        key
        for key in [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "NOVEUM_API_KEY",
            "NOVEUM_PROJECT",
        ]
        if not os.environ.get(key)
    ]

    if missing_vars:
        missing = ", ".join(missing_vars)
        raise RuntimeError(
            "Required environment variables not set for LangChain functional test: "
            f"{missing}."
        )


@pytest.fixture
def trace_export_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[list[dict], None, None]:
    """Capture traces passed to the HTTP transport export call."""

    captured: list[dict] = []
    original_export = HttpTransport.export_trace

    def capture_export(self: HttpTransport, trace) -> dict:
        captured.append(trace.to_dict())
        return {"success": True, "trace_id": trace.trace_id}

    monkeypatch.setattr(HttpTransport, "export_trace", capture_export)

    try:
        yield captured
    finally:
        monkeypatch.setattr(HttpTransport, "export_trace", original_export)


def _simulate_llm_interaction(
    handler: NoveumTraceCallbackHandler,
    *,
    model_name: str,
    provider_hint: str,
    display_name: str,
    temperature: float,
    response_text: str,
) -> None:
    """Fire synthetic LangChain events to exercise the callback handler."""

    run_id = uuid4()
    prompts = ["Human: What is the capital of France? Answer in one sentence."]

    serialized = {
        "name": display_name,
        "kwargs": {"model": model_name},
        "id": ["langchain", "chat_models", provider_hint, display_name],
    }

    handler.on_llm_start(
        serialized,
        prompts,
        run_id,
        invocation_params={"model": model_name, "temperature": temperature},
        temperature=temperature,
        batch_size=1,
    )

    response = SimpleNamespace(
        generations=[[SimpleNamespace(text=response_text)]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 18,
                "completion_tokens": 16,
                "total_tokens": 34,
            },
            "finish_reason": "stop",
        },
    )

    handler.on_llm_end(response, run_id)


@pytest.mark.disable_transport_mocking
def test_langchain_llm_traces_include_temperature_and_latency(trace_export_capture):
    """Ensure LangChain callback emits traces with temperature and latency."""

    _ensure_env()

    if noveum_trace.is_initialized():
        noveum_trace.shutdown()

    noveum_trace.init(
        project=os.environ["NOVEUM_PROJECT"],
        api_key=os.environ["NOVEUM_API_KEY"],
        environment=os.environ.get("NOVEUM_ENVIRONMENT"),
        transport_config={"batch_size": 1, "batch_timeout": 0.1},
    )

    handler = NoveumTraceCallbackHandler()

    try:
        _simulate_llm_interaction(
            handler,
            model_name="gpt-5",
            provider_hint="openai",
            display_name="ChatOpenAI",
            temperature=0.3,
            response_text="The capital of France is Paris.",
        )

        _simulate_llm_interaction(
            handler,
            model_name="gemini-2.0-flash",
            provider_hint="google",
            display_name="ChatGoogleGenerativeAI",
            temperature=0.4,
            response_text="Paris is the capital of France.",
        )

        assert len(trace_export_capture) == 2, "Expected two exported traces"

        first_trace = trace_export_capture[0]
        first_span_attrs = first_trace["spans"][0]["attributes"]
        assert first_span_attrs["llm.provider"].lower() == "openai"
        assert first_span_attrs["llm.model"] == "gpt-5"
        assert first_span_attrs["llm.input.temperature"] == 0.3
        assert first_span_attrs["llm.input.prompt_count"] == 1
        assert first_span_attrs["llm.output.response"][0].startswith(
            "The capital of France"
        )
        assert "llm.latency_ms" in first_span_attrs
        assert first_span_attrs["llm.latency_ms"] >= 0

        second_trace = trace_export_capture[1]
        second_span_attrs = second_trace["spans"][0]["attributes"]
        assert second_span_attrs["llm.provider"].lower() == "google"
        assert second_span_attrs["llm.model"] == "gemini-2.0-flash"
        assert second_span_attrs["llm.input.temperature"] == 0.4
        assert second_span_attrs["llm.output.response"][0].startswith("Paris is")

    finally:
        noveum_trace.shutdown()
