"""
Span comparison harness: Pipecat-native OpenTelemetry vs noveum_trace.

Runs ONE real OpenAI LLM turn through a minimal Pipecat pipeline with BOTH
tracing systems attached, and writes each system's spans to a local file:

  - otel_spans.jsonl    Pipecat's built-in OpenTelemetry output (enable_tracing=True),
                        captured via a custom file SpanExporter.
  - noveum_spans.jsonl  noveum_trace output (NoveumTraceObserver), captured by
                        monkeypatching the client transport so nothing is sent
                        over the network (no Noveum API key required).

No transport / STT / TTS / audio — only the LLM runs, so the same logical
operation appears in both outputs and can be compared directly.

Run:  python3 docs/examples/compare_otel_vs_noveum.py
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace as NS

from dotenv import load_dotenv
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.tracing.setup import setup_tracing

import noveum_trace
from noveum_trace.integrations.pipecat import NoveumTraceObserver

load_dotenv(override=True)

OTEL_FILE = os.path.abspath("otel_spans.jsonl")
NOVEUM_FILE = os.path.abspath("noveum_spans.jsonl")


# --------------------------------------------------------------------------- #
# Offline LLM: a real OpenAILLMService with the network call stubbed out, so   #
# the pipeline succeeds end-to-end (real frames, real metrics, real tracing)   #
# without needing API quota. Everything else in the service runs unchanged —   #
# @traced_llm, LLMFullResponse{Start,End}Frame, LLMTextFrame, usage metrics.   #
# --------------------------------------------------------------------------- #
_CANNED_DELTAS = ["The three", " primary colors", " are red,", " green, and blue."]


class _FakeStream:
    """Async-iterable mimicking openai's streaming response object."""

    def __init__(self) -> None:
        chunks = [
            NS(
                model="gpt-4o-mini",
                usage=None,
                choices=[NS(delta=NS(content=d, tool_calls=None))],
            )
            for d in _CANNED_DELTAS
        ]
        # Final usage-only chunk (choices=[]), as OpenAI sends with include_usage.
        chunks.append(
            NS(
                model="gpt-4o-mini",
                choices=[],
                usage=NS(
                    prompt_tokens=21,
                    completion_tokens=9,
                    total_tokens=30,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
            )
        )
        self._chunks = chunks

    def __aiter__(self):
        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()

    async def aclose(self) -> None:
        pass

    async def close(self) -> None:
        pass


class OfflineOpenAILLMService(OpenAILLMService):
    async def get_chat_completions(self, params_from_context):  # type: ignore[override]
        return _FakeStream()


# --------------------------------------------------------------------------- #
# OTel side: a SpanExporter that appends each finished span as one JSON line.  #
# --------------------------------------------------------------------------- #
class FileSpanExporter(SpanExporter):
    def __init__(self, path: str) -> None:
        self._path = path

    def export(self, spans) -> SpanExportResult:
        with open(self._path, "a") as f:
            for span in spans:
                # ReadableSpan.to_json() is multi-line; collapse to one line.
                f.write(json.dumps(json.loads(span.to_json())) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # noqa: D401
        pass


# --------------------------------------------------------------------------- #
# Noveum side: capture finished traces to file instead of POSTing them.       #
# --------------------------------------------------------------------------- #
def _patch_noveum_capture() -> None:
    client = noveum_trace.get_client()

    def _capture_export_trace(trace) -> None:
        with open(NOVEUM_FILE, "a") as f:
            f.write(json.dumps(trace.to_dict(), default=str) + "\n")

    client.transport.export_trace = _capture_export_trace  # type: ignore[assignment]
    client.transport.export_audio = lambda *a, **k: None  # type: ignore[assignment]
    client.transport.flush = lambda *a, **k: None  # type: ignore[assignment]


async def main() -> None:
    for path in (OTEL_FILE, NOVEUM_FILE):
        if os.path.exists(path):
            os.remove(path)

    # 1) Pipecat OpenTelemetry → file
    setup_tracing("span-compare", exporter=FileSpanExporter(OTEL_FILE))

    # 2) noveum_trace → local capture (dummy key; nothing leaves the machine)
    noveum_trace.init(api_key="local-capture", project="span-compare")
    _patch_noveum_capture()
    noveum_obs = NoveumTraceObserver(record_audio=False)

    # 3) Minimal real pipeline: just the OpenAI LLM + context aggregators
    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": "You are a terse assistant. One short sentence.",
            },
            {"role": "user", "content": "Name three primary colors."},
        ]
    )
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    llm = OfflineOpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY") or "offline",
        model="gpt-4o-mini",
    )

    pipeline = Pipeline([user_agg, llm, assistant_agg])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        enable_tracing=True,
        enable_turn_tracking=True,
        observers=[noveum_obs],
    )

    runner = PipelineRunner(handle_sigint=False)
    await noveum_obs.attach_to_task(task)

    # Kick off one LLM run. Delay the EndFrame so the LLM's async response frames
    # (LLMFullResponseStart/Text/End + MetricsFrame) are fully observed before the
    # Noveum observer tears the trace down — otherwise EndFrame races them.
    await task.queue_frames([LLMRunFrame()])

    async def _delayed_stop() -> None:
        await asyncio.sleep(3)
        await task.stop_when_done()

    stopper = asyncio.create_task(_delayed_stop())
    try:
        await runner.run(task)
    finally:
        stopper.cancel()

    # Flush both pipelines.
    noveum_trace.flush()
    noveum_trace.shutdown()
    from opentelemetry import trace as otel_trace

    provider = otel_trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    if hasattr(provider, "shutdown"):
        provider.shutdown()

    _summarize()


def _load_otel() -> list[dict]:
    if not os.path.exists(OTEL_FILE):
        return []
    return [json.loads(line) for line in open(OTEL_FILE) if line.strip()]


def _load_noveum_spans() -> list[dict]:
    spans = []
    if not os.path.exists(NOVEUM_FILE):
        return spans
    for line in open(NOVEUM_FILE):
        if not line.strip():
            continue
        trace = json.loads(line)
        spans.append(
            {
                "name": trace.get("name"),
                "attributes": trace.get("attributes", {}),
                "_trace_level": True,
            }
        )
        spans.extend(trace.get("spans", []))
    return spans


def _print_attrs(attrs: dict, indent: str = "      ") -> None:
    for k, v in attrs.items():
        vs = str(v)
        print(f"{indent}{k} = {(vs[:100] + '…') if len(vs) > 100 else vs}")


def _summarize() -> None:
    otel = _load_otel()
    nov = _load_noveum_spans()

    print("\n" + "=" * 78)
    print("FILES")
    print("=" * 78)
    for label, path in (("OTEL", OTEL_FILE), ("NOVEUM", NOVEUM_FILE)):
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {label:7} {path}  ({size} bytes)")

    print("\n" + "=" * 78)
    print("OTEL SPANS (Pipecat native OpenTelemetry)")
    print("=" * 78)
    for s in otel:
        print(f"• {s.get('name')}  (parent={s.get('parent_id')})")
        _print_attrs(s.get("attributes", {}))
        st = s.get("status", {})
        if st.get("status_code") and st.get("status_code") != "UNSET":
            print(f"      STATUS = {st.get('status_code')}: {st.get('description')}")

    print("\n" + "=" * 78)
    print("NOVEUM SPANS (NoveumTraceObserver)")
    print("=" * 78)
    for s in nov:
        tag = "[trace]" if s.get("_trace_level") else "[span] "
        print(f"• {tag} {s.get('name')}")
        _print_attrs(s.get("attributes", {}))

    # Focused LLM span comparison
    otel_llm = next((s for s in otel if s.get("name") == "llm"), None)
    nov_llm = next((s for s in nov if s.get("name") == "pipecat.llm"), None)
    print("\n" + "=" * 78)
    print("LLM SPAN — KEY FIELDS SIDE BY SIDE")
    print("=" * 78)
    fields = [
        ("model", "gen_ai.request.model", "llm.model"),
        ("output", "output", "llm.output"),
        ("input tokens", "gen_ai.usage.input_tokens", "llm.input_tokens"),
        ("output tokens", "gen_ai.usage.output_tokens", "llm.output_tokens"),
        ("cost", "(not emitted)", "llm.cost.total"),
    ]
    oa = otel_llm.get("attributes", {}) if otel_llm else {}
    na = nov_llm.get("attributes", {}) if nov_llm else {}
    for label, ok, nk in fields:
        ov = oa.get(ok, "—") if ok in oa else "—"
        nv = na.get(nk, "—") if nk in na else "—"
        print(
            f"  {label:14} OTEL[{ok}]={str(ov)[:40]!r:42}  NOVEUM[{nk}]={str(nv)[:40]!r}"
        )
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
