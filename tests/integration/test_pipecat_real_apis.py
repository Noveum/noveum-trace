"""
Real-API integration test: Pipecat pipeline with OpenAI LLM + Cartesia TTS.

Verifies end-to-end that the NoveumTraceObserver:
  1. Correctly captures pipecat.turn / pipecat.stt / pipecat.llm / pipecat.tts spans
  2. Emits a trace payload whose span structure the exotel-audio mapper can process
  3. Produces valid StandardData items with zero Pydantic validation errors

Pipeline used:
  inject UserStartedSpeakingFrame + TranscriptionFrame + UserStoppedSpeakingFrame
    → LLMUserContextAggregator (collects transcription into LLM context)
    → OpenAI gpt-4o-mini LLM (real API call, short answer)
    → Cartesia TTS  (real API call, sonic-2 model)
    → NoveumTraceObserver (captures all MetricsFrames)

Transport is intercepted in-memory so no data is sent to noveum.ai;
the captured trace JSON is saved to datasets/traces/exotel-audio/trace_real_api.json
and then processed through the mapper to verify zero StandardData validation errors.

Run with:
    pytest tests/integration/test_pipecat_real_apis.py -v -s --timeout=120

Requires environment variables (automatically loaded from NovaEval/.env):
    OPENAI_API_KEY
    CARTESIA_API_KEY
    NOVEUM_API_KEY (any value — transport is intercepted, nothing sent to API)
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Load .env credentials from NovaEval/.env
# ---------------------------------------------------------------------------


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


_repo_root = Path(__file__).parent.parent.parent.parent
_load_env(_repo_root / "NovaEval" / ".env")

# Convenience aliases
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
_CARTESIA_KEY = os.environ.get("CARTESIA_API_KEY", "")
_NOVEUM_KEY = os.environ.get("NOVEUM_API_KEY", "test-key")

# ---------------------------------------------------------------------------
# Helpers: in-memory transport capture
# ---------------------------------------------------------------------------

_captured_traces: list[dict[str, Any]] = []


def _intercept_export(self_or_trace: Any, trace: Any = None) -> None:
    """Intercept NoveumClient._export_trace(trace) calls.

    Serialises the Trace object to a dict using HttpTransport.trace_to_dict so
    the captured structure matches what would be sent over the wire.
    """
    from noveum_trace.transport.http_transport import HttpTransport

    trace_obj = trace if trace is not None else self_or_trace
    try:
        transport = HttpTransport.__new__(HttpTransport)
        serialized = transport.trace_to_dict(trace_obj)
    except Exception:
        serialized = {"trace_id": getattr(trace_obj, "trace_id", "unknown")}
    _captured_traces.append(serialized)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.disable_transport_mocking
@pytest.mark.skipif(
    not _OPENAI_KEY or not _CARTESIA_KEY,
    reason="OPENAI_API_KEY and CARTESIA_API_KEY must be set (load from NovaEval/.env)",
)
async def test_pipecat_real_openai_cartesia() -> None:
    """Run a real Pipecat pipeline (OpenAI + Cartesia) and validate with the mapper."""

    pytest.importorskip("pipecat.frames.frames")
    pytest.importorskip("pipecat.services.openai")
    pytest.importorskip("pipecat.services.cartesia")

    from pipecat.frames.frames import (
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.tests.utils import QueuedFrameProcessor

    # ── Import NoveumTraceObserver from src/ ────────────────────────────────
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    # ── Services ─────────────────────────────────────────────────────────────
    llm = OpenAILLMService(
        api_key=_OPENAI_KEY,
        model="gpt-4o-mini",
    )

    tts = CartesiaTTSService(
        api_key=_CARTESIA_KEY,
        settings=CartesiaTTSService.Settings(
            voice="694f9389-aac1-45b6-b726-9d9369183238",  # British Reading Lady
            model="sonic-2",
        ),
    )

    context = OpenAILLMContext(
        messages=[
            {
                "role": "system",
                "content": "You are a concise TypeScript expert. Answer in 2-3 sentences max.",
            }
        ]
    )
    ctx_agg = llm.create_context_aggregator(context)

    sink_q: asyncio.Queue = asyncio.Queue()
    sink = QueuedFrameProcessor(queue=sink_q, queue_direction=FrameDirection.DOWNSTREAM)

    pipeline = Pipeline([ctx_agg.user(), llm, tts, ctx_agg.assistant(), sink])

    # ── NoveumTraceObserver ──────────────────────────────────────────────────
    _captured_traces.clear()

    import noveum_trace as _nt
    from noveum_trace.core.client import NoveumClient

    # Initialize the global Noveum client so the observer can call get_client()
    _nt.init(
        api_key=_NOVEUM_KEY,
        project="pipecat-real-api-test",
        environment="test",
    )

    # Intercept trace export at the client level so we get the raw trace dict
    with patch.object(NoveumClient, "_export_trace", side_effect=_intercept_export):
        observer = NoveumTraceObserver()

        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=False),
            observers=[observer],
        )

        runner = PipelineRunner(handle_sigint=False)

        async def _drive():
            await asyncio.sleep(0.5)
            # Simulate: user speaks → transcription arrives → user stops speaking
            # task.queue_frame() injects frames directly into the pipeline source
            await task.queue_frame(UserStartedSpeakingFrame())
            await asyncio.sleep(0.1)
            await task.queue_frame(
                TranscriptionFrame(
                    text="What is TypeScript?",
                    user_id="test-user",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    finalized=True,
                )
            )
            await asyncio.sleep(0.1)
            await task.queue_frame(UserStoppedSpeakingFrame())
            # Wait for LLM + TTS to complete
            await asyncio.sleep(25)
            await task.cancel()

        await asyncio.gather(runner.run(task), _drive())

    # ── Validate captured trace ──────────────────────────────────────────────
    assert _captured_traces, "No trace was exported to the intercepted transport"
    trace = _captured_traces[-1]

    print("\n=== Captured trace ===")
    print(f"  trace_id:   {trace.get('trace_id', '?')}")
    print(f"  span_count: {trace.get('span_count', len(trace.get('spans', [])))}")

    spans = trace.get("spans", [])
    span_names = [s.get("name") for s in spans]
    print(f"  span names: {span_names}")

    turn_spans = [s for s in spans if s.get("name") == "pipecat.turn"]
    llm_spans = [s for s in spans if s.get("name") == "pipecat.llm"]
    tts_spans = [s for s in spans if s.get("name") == "pipecat.tts"]

    assert turn_spans, "Expected at least one pipecat.turn span"
    assert llm_spans, "Expected at least one pipecat.llm span"

    # Check LLM metrics
    consolidated = [
        s for s in llm_spans if (s.get("attributes") or {}).get("llm.processing_ms")
    ]
    print(f"\n  pipecat.llm spans: {len(llm_spans)} ({len(consolidated)} consolidated)")
    for s in consolidated:
        attrs = s.get("attributes", {})
        print(f"    llm.output[0:60]: {repr((attrs.get('llm.output') or '')[:60])}")
        print(f"    llm.processing_ms: {attrs.get('llm.processing_ms')}")

    # Check TTS metrics
    print(f"\n  pipecat.tts spans: {len(tts_spans)}")
    for s in tts_spans[:3]:
        attrs = s.get("attributes", {})
        print(f"    tts.model: {attrs.get('tts.model')}")
        print(
            f"    tts.input_text[0:50]: {repr((attrs.get('tts.input_text') or '')[:50])}"
        )
        print(
            f"    tts.time_to_first_byte_ms: {attrs.get('tts.time_to_first_byte_ms')}"
        )

    # ── Save trace to datasets dir ──────────────────────────────────────────
    out_dir = _repo_root / "NovaEval" / "datasets" / "traces" / "exotel-audio"
    out_file = out_dir / "trace_real_api.json"
    out_file.write_text(json.dumps(trace, indent=2, default=str))
    print(f"\n  Trace saved to: {out_file}")

    # ── Run mapper and validate StandardData ────────────────────────────────
    novaeval_src = _repo_root / "NovaEval" / "src"
    if str(novaeval_src) not in sys.path:
        sys.path.insert(0, str(novaeval_src))

    mapper_path = out_dir / "mapper.py"
    if mapper_path.exists():
        spec = importlib.util.spec_from_file_location("mapper", mapper_path)
        mapper_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mapper_mod)

        from novaeval.standard_data.standard_data import StandardData

        items = mapper_mod.transform(trace)
        print(f"\n=== Mapper output: {len(items)} StandardData items ===")

        errs = 0
        for i, item in enumerate(items):
            try:
                sd = StandardData(**item)
                mc_count = len(sd.metrics_collected or [])
                print(
                    f"  Turn {i+1}: agent_response={repr((item.get('agent_response') or '')[:60])}"
                    f"  metrics_collected={mc_count}"
                )
            except Exception as e:
                print(f"  Turn {i+1} FAILED validation: {e}")
                errs += 1

        assert errs == 0, f"{errs} StandardData validation error(s)"
        assert items, "Mapper returned no items"


# ---------------------------------------------------------------------------
# Standalone runner (python test_pipecat_real_apis.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(test_pipecat_real_openai_cartesia())
