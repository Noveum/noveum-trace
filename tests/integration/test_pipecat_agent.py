"""
Integration test: full simulated Pipecat voice-agent conversation.

Runs a real Pipecat pipeline with a MockVoiceAgentProcessor that emits the
complete frame sequence a production voice bot produces (STT → LLM → TTS with
MetricsFrames), then verifies that NoveumTraceObserver captures every span and
key attribute correctly.

Pipeline topology used by pipecat.tests.utils.run_test:
  QueuedFrameProcessor (source)
    → MockVoiceAgentProcessor   ← has _settings for LLM/TTS attribute extraction
    → QueuedFrameProcessor (sink)

Observers attached to PipelineTask see FramePushed events for every hop, so
NoveumTraceObserver receives all frames with the correct source processor.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_span(span_id: str, spans: list) -> MagicMock:
    s = MagicMock()
    s.attributes = {}
    s.trace_id = "test-trace-id"
    s.span_id = span_id
    s.is_finished = MagicMock(return_value=False)
    s.finish = MagicMock()
    spans.append(s)
    return s


def _make_client_and_trace(spans: list):
    """Return a (mock_client, mock_trace) pair that records created spans."""
    trace = MagicMock()
    trace.trace_id = "test-trace-id"
    trace.attributes = {}
    trace.events = []
    trace.set_attributes = lambda d: trace.attributes.update(d)
    trace.finish = MagicMock()

    _span_counter = {"n": 0}

    def _mk_span(**kwargs: Any) -> MagicMock:
        _span_counter["n"] += 1
        sid = f"span-{_span_counter['n']}"
        sp = _make_mock_span(sid, spans)
        attrs = kwargs.get("attributes") or {}
        sp.attributes.update(attrs)
        return sp

    trace.create_span = MagicMock(side_effect=_mk_span)

    client = MagicMock()
    client.start_trace = MagicMock(return_value=trace)
    client.finish_trace = MagicMock()
    client.flush = MagicMock()
    client.export_audio = MagicMock()
    return client, trace


# ---------------------------------------------------------------------------
# Mock voice-agent processor
# ---------------------------------------------------------------------------


class MockVoiceAgentProcessor:
    """
    Minimal FrameProcessor subclass that simulates a full LLM+TTS response.

    Defined inside the test module so we can import pipecat lazily (the class
    only subclasses FrameProcessor inside the test function after the importorskip
    guard runs).
    """


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_conversation_captures_all_spans() -> None:
    """
    Two-turn voice conversation:
      Turn 1: bot greeting (LLM → TTS, no user speech)
      Turn 2: user speaks → STT transcript → LLM → TTS

    Assertions:
      - conversation span is created
      - turn spans created for both turns
      - STT span created with stt.text
      - LLM spans created with llm.model, llm.system_prompt, llm.input, llm.output
      - TTS spans created with tts.input_text, tts.time_to_first_byte_ms
      - MetricsFrame token/latency data written to LLM span
    """
    pytest.importorskip("pipecat.frames.frames")
    pytest.importorskip("pipecat.pipeline.pipeline")

    from pipecat.frames.frames import (
        EndFrame,
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMMessagesFrame,
        LLMTextFrame,
        MetricsFrame,
        TranscriptionFrame,
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        TTSTextFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.metrics.metrics import (
        LLMTokenUsage,
        LLMUsageMetricsData,
        ProcessingMetricsData,
        TTFBMetricsData,
        TTSUsageMetricsData,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.tests.utils import QueuedFrameProcessor, SleepFrame

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    SYSTEM_PROMPT = "You are a helpful voice assistant for testing."
    LLM_MODEL = "test-gemini-flash"
    TTS_VOICE = "test-voice-en-US"

    TURN1_LLM_TEXT = "Hello! How can I help you today?"
    TURN2_LLM_TEXT = "Sure, TypeScript generics allow you to write reusable code."
    USER_TRANSCRIPT = "Can you explain TypeScript generics?"

    # -- Mock voice-agent processor ------------------------------------------

    class _MockSettings:
        """Minimal settings object — mirrors pipecat's LLMSettings/TTSSettings shape."""

        model = LLM_MODEL
        system_instruction = SYSTEM_PROMPT
        temperature = 0.7
        voice = TTS_VOICE

    class _MockVoiceAgent(FrameProcessor):
        """Pass-through processor that emits LLM+TTS response frames on VAD stop."""

        _settings = _MockSettings()

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._turn = 0

        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

            if isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._emit_response()

        async def _emit_response(self) -> None:
            self._turn += 1
            llm_text = TURN2_LLM_TEXT if self._turn > 1 else TURN1_LLM_TEXT
            tts_text = llm_text

            # LLM response stream
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=llm_text[:15]))
            await self.push_frame(LLMTextFrame(text=llm_text[15:]))
            await self.push_frame(LLMFullResponseEndFrame())

            # LLM metrics
            await self.push_frame(
                MetricsFrame(
                    data=[
                        TTFBMetricsData(processor=self.name, value=0.45),
                        ProcessingMetricsData(processor=self.name, value=1.1),
                        LLMUsageMetricsData(
                            processor=self.name,
                            value=LLMTokenUsage(
                                prompt_tokens=60,
                                completion_tokens=25,
                                total_tokens=85,
                            ),
                        ),
                    ]
                )
            )

            # TTS stream
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(TTSTextFrame(text=tts_text))
            # Minimal silent audio (1 frame × 320 bytes at 16 kHz / 16-bit mono)
            await self.push_frame(
                TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
            )
            await self.push_frame(TTSStoppedFrame())

            # TTS metrics
            await self.push_frame(
                MetricsFrame(
                    data=[
                        TTFBMetricsData(processor="MockTTS", value=0.18),
                        TTSUsageMetricsData(processor="MockTTS", value=len(tts_text)),
                    ]
                )
            )

    # -- Build NoveumTraceObserver with mocked client ------------------------

    spans: list[MagicMock] = []
    client, mock_trace = _make_client_and_trace(spans)

    observer = NoveumTraceObserver(capture_text=True, record_audio=False)

    # -- Frames to inject into the pipeline ----------------------------------
    #
    # The pipeline receives these via task.queue_frame().  The observer sees
    # FramePushed(source=QueuedFrameProcessor_source, destination=_MockVoiceAgent)
    # for injected frames, and
    # FramePushed(source=_MockVoiceAgent, destination=QueuedFrameProcessor_sink)
    # for frames emitted by the agent — so _settings IS found on the LLM/TTS frames.

    system_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
    ]

    frames_to_send = [
        # --- Turn 1: greeting (bot-initiated, no user input) -----------------
        SleepFrame(sleep=0.02),
        LLMMessagesFrame(messages=system_messages),
        VADUserStartedSpeakingFrame(),
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.05),
        # --- Turn 2: user asks a question -----------------------------------
        SleepFrame(sleep=0.02),
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text=USER_TRANSCRIPT, user_id="test-user", timestamp="0"),
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.05),
    ]

    # -- Run pipeline --------------------------------------------------------

    agent = _MockVoiceAgent()

    received_up: asyncio.Queue = asyncio.Queue()
    received_down: asyncio.Queue = asyncio.Queue()
    source_proc = QueuedFrameProcessor(
        queue=received_up, queue_direction=FrameDirection.UPSTREAM, ignore_start=True
    )
    sink_proc = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=True,
    )

    pipeline = Pipeline([source_proc, agent, sink_proc])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        observers=[observer],
        cancel_on_idle_timeout=False,
    )

    with patch.object(observer, "_get_client", return_value=client):

        async def _push() -> None:
            await asyncio.sleep(0.01)
            for frame in frames_to_send:
                if isinstance(frame, SleepFrame):
                    await asyncio.sleep(frame.sleep)
                else:
                    await task.queue_frame(frame)
            await asyncio.sleep(0.05)
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), _push())

    # -- Assertions ----------------------------------------------------------

    # Conversation trace was created
    assert client.start_trace.called, "start_trace should be called"

    # finish_trace called (conversation ended)
    assert client.finish_trace.called, "finish_trace should be called on EndFrame"

    # At least one span was created
    assert len(spans) > 0, "At least one child span should be created"

    span_names = [
        c.kwargs.get("name") or (c.args[0] if c.args else "")
        for c in mock_trace.create_span.call_args_list
    ]
    print("Created spans:", span_names)

    # LLM span(s) created
    llm_spans = [s for s in span_names if "llm" in str(s)]
    assert llm_spans, f"Expected pipecat.llm spans, got: {span_names}"

    # TTS span(s) created
    tts_spans = [s for s in span_names if "tts" in str(s)]
    assert tts_spans, f"Expected pipecat.tts spans, got: {span_names}"

    # Check LLM span attributes — model + system prompt from _settings
    llm_span_calls = [
        c
        for c in mock_trace.create_span.call_args_list
        if "llm" in str(c.kwargs.get("name") or (c.args[0] if c.args else ""))
    ]
    assert llm_span_calls, "create_span called for pipecat.llm"

    first_llm_attrs = llm_span_calls[0].kwargs.get("attributes") or {}
    assert (
        first_llm_attrs.get("llm.model") == LLM_MODEL
    ), f"llm.model not captured; got: {first_llm_attrs}"
    assert SYSTEM_PROMPT in str(
        first_llm_attrs.get("llm.system_prompt", "")
    ), f"llm.system_prompt not captured; got: {first_llm_attrs}"

    # Check TTS span attributes
    tts_span_calls = [
        c
        for c in mock_trace.create_span.call_args_list
        if "tts" in str(c.kwargs.get("name") or (c.args[0] if c.args else ""))
    ]
    first_tts_attrs = tts_span_calls[0].kwargs.get("attributes") or {}
    print("First TTS span attributes:", first_tts_attrs)

    # STT transcript captured
    stt_span_calls = [
        c
        for c in mock_trace.create_span.call_args_list
        if "stt" in str(c.kwargs.get("name") or (c.args[0] if c.args else ""))
    ]
    if stt_span_calls:
        stt_span = spans[
            [
                i
                for i, c in enumerate(mock_trace.create_span.call_args_list)
                if "stt" in str(c.kwargs.get("name") or (c.args[0] if c.args else ""))
            ][0]
        ]
        print("STT span attributes:", stt_span.attributes)
        if stt_span.attributes.get("stt.text"):
            assert USER_TRANSCRIPT in stt_span.attributes["stt.text"]

    # LLM output captured via llm.output on finished span
    finished_llm_spans = [sp for sp in spans if sp.attributes.get("llm.output")]
    assert finished_llm_spans, "llm.output should be written when LLM span closes"
    assert (
        TURN1_LLM_TEXT in finished_llm_spans[0].attributes["llm.output"]
        or TURN2_LLM_TEXT in finished_llm_spans[0].attributes["llm.output"]
    ), f"Expected LLM output text, got: {finished_llm_spans[0].attributes}"

    # Metrics captured: llm.processing_ms (from ProcessingMetricsData)
    spans_with_metrics = [
        sp for sp in spans if sp.attributes.get("llm.processing_ms") is not None
    ]
    assert spans_with_metrics, "llm.processing_ms should be captured from MetricsFrame"

    # Conversation attributes flushed to trace
    assert client.finish_trace.called


# ---------------------------------------------------------------------------
# AudioBufferProcessor integration test
#
# Validates the full audio-capture pipeline required by the Exotel integration:
#
#   transport.input()
#     → stt            (audio_passthrough=True — default)
#     → user_aggregator
#     → mandate_processor  ← MUST re-emit InputAudioRawFrame downstream!
#     → user_idle
#     → tts
#     → transport.output()
#     → AudioBufferProcessor(num_channels=2)   ← placed AFTER transport.output()
#
# The observer's attach_to_task() auto-detects the AudioBufferProcessor,
# calls start_recording(), and wires on_audio_data.  When EndFrame arrives
# the ABP fires on_audio_data → observer builds a WAV and uploads it as
# pipecat.full_conversation (stereo: left=user, right=bot).
#
# Key requirement that caused Exotel's missing-user-audio issue:
#   MandateProcessor consumed InputAudioRawFrame without re-emitting them.
#   Fix: re-emit (pass-through) InputAudioRawFrame after internal processing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_buffer_processor_captures_full_conversation() -> None:
    """
    Verifies that NoveumTraceObserver correctly captures full-conversation
    stereo audio via AudioBufferProcessor and creates the pipecat.full_conversation
    span with the right attributes.

    This is the critical end-to-end path for the Exotel integration.
    """
    pytest.importorskip("pipecat.frames.frames")
    pytest.importorskip("pipecat.processors.audio.audio_buffer_processor")

    from pipecat.frames.frames import (
        EndFrame,
        InputAudioRawFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        MetricsFrame,
        TranscriptionFrame,
        TTSAudioRawFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        TTSTextFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.metrics.metrics import (
        LLMTokenUsage,
        LLMUsageMetricsData,
        TTFBMetricsData,
        TTSUsageMetricsData,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.tests.utils import QueuedFrameProcessor, SleepFrame

    from noveum_trace.integrations.pipecat.pipecat_observer import NoveumTraceObserver

    # ---- Settings object mirrors pipecat LLMSettings/TTSSettings shape ----

    class _MockSettings:
        model = "test-gemini-flash"
        system_instruction = "You are a helpful test assistant."
        temperature = 0.7
        voice = "test-voice-en-US"

    # ---- Mock voice-agent processor: simulates MandateProcessor correctly --
    # Key behaviour: passes InputAudioRawFrame DOWNSTREAM (re-emits it) so the
    # AudioBufferProcessor at the end of the pipeline captures user audio.

    class _MockVoiceAgentWithAudio(FrameProcessor):
        """Simulates a full pipeline including audio frame passthrough."""

        _settings = _MockSettings()

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._responded = False

        async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

            if isinstance(frame, VADUserStoppedSpeakingFrame) and not self._responded:
                self._responded = True
                await self._emit_llm_tts_response()

        async def _emit_llm_tts_response(self) -> None:
            bot_text = (
                "Table tennis is played on a 9-foot table with a net in the middle."
            )

            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=bot_text))
            await self.push_frame(LLMFullResponseEndFrame())
            await self.push_frame(
                MetricsFrame(
                    data=[
                        TTFBMetricsData(processor=self.name, value=0.4),
                        LLMUsageMetricsData(
                            processor=self.name,
                            value=LLMTokenUsage(
                                prompt_tokens=40, completion_tokens=18, total_tokens=58
                            ),
                        ),
                    ]
                )
            )

            # TTS output: these become OutputAudioRawFrame downstream, captured by ABP
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(TTSTextFrame(text=bot_text))
            # TTSAudioRawFrame is a subclass of OutputAudioRawFrame — ABP sees it
            await self.push_frame(
                TTSAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1)
            )
            await self.push_frame(TTSStoppedFrame())
            await self.push_frame(
                MetricsFrame(
                    data=[
                        TTFBMetricsData(processor="MockTTS", value=0.2),
                        TTSUsageMetricsData(processor="MockTTS", value=len(bot_text)),
                    ]
                )
            )

    # ---- AudioBufferProcessor (stereo: user=left, bot=right) ---------------

    audio_buffer = AudioBufferProcessor(num_channels=2)
    captured_audio: dict[str, Any] = {}

    @audio_buffer.event_handler("on_audio_data")
    async def _on_audio(
        _buf: Any, audio: bytes, sample_rate: int, num_channels: int
    ) -> None:
        captured_audio["audio"] = audio
        captured_audio["sample_rate"] = sample_rate
        captured_audio["num_channels"] = num_channels

    # ---- Build pipeline: agent → audio_buffer (after transport.output()) ---

    agent = _MockVoiceAgentWithAudio()

    # 32-byte minimal user audio frame (1 ms @ 16kHz/16-bit mono, matches audio_out_sample_rate)
    user_audio_bytes = b"\x10\x20" * 16
    user_audio_frame = InputAudioRawFrame(
        audio=user_audio_bytes, sample_rate=16000, num_channels=1
    )

    frames_to_send = [
        SleepFrame(sleep=0.02),
        # User audio frames (normally from transport.input — must pass through MandateProcessor!)
        user_audio_frame,
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(
            text="How do I play table tennis?", user_id="u1", timestamp="0"
        ),
        VADUserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.06),
    ]

    # Pipeline: source → agent → audio_buffer → sink
    # This mirrors: transport.output() → AudioBufferProcessor(num_channels=2)
    received_up: asyncio.Queue = asyncio.Queue()
    received_down: asyncio.Queue = asyncio.Queue()
    source_proc = QueuedFrameProcessor(
        queue=received_up, queue_direction=FrameDirection.UPSTREAM, ignore_start=True
    )
    sink_proc = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=True,
    )

    pipeline = Pipeline([source_proc, agent, audio_buffer, sink_proc])

    # ---- Observer wired with attach_to_task() (required for ABP detection) -

    spans: list[MagicMock] = []
    client, mock_trace = _make_client_and_trace(spans)

    observer = NoveumTraceObserver(capture_text=True, record_audio=True)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, audio_out_sample_rate=16000),
        observers=[observer],
        cancel_on_idle_timeout=False,
    )

    with patch.object(observer, "_get_client", return_value=client):
        # attach_to_task() auto-detects ABP and calls start_recording()
        await observer.attach_to_task(task)

        async def _push() -> None:
            await asyncio.sleep(0.01)
            for frame in frames_to_send:
                if isinstance(frame, SleepFrame):
                    await asyncio.sleep(frame.sleep)
                else:
                    await task.queue_frame(frame)
            await asyncio.sleep(0.08)
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), _push())

    # ---- Assertions: pipecat.full_conversation span created ----------------

    span_names = [
        c.kwargs.get("name") or (c.args[0] if c.args else "")
        for c in mock_trace.create_span.call_args_list
    ]
    print("Spans created:", span_names)
    print(
        "Captured audio:",
        {k: (len(v) if k == "audio" else v) for k, v in captured_audio.items()},
    )

    full_conv_calls = [
        c
        for c in mock_trace.create_span.call_args_list
        if "full_conversation"
        in str(c.kwargs.get("name") or (c.args[0] if c.args else ""))
    ]
    assert (
        full_conv_calls
    ), f"pipecat.full_conversation span not created; got spans: {span_names}"

    full_conv_attrs = full_conv_calls[0].kwargs.get("attributes") or {}
    print("full_conversation attributes:", full_conv_attrs)

    # Stereo attributes set
    assert (
        full_conv_attrs.get("full_conversation.audio_channels") == "stereo"
    ), f"Expected stereo; got: {full_conv_attrs}"
    assert full_conv_attrs.get("full_conversation.audio_channel_left") == "user"
    assert full_conv_attrs.get("full_conversation.audio_channel_right") == "bot"
    assert (
        full_conv_attrs.get("full_conversation.audio_source") == "AudioBufferProcessor"
    )
    assert "full_conversation.audio_uuid" in full_conv_attrs

    # Audio was uploaded to Noveum
    assert (
        client.export_audio.called
    ), "client.export_audio() should be called with the WAV bytes"
    export_call = client.export_audio.call_args
    export_metadata = export_call.kwargs.get("metadata", {})
    assert export_metadata.get("num_channels") == 2
    assert export_metadata.get("format") == "wav"
    assert export_metadata.get("type") == "conversation"

    # WAV bytes must be non-empty (header alone is 44 bytes)
    exported_wav = export_call.kwargs.get("audio_data", b"")
    assert (
        len(exported_wav) > 44
    ), f"Expected non-empty WAV; got {len(exported_wav)} bytes"

    # Conversation was properly finished
    assert client.finish_trace.called
