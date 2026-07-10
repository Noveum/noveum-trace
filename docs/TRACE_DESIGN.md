# Trace Design — what a good trace looks like

This document describes the **trace design philosophy** behind the Noveum Trace
SDK and its integrations (LiveKit, LangChain, Pipecat, CrewAI, …). It is
deliberately *not* a per-integration reference — for the exact attribute keys a
given integration emits, read the source or `AGENTS.md`.

The goal here is **prediction**: given a piece of instrumented application code
(e.g. a Pipecat pipeline), you should be able to *guess the shape of the trace
it produces* — how many spans, how they nest, and what attributes hang off each
one. The rules below are the procedure for doing that.

---

## 1. The guiding heuristic (refined)

> **Capture every piece of information that is available at the point of
> instrumentation — because filtering, aggregation, and arrangement happen on
> the ETL/analytics side, not at capture time.**

The trace is a *wide, lossless-as-possible record*. We do not decide at capture
time what will be "useful"; we record it and let downstream ETL slice it. If a
model name, a temperature, a token count, a confidence score, a latency, a
request ID, or a raw input is reachable, it goes on the span.

But the raw heuristic has **four practical bounds** that you must apply when
predicting a trace, or your prediction will be wrong:

1. **Size caps.** Buffers and histories are bounded (`MAX_CONVERSATION_HISTORY`,
   `MAX_TEXT_BUFFER_LENGTH`, `MAX_STT_AUDIO_FRAMES`, …). Oldest data is dropped
   on overflow.
2. **First-N sampling.** Large collections are sampled, not dumped — LangChain
   stores `llm.input.prompts` as the *first 5*, `retrieval.sample_results` as
   the *first 10*. When a producer truncates, it **emits a flag**
   (`retrieval.results_truncated: true`).
3. **PII redaction.** Text can be redacted/pseudonymized before it lands on a
   span (`utils/pii_redaction.py`).
4. **Binary goes out-of-band.** Audio and images are **not** inlined. They are
   uploaded separately and referenced by a UUID attribute
   (`stt.audio_uuid`, `tts.audio_uuid`, `full_conversation.audio_uuid`,
   `llm.input.image_uuids`). The span carries the pointer + metadata
   (format, duration, sample rate, channels), not the bytes.

So the honest, predictive form of the heuristic is:

> *Capture all available info, bounded by size caps and PII redaction; binary is
> referenced by UUID, not inlined; and whenever you truncate, emit a flag.*

---

## 2. The data model (the prediction target)

Every trace serializes (`Trace.to_dict()` / `Span.to_dict()`) into this skeleton.
When you predict a trace, you are filling in this template:

```jsonc
{
  "trace_id": "...",
  "name": "pipecat.conversation",          // root = the session/interaction
  "status": "ok",                          // ok | error | unset (+ timeout/cancelled)
  "duration_ms": 12345.6,
  "metadata": { "user_id", "session_id", "request_id", "tags", "custom_attributes" },
  "attributes": { ... },                   // session/conversation-wide facts
  "events": [ { "name", "timestamp", "attributes" } ],   // trace-level milestones
  "spans": [
    {
      "span_id": "...",
      "parent_span_id": "...",             // null for a root/turn-level span
      "name": "pipecat.llm",
      "start_time": "...", "end_time": "...", "duration_ms": 42.0,
      "status": "ok", "status_message": null,
      "attributes": { ... },               // everything known about this operation
      "events": [ ... ],                   // point-in-time occurrences within it
      "links": [ ... ],                    // pointers to related spans
      "exception": { "type", "message", "stack_trace" }   // if it failed
    }
  ]
}
```

> **Serialization note (verified against the live backend, §10):** a
> root/trace-level span (a turn, or the `full_conversation` span) has its
> `parent_span_id` serialized as `""` (empty string), **not** `null`. And the
> trace-by-id backend API surfaces **span** `events` but was not observed to
> return **trace-level** `events` — so prefer attaching milestones to a span
> (e.g. the turn) when you need them visible downstream.

There are exactly **four building blocks**. Choosing the right one for a given
piece of code is the single most important prediction skill (see §3):

| Block | What it is | Lives on |
|---|---|---|
| **Trace** | One complete interaction/session | top-level |
| **Span** | One independently-timed operation | nested under trace/span |
| **Attribute** | A property/input/output/metric *of* a span or trace | `attributes` dict |
| **Event** | A point-in-time occurrence with no duration | `events` list |

(Plus **Link**: a cross-reference from one span to another — used rarely, e.g.
correlating an uploaded audio file back to its span.)

`SpanStatus` has five enum values — `unset`, `ok`, `error`, `timeout`,
`cancelled` — but **only `ok`/`error`/`unset` are reliably used via the native
`set_status()` path. Never write `"success"`. (See §7 for the Pipecat
exception, which does *not* use the enum at all.)

---

## 3. The core decision: span vs. attribute vs. event

This is the rule that determines the *shape* of every predicted trace. Apply it
to each construct in the code:

### → It becomes a **span** if it is an independently-timed operation.

It has a start and an end, and you care how long it took. Examples across all
integrations:

- A speech-to-text recognition (`pipecat.stt`, `stt.recognize`)
- An LLM call (`pipecat.llm`, `llm.chat`, `llm.{model}`)
- A text-to-speech synthesis (`pipecat.tts`, `tts.synthesize`)
- A conversational turn (`pipecat.turn`)
- A chain / agent / retriever invocation (LangChain `chain.*`, `agent.*`,
  `retrieval.*`)
- The whole session/conversation (the root trace itself)

### → It becomes an **attribute** if it is a property, input, output, or metric *of* an operation.

It describes a span; it is not itself a timed thing. Examples:

- Model name, temperature, max_tokens, system prompt → attributes on the LLM span
- The prompt/messages in, the completion out → attributes
- Token counts and cost → attributes
- STT transcript, confidence, language → attributes on the STT span
- **Tool / function calls are attributes, not spans.** This trips people up:
  in *all three* integrations, a tool call is recorded as an attribute on the
  **LLM span**, **not** as its own child span. There are *two families* — both
  on the LLM span (live-verified except where noted, see §10):
  - **Requested** (the model asked to call a tool): `llm.output.tool_calls`,
    `llm.output.tool_calls.count`, `llm.output.tool_calls.names` (LangChain).
    Present whenever the model returns tool calls — even if nothing executes them
    (e.g. a bare `llm.bind_tools(...).invoke(...)`).
  - **Executed** (a tool actually ran and returned): `llm.function_calls` +
    `llm.function_call_results` (Pipecat — *verified live*, §10), and
    `llm.executed_tool_calls` (LangChain agent/tool loop — *attested from source*,
    not exercised in the live run). Requires an execution loop, so it appears in
    agents/voice bots, not a bare LLM call.

  If the code shows the model calling a function, predict an attribute on the LLM
  span — never a new span. Which family depends on whether the tool is also
  *executed* in that code path.

### → It becomes an **event** if it is a point-in-time occurrence with no meaningful duration.

A thing *happened* at an instant, inside an operation or the session. Examples:

- `pipecat.error` (an error frame arrived)
- `stt.interim_transcription` (an interim hypothesis was emitted)
- `client.connected` / `bot.connected` (transport milestones — trace-level)
- `user.muted` / `user.unmuted`
- `exception` (auto-added by `record_exception()`)

Get this wrong and the whole predicted trace has the wrong topology. When unsure:
*does it have a duration I care about?* → span. *Is it a fact about something?*
→ attribute. *Did it just happen at a moment?* → event.

---

## 4. The hierarchy principle

> **The trace mirrors the application's actual execution structure. The root is
> the session / top-level interaction; children are the operations it spawns,
> nested the way they nest at runtime.**

The three integrations realize this principle with genuinely *different
topologies* — don't average them. Learn the principle, then recognize which
topology a given framework uses.

**Pipecat — clean, fixed nesting (use this as your default mental model):**

```
Trace: pipecat.conversation                 ← the session
  ├─ Span: pipecat.turn  (turn.number=1)    ← one user↔bot exchange
  │    ├─ Span: pipecat.stt                 ← children of the turn
  │    ├─ Span: pipecat.llm
  │    └─ Span: pipecat.tts
  ├─ Span: pipecat.turn  (turn.number=2)
  │    └─ ...
  └─ Span: pipecat.full_conversation        ← trace-level, stereo audio
```

Turn spans are direct children of the trace (`parent_span_id` = the turn's id is
*not* set — they sit under the trace). STT/LLM/TTS spans are children of the
*current turn*.

**LangChain — a `run_id` / `parent_run_id` tree.** LangChain hands every
callback a `run_id` and a `parent_run_id`; the integration maps those to span
parent/child. Nesting is whatever the chain/graph does at runtime
(chain → llm → … ), with a 3-tier parent resolution (explicit parent_run_id →
manual `metadata["noveum"]["parent_name"]` → current-context fallback).

**LiveKit — a session trace with event-driven spans.** One root
`livekit.agent_session` trace; each framework event (`user_input_transcribed`,
`metrics_collected`, `speech_created`, …) becomes a span, with *special parent
resolution* (e.g. metrics spans attach to the latest `agent_state_changed`
rather than the context's current span).

The common thread: **root = the interaction; spans = the real operations; the
tree = the real call/event structure.**

---

## 5. The shared vocabulary (what every operation span fills in)

This is what makes the design integration-agnostic. Regardless of framework,
every operation span fills the same **slots**, using the same **namespaced,
dot-delimited keys**. Predict these slots first; the exact key is almost always
`<domain>.<field>` or `<domain>.<group>.<field>`.

### 5.1 Namespace prefixes

| Prefix | Domain |
|---|---|
| `llm.` | LLM call: model, params, input, output, tokens, cost, timing, tools |
| `stt.` | Speech-to-text: transcript, confidence, language, audio, latency |
| `tts.` | Text-to-speech: input text, voice, model, characters, audio, latency |
| `tool.` / `llm.*tool*` | Tool/function calls (as attributes on the LLM span) |
| `retrieval.` | Retriever: query, result_count, sample_results, truncated flag |
| `agent.` | Agent: name, type, role, goal, available_tools, input/output |
| `turn.` | Conversational turn: number, duration, user_input, latency, EOU metrics |
| `conversation.` | Conversation-wide rollups: total tokens, total cost, turn count |
| `session.` / `pipeline.` | Transport & pipeline config: room, transport type, sample rate |
| `job.` / `langchain.` / `langgraph.` | Framework bookkeeping (run ids, job ids, graph nodes) |
| `code.` / `function.definition.` | Source location of the instrumented call |
| `full_conversation.` | The stereo session-audio span |
| `exception.` / `error.` | Error type, message, stacktrace |
| `noveum.` | User-supplied custom attributes |

### 5.2 The slots every operation span should fill

1. **Identity** — what kind of operation and which provider/model.
   `*.model`, `*.provider`, `*.request_id`, `*.operation`, `*.mode`
   (`"batch"`/`"streaming"`).
2. **Timing & latency** — see §6 for the full taxonomy. Always at least the
   span's own `duration_ms`.
3. **Input** — the full input, bounded by §1. `llm.input` / `llm.input.messages`
   / `llm.system_prompt`, `stt` audio (→ uuid), `tts.input_text`,
   `retrieval.query`.
4. **Output** — the full output, bounded by §1. `llm.output` / `llm.response`,
   `stt.text` / `stt.transcript`, `tts` audio (→ uuid),
   `retrieval.sample_results`.
5. **Tokens & cost** — for LLM spans, always (see §5.3).
6. **Status** — `ok` / `error` (see §7).
7. **Config / params** — every sampling knob you can see: `llm.temperature`,
   `llm.top_p`, `llm.top_k`, `llm.max_tokens`, `llm.frequency_penalty`,
   `llm.presence_penalty`, `llm.seed`, `llm.tool_choice`,
   `llm.parallel_tool_calls`. *Filter out unset/sentinel values* — if a param
   was never set (`NOT_GIVEN`, empty dict), it is **not** emitted.

### 5.3 Tokens & cost are computed, not just passed through

When usage is available, the SDK records the token breakdown **and derives cost
locally** from a built-in pricing table (`utils/llm_utils.py` `MODEL_REGISTRY`,
`estimate_cost`). So predict both:

- Tokens: `llm.input_tokens` / `llm.prompt_tokens`, `llm.output_tokens` /
  `llm.completion_tokens`, `llm.total_tokens`, plus when present
  `llm.cache_read_tokens`, `llm.cache_creation_tokens`,
  `llm.reasoning_tokens`.
- Cost (derived): `llm.cost.input`, `llm.cost.output`, `llm.cost.total`,
  `llm.cost.currency` (`"USD"`).
- Rolled up onto the trace: `conversation.total_input_tokens`,
  `conversation.total_output_tokens`, `conversation.total_cost`.

If the code uses a model that's in `MODEL_REGISTRY`, predict cost attributes even
if the framework never reports cost — the SDK computes it.

### 5.4 Complex values are JSON-encoded strings

Lists/dicts of structured data are stored as JSON strings: `llm.input` (message
array), `llm.tools`, `llm.tool_choice`, `llm.function_calls`,
`stt.interim_results`, `*.available_tools.schemas`. Scalars stay native
(int/float/bool/str).

---

## 6. The timing & latency taxonomy

A good trace records far more than wall-clock duration. For any streaming
operation, predict the relevant sub-latencies:

| Concept | Typical keys |
|---|---|
| Span wall-clock | `duration_ms` (always), `*.duration` / `*.duration_ms` |
| LLM time-to-first-token | `llm.ttft`, `llm.ttft_ms`, `llm.time_to_first_token_ms` |
| LLM throughput | `llm.tokens_per_second` |
| LLM processing | `llm.processing_ms`, `llm.latency_ms` |
| TTS time-to-first-byte | `tts.time_to_first_byte_ms` |
| TTS text aggregation | `tts.text_aggregation_ms` |
| STT VAD→final / first interim | `stt.vad_to_final_ms`, `stt.first_text_latency_ms` |
| Turn duration & user speech | `turn.duration_seconds`, `turn.user_speech_duration_seconds` |
| User→bot responsiveness | `turn.user_bot_latency_seconds` |
| End-of-utterance detection | `turn.eou_processing_time_ms`, `turn.eou_inference_ms`, `turn.eou_confidence`, `turn.eou_is_complete` |
| Audio length | `full_conversation.duration_ms`, `*.audio_duration_ms` |

Rule of thumb: **if the framework exposes a timestamp or a latency, capture it as
its own attribute** — don't fold it into the span duration.

---

## 7. Errors & status

Two recording mechanisms, and **they differ by integration** — this matters for
prediction:

- **Native status (LiveKit, LangChain):** `span.set_status(SpanStatus.OK)` /
  `set_status(SpanStatus.ERROR, message)`, and `span.record_exception(err)`
  which auto-adds `exception.type` / `exception.message` /
  `exception.stacktrace` attributes plus an `exception` event. This shows up in
  the serialized `status` field.
- **Custom-attribute status (Pipecat):** Pipecat does **not** use the enum.
  It writes a plain attribute `pipecat_span_status` with values
  `"ok"` / `"error"` / `"cancelled"` / `"upload_failed"` (note: the last two are
  not even in the enum), plus `pipecat_span_status_message`. Errors also append a
  `pipecat.error` **event** (with `error.message`, `error.type`) to the active
  **turn span** and to the **trace**. So when predicting a Pipecat trace, emit the
  *attribute* form, not the native `status` field.

  > **Critical for ETL / filtering (verified, §10):** on a Pipecat error the
  > native `status` stays `"ok"` and the trace's `error_count` stays `0` — the
  > error lives *only* in the `pipecat_span_status` attribute (+ the
  > `pipecat.error` event). A query that filters by native `status=error` or
  > `error_count>0` will **silently miss every Pipecat error**. Filter on the
  > `pipecat_span_status` attribute instead.

When predicting: an error path produces (a) an error status (in whichever form
the integration uses), (b) a status/error message, and often (c) an event
recording the moment it happened. Interruptions/cancellations are their own
status (`cancelled`), not errors.

---

## 8. The prediction procedure (code → trace)

Given an instrumented code example, walk it like this:

1. **Find the root.** What's the top-level interaction being traced? That's the
   trace, and its `name` usually reflects the integration
   (`pipecat.conversation`, `livekit.agent_session`, or the first chain/graph).
   Attach session-wide facts to it (`session.*`, `pipeline.*`, conversation
   rollups).
2. **Identify the timed operations** → one span each (§3). For voice pipelines,
   that's turns and the STT/LLM/TTS within each turn. For LangChain, that's each
   chain/llm/agent/retriever callback.
3. **Lay out the tree** using the integration's topology (§4). Default to the
   Pipecat conversation→turn→{stt,llm,tts} shape unless the framework dictates
   otherwise.
4. **For each span, fill the slots** (§5.2): identity, timing (§6), input,
   output, tokens+cost (§5.3 — derive cost if the model is known), status (§7),
   and every config/param you can see (dropping unset/sentinel values).
5. **Demote the non-operations.** Tool/function calls → attributes on the LLM
   span. Interim results, errors, connection milestones → events.
6. **Apply the bounds** (§1): sample large collections (first-N + a truncated
   flag), reference binary by UUID + metadata (never inline), assume PII may be
   redacted.
7. **Roll up** onto the trace: total tokens, total cost, turn count, last user
   input, overall status.

---

## 9. Worked example (verified against a real test)

The following is reverse-engineered from
`tests/integration/pipecat/test_llm_behavior.py`, which drives **real Pipecat
frames** against a **real `Trace`** and asserts the resulting span — so this is
ground truth, not a guess.

**The code (a Pipecat LLM response cycle):**

```python
# An LLM service configured like this...
settings = SimpleNamespace(
    model="gpt-4o",
    system_instruction="be brief",
    temperature=0.7,
    top_p=NOT_GIVEN,         # sentinel
    frequency_penalty={},    # empty
    max_tokens=256,
    max_completion_tokens=512,
)
# ...with this context and tools, producing a response:
context.get_messages() == [{"role": "user", "content": "hi"}]
tools == [{"type": "function", "function": {"name": "f", "parameters": {}}}]
tool_choice == "auto"
# frames: LLMContextFrame -> LLMSetToolsFrame -> LLMSetToolChoiceFrame
#         -> LLMFullResponseStartFrame -> (text) -> LLMFullResponseEndFrame
#         -> MetricsFrame(tokens)
```

**The trace you should predict (and what the test asserts):**

```
Trace: pipecat.conversation
  └─ Span: pipecat.turn          (turn.number = 1)
       └─ Span: pipecat.llm      (parent_span_id = turn.span_id)
            attributes:
              llm.model         = "gpt-4o"
              llm.system_prompt = "be brief"
              llm.temperature   = 0.7
              llm.max_tokens    = 512          # max_completion_tokens overwrites
              # llm.top_p           NOT present  (sentinel filtered)
              # llm.frequency_penalty NOT present (empty filtered)
              llm.input         = '[{"role":"user","content":"hi"}]'   # JSON string
              llm.tools         = '[{"type":"function",...}]'          # JSON string
              llm.tool_choice   = "auto"
              llm.output        = "<assistant text>"
              llm.input_tokens / llm.output_tokens / llm.total_tokens
              llm.cost.input / llm.cost.output / llm.cost.total / llm.cost.currency
              pipecat_span_status = "ok"
```

Things this example pins down, each an instance of a rule above:

- **One LLM span**, child of the **turn** (§3 span rule, §4 topology).
- **Sentinel/empty params are dropped** — `top_p` and `frequency_penalty` do
  *not* appear (§5.2 "filter out unset values").
- **`max_completion_tokens` overwrites `max_tokens`** → `llm.max_tokens = 512`.
- **Structured inputs/tools are JSON strings** (§5.4).
- **Cost is derived** from `gpt-4o` via `MODEL_REGISTRY`, even though no cost was
  reported by the framework (§5.3).
- **Status is a custom attribute** `pipecat_span_status="ok"`, not the native
  enum (§7).

If the same call had requested a function instead of returning text, you would
*not* add a span — you'd add `llm.function_calls` / `llm.function_call_results`
attributes to this same `pipecat.llm` span (§3 tool rule).

---

## 10. Validation — these rules were checked against live traces

This doc was validated end-to-end on 2026-06-28 by running real integrations and
inspecting the traces both in-SDK and after round-tripping through the Noveum
backend (project `trace-design-validation`).

**LangChain** — ran a real `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) call and
a `bind_tools(...).invoke(...)` tool call through `NoveumTraceCallbackHandler`,
exported to the backend. (Not the bundled example file — `docs/examples/langchain_integration_example.py`
is OpenAI-only and its chain/agent helpers are removed in langchain 1.x; this was
an equivalent custom driver. Re-run against repo `src/` via `PYTHONPATH=src` —
byte-identical span shape to the installed build, so the langchain path does not
suffer the stale-venv divergence below.) Confirmed:
- A bare LLM call produces a single root span named `llm.<model>` — the LLM span
  *is* the trace root (`parent_span_id` empty).
- A tool-requesting call produces **one** span with `llm.output.tool_calls{,.count,.names}`
  — **no separate tool span** (the headline claim).
- Cost (`llm.cost.{input,output,total,currency}`) is present **even though the
  provider API returns no cost** — derived locally from `MODEL_REGISTRY`.
- Native `status: "ok"` (LangChain uses the real `SpanStatus` enum).
- `code.{file,line,function,module,context}` and JSON-encoded `llm.input.messages`
  / `llm.available_tools.schemas` all present.

**Pipecat** — the voice examples need a live WebRTC/Daily call (Deepgram + a human
speaker) and **cannot run headless**, so Pipecat was validated by driving *real
Pipecat frames* (recipes lifted from `tests/integration/pipecat/`) through
`NoveumTraceObserver` against a real client, then exporting. A full turn
(STT → LLM with tools + an executed function call → TTS → metrics → error)
confirmed:
- Topology `conversation → turn → {stt, llm, tts}` exactly: stt/llm/tts
  `parent_span_id` = the turn span; turn and `full_conversation` are trace-level.
- `pipecat_span_status` is a **custom attribute**, and on error the native
  `status` stayed `"ok"` with trace `error_count: 0` — see the ETL warning in §7.
- `pipecat.error` event landed on the turn span (and the trace).
- Executed tool call → `llm.function_call_results` on the LLM span, no tool span.
- Derived cost + `conversation.total_*` rollups; sentinel `top_p="NOT_GIVEN"`
  dropped; `llm.max_tokens` kept; `full_conversation.missing_reason` on the
  audio span when no `AudioBufferProcessor` is attached.

**Environment caveat discovered during validation:** the sibling `../.venv` had a
**stale build of `noveum_trace` (same version string `1.5.17`)** whose `Trace`
lacked `.events` / `add_event`. Running the harness against the installed package
made the trace-level `pipecat.error` event silently no-op (the `AttributeError`
is swallowed by a `try/except`). Re-running against the repo working tree
(`PYTHONPATH=src`) produced the trace-level event as designed. Lesson for anyone
re-validating: import from the repo `src/` (or `pip install -e .` to refresh the
venv), not a stale installed wheel — and note that error-path appends are
wrapped in `try/except`, so a schema mismatch fails *silently*.

---

## 11. Checklist for "is this a good trace?"

- [ ] Root trace = the whole interaction, with session-wide attributes on it.
- [ ] Every independently-timed operation is its own span; the tree matches the
      real execution structure.
- [ ] Tool/function calls are attributes on the LLM span, not spans.
- [ ] Point-in-time things (errors, interim results, connection) are events.
- [ ] Each span fills all available slots: identity, timing+latency, input,
      output, tokens, cost, params, status.
- [ ] All reachable params/metrics captured; unset/sentinel values dropped.
- [ ] Cost derived locally when the model is known.
- [ ] Large collections sampled with a truncated flag; binary referenced by
      UUID + metadata, never inlined.
- [ ] Latencies captured as discrete attributes (TTFT, TTFB, VAD→final, …).
- [ ] Errors recorded with status + message + (usually) an event; cancellations
      distinguished from errors.
- [ ] Rollups (total tokens/cost/turns) on the trace.
- [ ] Keys are namespaced, dot-delimited; complex values JSON-encoded.
```
