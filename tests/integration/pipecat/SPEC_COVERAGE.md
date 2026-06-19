# Pipecat integration — spec → implementation → test coverage

Maps every item in `Noveum_Pipecat_SDK_Integration_Spec.md` and the
`.cursor/plans/pipecat-plan-tier-*` design docs to its implementation status and
the test that pins the **spec outcome** (not the plan pseudocode).

Tests assert what the spec *requires*, independent of what the code happens to
do. Where the source diverges from a required outcome, the failure is encoded as
`xfail(strict=True)` with a `file:line` reason so it auto-flips to PASS when fixed
— never weakened to match buggy code.

Legend: ✅ implemented + tested · ⚠️ implemented + divergent (xfail) · 🚧 not implemented (no test)

## Tiers (the four-commit overhaul)

| Tier | What | Status | Test file |
|---|---|---|---|
| 0 — wiring | `NoveumPipecatTracer`, two stable calls, `add_observer` (no `observers=[…]`), turn-tracking fallback | ✅ | `test_tracer_wiring.py` |
| 1 — auto-ABP | `observe_pipeline` appends `AudioBufferProcessor(num_channels=2)`, rebuild-and-return, idempotent, order-preserving | ✅ | `test_tracer_abp_insertion.py` |
| 2 — transport tap | `register_task_handlers` monkeypatches `transport.input().push_audio_frame` for raw pre-filter audio (no class-swap) | ✅ | `test_tracer_raw_audio_tap.py` |
| 3 — custom spans | `NoveumCustomSpanProcessor` folds plain-OTEL customer spans under the active turn; provider own/add modes; drain on finish | ✅ | `test_custom_spans.py` |

## Caveat register (C-items) and strategic upgrades (S-items)

| Item | Requirement | Status | Test |
|---|---|---|---|
| **C1** | Auto-inject `AudioBufferProcessor`; full-conversation audio with no manual edit | ✅ | `test_tracer_abp_insertion.py` |
| **C2** | Custom spans captured automatically (OTEL) | ✅ | `test_custom_spans.py` |
| **C3** | STT capture without the transport class-swap | ✅ | `test_tracer_raw_audio_tap.py` |
| **C4** | Auto-set `enable_metrics` / `enable_usage_metrics` | ✅ | `test_tracer_wiring.py` |
| **C5** | Custom LLM processor attrs via duck-typing + registration hook | 🚧 not implemented | — |
| **C6** | User audio captured before custom processors swallow it | ✅ (transport tap) | `test_tracer_raw_audio_tap.py` |
| **C7** | Fragile `attach_to_task` ordering folded into the tracer | ✅ | `test_tracer_wiring.py` |
| **C8** | Interrupted-STT partial transcript (version ≥1.6.x) | 🚧 version/ops, not this branch | — |
| **C9** | Startup version-compat check (`_compat`) | 🚧 not implemented | — |
| **S1** | Pipeline-wrapping single call (`observe_and_create_task`) | ✅ | `test_tracer_wiring.py` |
| **S2** | Auto stereo channel assignment (`num_channels=2`) | ✅ | `test_tracer_abp_insertion.py` |
| **S3** | Prod vs test modes (`observe_*` / `track_*`, `mode="track"`) | 🚧 not implemented | — |
| **S4** | Copy-paste coding-agent prompt | 🚧 docs, not code | — |
| **S5** | Deferred upload / consent gating (`defer_upload`) | 🚧 not implemented | — |
| **S6** | `set_custom_metadata` API | 🚧 not implemented | — |
| **S7** | Migration + concurrency guidance | 🚧 docs, not code | — |

## Beyond the spec (implemented; lightly covered)

| Feature | Status | Test |
|---|---|---|
| Session metadata (`capture_session_metadata`, `_store_transport`/`_flush`) | ✅ | `test_observer_session_metadata.py` |
| `capture_errors` / `capture_system_logs` flags | flag-forwarding only | `test_tracer_wiring.py` (constructor) |

## Regression / "existing functionality not broken"

| Guarantee | Test |
|---|---|
| Legacy `observers=[obs]` + `attach_to_task` path still attaches + detects ABP | `test_tracer_parity_regression.py` |
| `observe_pipeline` never reorders customer processors (append-only) | `test_tracer_parity_regression.py`, `test_tracer_abp_insertion.py` |
| Tracer path ≡ manual path (proxy delivery + ABP detection) | `test_tracer_parity_regression.py` |
| `setup_pipecat_tracing` factory unchanged | `test_tracer_parity_regression.py` |

## Source bugs found by these tests — now FIXED (regression-tested)

All three were first encoded as `xfail(strict=True)`, then fixed in source; the
tests are now plain passing regression guards.

1. **Raw-audio tap is now side-band** — `tracer.py`. The `_patched` wrapper now
   wraps `capture_raw_input_audio(frame)` in `try/except`, so a capture failure
   can never propagate onto the audio path (Tier-2 §3 / non-intrusive guarantee).
   Test: `test_tracer_raw_audio_tap.py::test_capture_exception_must_not_reach_audio_path`.

2. **Errored custom span serialises** — `custom_spans.py`. Now passes
   `SpanStatus.ERROR` (not the bare string `"error"`) to `Span.set_status`, so
   `Span.to_dict()` no longer raises on `status.value` when the errored customer
   span is exported.
   Test: `test_custom_spans.py::test_errored_custom_span_is_recorded_and_serialisable`.

3. **Raw-audio tap is now idempotent** — `tracer.py`. The tap sets a
   `_noveum_tap_applied` flag on the input-transport instance and skips if already
   set, so re-registering on the same transport no longer stacks wrappers /
   double-captures frames (Tier-2 plan §3).
   Test: `test_tracer_raw_audio_tap.py::test_tap_is_idempotent`.

## Known limitations (noted, not failed)

- **C4 can't distinguish unset from explicit `False`** — `PipelineParams` defaults
  both flags to `False`, so the auto-enable flips any falsy value to `True`,
  including a deliberate `enable_metrics=False`. The spec's intent (metrics never
  silently missing) is met; the plan's "never clobber explicit values" is
  unreachable without a tri-state. Covered by
  `test_tracer_wiring.py::test_metrics_flags_*`.
- **Tier-2 wrapper deprecation not started** — `transports.py` still ships the 11
  `Noveum*Transport` classes with **no** `DeprecationWarning` (plan §4 schedules
  one). They still work and are exercised by `test_pipecat_transports.py`.
- **Custom spans are single-conversation-per-process (v1)** — per plan §0a; not a
  bug, a stated scope boundary.
