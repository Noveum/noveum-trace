# CrewAI Integration — Test Plan & Regression Suite Design

**Target:** `src/noveum_trace/integrations/crewai/` (~12,800 LOC across 12 handler mixins + `crewai_utils.py`, `crewai_state.py`, `crewai_listener.py`).
**Existing suite:** `tests/unit/integrations/test_crewai_integration.py` (~2,340 LOC, 25 test classes).
**Cross-checked against:** the real CrewAI event classes in `../crewAI/lib/crewai/src/crewai/events/types/*.py` (the version the integration targets, `crewai>=0.177.0`).

This plan exists to serve the two stated goals:

1. **Verify the integration actually works** — *not satisfiable by the current suite* (see §1).
2. **Regression protection** — catch upstream CrewAI field/event drift and our own refactors.

---

## ✅ Implementation status (2026-06-22) — IMPLEMENTED & GREEN

crewAI **1.14.2a2** installed editable into `../.venv`. **149 new tests written, all passing** (alongside the 196 existing → 345 green); black/isort/ruff clean. Files:

| File | Tests | Scope |
|---|---|---|
| `test_crewai_utils.py` | 64 | pure utils (serialize, tokens, cost, messages, durations) |
| `test_crewai_re_guardrail_task.py` | 7 | guardrail + task eval (bugs) |
| `test_crewai_re_crew_agent.py` | 8 | crew (correct) + agent (bugs) |
| `test_crewai_re_llm_tool_memory.py` | 12 | llm/tool/memory lifecycles + truncate_str bug |
| `test_crewai_re_knowledge_mcp_reasoning.py` | 28 | knowledge/mcp/reasoning (many bugs) |
| `test_crewai_re_a2a_flow.py` | 26 | a2a + flow (bugs) + dead-handler confirmation |
| `test_crewai_bus_e2e.py` | 4 | bus routing + real `kickoff()` e2e span tree |

All §2 findings confirmed against the installed version; §2D fallbacks refuted as bugs (they work). **New bugs found beyond §2** (also baselined): memory.type always the event literal; **MCP and reasoning started↔completed pairing broken** (`_resolve_mcp_key`/`_resolve_reasoning_id` omit event_id); mcp.url/arguments field drift; observation `step_description` drift; replan_reason dropped; `FlowFailedEvent` also absent; a2a.status delegation/conversation asymmetry. The integration **works end-to-end** (real kickoff → crew→task→agent tree with correct nesting + status).

⚠️ Installing crewai upgraded `opentelemetry-sdk` to 1.34.1 (crewai pins `~=1.34.0`), which **breaks `import livekit.agents` in `../.venv`** → livekit tests now skip there. pipecat unaffected.

---

## 0. How to run (read first — affects the word "verify")

- **CrewAI is not installed in any local venv** (`../.venv`/`.venv` are Python 3.13; system Python is 3.10.12; none import `crewai`). The *entire* existing crewAI test file is gated by `pytest.importorskip("crewai")` (line 40) because the listener does `from crewai.events import BaseEventListener`. **Nothing in the crewAI suite — old or new — runs until CrewAI is installed.**
- To actually *verify*: `pip install crewai>=0.177.0` (or `pip install -e ../crewAI/lib/crewai`) into a 3.13 venv, then `pytest tests/unit/integrations/test_crewai_integration.py`. Until then this is a *plan*, not a passing suite.
- All new **real-event / bus / e2e** tests must be marked `@pytest.mark.crewai` (declared in `pyproject.toml`) so CI can gate them on the `crewai` extra + Python ≥3.10. Pure **unit** tests of `crewai_utils.py` still need `crewai` importable only because the module imports it at top level — keep them in the same gated file or a sibling gated module.

---

## 1. Strategy: why the current suite gives false confidence

**Every existing test feeds the handlers `MagicMock` events.** A `MagicMock` returns a child mock for *any* attribute access, so a handler that reads `event.score` "works" even if the real event has no `score` field. The existing tests even fight this by hand — e.g. `ev.crew_id = None  # MagicMock would auto-create truthy placeholders`. That manual nulling is the tell: the tests encode the handler's *assumptions*, not CrewAI's *reality*.

**Consequence:** the suite cannot detect the single most important class of failure — a mismatch between the field a handler reads and the field the real CrewAI event actually carries. That is exactly what both stated goals need.

**The fix — make real-event contract tests the spine.** Construct the *real* Pydantic event (`crewai.events.types.*`) with real field values, feed it to the handler, assert the resulting span attributes/status. These cannot silently pass on a renamed field, and they double as the regression tripwire for upstream drift. Build each from the *actual event class definition*, never from the handler's assumptions.

**Proof this matters — two bugs already found and hand-verified during this analysis** (see §2): `truncate_str` is a no-op, and **task evaluation captures nothing from a real event** even though `TestTaskEvaluationScore` passes green.

---

## 2. 🔴 Latent bugs found by the field cross-check (highest-value output)

Each handler reads event fields defensively via `safe_getattr(...)`, so a wrong field name **does not crash** — the attribute is simply *never set* and the trace silently loses data. The analysis diffed "fields each handler reads" vs "fields the real event class defines" and adversarially re-verified each hit. Findings below are grouped by confidence/impact.

> **Caveat:** confirmed against the `../crewAI` working copy. A few *could* be version drift if the integration was written against a different CrewAI release. Treat each as **"confirm against your pinned version, then either fix the handler or baseline current behavior with a test."** Precedent in this repo: the LangChain known-bugs were *baselined by tests, not fixed* (see memory `project_langchain_known_bugs`) — same pattern applies here unless you choose to fix.

### 2A. Confirmed data-loss — handler reads a wrong/nonexistent field; real field exists or data is dropped

| # | Area | Handler reads | Real event / field | Effect |
|---|------|---------------|--------------------|--------|
| 1 | **task eval** | `event.score`, `.feedback`, `.model`, `.criteria`, `.result.score` | `TaskEvaluationEvent` defines only `type, evaluation_type, task` | **All `task.evaluation_*` attrs silently never written.** `TestTaskEvaluationScore` passes only via MagicMock. *(hand-verified)* |
| 2 | **guardrail** | `event.validation_success` (→`passed`/`accepted`/`valid`) | `LLMGuardrailCompletedEvent.success: bool` | Guardrail pass/fail signal lost; spans report `ok` even on rejection. |
| 3 | **guardrail** | `event.results` (plural), `.checks` | real field is `result` (singular) | Check results lost. |
| 4 | **guardrail** | always passes `error=None` to finish; ignores `event.error` | `LLMGuardrailCompletedEvent.error: str\|None` | On `success=False`, error message dropped + span marked success. |
| 5 | **guardrail** | `event.call_id`/`llm_call_id`/`run_id`; `event.input`/`output`/`llm_output`/`text` | none exist on `LLMGuardrailStartedEvent` | `llm.call_id` and `guardrail.input` never set. |
| 6 | **knowledge** | `event.query`/`search_query`/`text`/`input` | `KnowledgeQueryStartedEvent.task_prompt` | Query text never captured. |
| 7 | **knowledge** | `event.results`/`content`/`chunks`/`documents` | `KnowledgeRetrievalCompletedEvent.retrieved_knowledge`; `KnowledgeQueryCompletedEvent` has none | Retrieved knowledge + enrich/scores never run. |
| 8 | **knowledge** | `event.sources`/`top_k`/`limit`/`sources_used` | not defined | All knowledge metadata lost. |
| 9 | **knowledge** | treats `error` as exception (`__traceback__`, `type(e).__name__`) | `error: str` | Error type = `'str'`; no traceback. |
| 10 | **flow** | `event.value`/`input` (input received); `event.field`/`field_name` | `FlowInputReceivedEvent.response` | Human-input value silently lost; `flow.input_field` never set. |
| 11 | **flow** | `event.possible_outcomes`/`outcomes` (paused) | `FlowPausedEvent.emit` | Pause outcomes lost. |
| 12 | **a2a** | `event.status`, `event.attempt` (polling) | `A2APollingStatusEvent.state`, `.poll_count` | Polling status + attempt count lost. |
| 13 | **a2a** | `event.auth_method` | `A2AAuthenticationFailedEvent.auth_type` | Auth method lost. |
| 14 | **agent** | `event.iterations`/`step` (completed) | `AgentExecutionCompletedEvent` has `agent, task, output` only | Iteration count lost. |
| 15 | **agent** | source attrs `role/goal/backstory/...` for **Lite** agent | `LiteAgentExecutionStartedEvent.agent_info` (dict) | Lite-agent identity lost (it's a dict, not an Agent obj). |
| 16 | **agent** | `event.passed`, `event.result`, `event.model`/`evaluator_model` (eval completed) | `AgentEvaluationCompletedEvent` has `score, metric_category, iteration,...` (no `passed/result/model`) | Eval pass/model lost (note: `score` *does* exist — read it). |
| 17 | **agent** | `event.criteria` (eval started) | `AgentEvaluationStartedEvent` has no `criteria` | Eval criteria lost. |
| 18 | **mcp** | `event.tools`/`available_tools` (conn completed) | `MCPConnectionCompletedEvent` has no tools field | `mcp.available_tools`/`tool_count` never set. |
| 19 | **mcp** | `event.config` (config-fetch failed) | `MCPConfigFetchFailedEvent` has `slug, error, error_type` | Config snapshot lost. |
| 20 | **reasoning** | `event.is_ready` (reasoning completed) | `AgentReasoningCompletedEvent.ready` | `reasoning.is_ready` **never set** (HIGH). |
| 21 | **reasoning** | `event.completed_steps` (replan) | `PlanReplanTriggeredEvent.completed_steps_preserved` (int) | Lost + type mismatch — handler treats it as a list/JSON, real field is an int (HIGH). |
| 22 | **reasoning** | `event.step`/`current_step` (goal-early) | `GoalAchievedEarlyEvent.step_number` (from `ObservationEvent`) | Step correlation lost (MEDIUM). Also: `reasoning_started` reads `is_ready` which doesn't exist on the *started* event. |

### 2B. Confirmed broken utility (affects every handler)

- **`truncate_str(text, max_len)` is a no-op** (`crewai_utils.py:181-185`): docstring *"Return full string content"*, body `return text` — `max_len` ignored everywhere. **No payload (LLM responses, tool args/outputs, task results, feedback, memory previews) is ever truncated** → unbounded span attributes, memory and transport bloat. *(hand-verified)*. The handler-level constants (`MAX_TEXT_LENGTH=8192`, etc.) are silently ineffective wherever they route through `truncate_str`.

### 2C. Dead-code handlers — subscribed event class does not exist in this CrewAI version (by design)

`setup_listeners` isolates each of these in its own `try/except ImportError` (see `crewai_listener.py:1258-1259` comment), so they're **harmlessly skipped, not a crash** — but the handlers never fire:

- `on_a2a_delegation_failed` (`A2ADelegationFailedEvent`), `on_a2a_conversation_failed` (`A2AConversationFailedEvent`), `on_a2a_message_received` (`A2AMessageReceivedEvent`), `on_a2a_streaming_completed` (`A2AStreamingCompletedEvent`), `on_llm_guardrail_failed` (`LLMGuardrailFailedEvent`).
- **Real failure path:** failures arrive via the corresponding `*Completed` event with `status="failed"` / `success=False`. The existing `test_conversation_completed_failed_sets_error_status` proves the conversation path; the others need equivalent coverage (§3). **Action:** lock the live path with a test; optionally delete the dead handlers or keep for forward-compat.

### 2D. Reads a nonexistent *event* field but recovers via a real fallback (NOT bugs — but pin the fallback)

These are correct-by-fallback: the field isn't on the event, but the handler then reads it from `source`/`llm`/the tool object. Real-event tests should assert the fallback path so a future refactor can't break it silently.

- **llm:** `temperature/max_tokens/top_p/seed` (→ `source`/`llm` obj), `finish_reason` (→ response parse), `streaming` (→ llm obj).
- **tool:** `event.tool` (→ source), `tool_description` (→ tool obj), `task_description` (→ source.task), `tool_input/arguments/args` (→ real `tool_args`).
- **memory:** retrieval-completed fallbacks `context/memories/results` are dead but `memory_content` (required) is always read first — no loss.
- **crew:** `event.output` on `CrewTestCompletedEvent` (None by design; quality comes from `CrewTestResultEvent`).

**Every item in §2A/2B/2C gets a dedicated test in §3/§4 (marked 🐛).** Decide per-item: assert the *correct* behavior (and `xfail`/`skip` until fixed) **or** baseline the *current* behavior with a comment pointing here.

---

## 3. Real-event contract tests — the regression spine (by family)

Construct the **real Pydantic event** and assert span attributes/status/lifecycle. P0 = must-have; P1 = high value; 🐛 = pins a §2 finding; ⬆ = upgrades a named MagicMock test to catch field drift.

### 3.1 Crew (`_handlers_crew.py`)
- **P0** `test_kickoff_failed_string_error_parsing_with_type_prefix` — `CrewKickoffFailedEvent(error='ValueError: bad config')` → `error.type='ValueError'`, `error.message='bad config'`, `crew.status='error'`, `set_status(ERROR)`, span removed + trace closed. (`error` is a *string*, not Exception — the only branch real events hit.) ⬆ `test_kickoff_failed_clears_crew_span`.
- **P0** `test_kickoff_failed_error_no_colon_fallback` — `error='connection timeout'` → `error.type='Error'`, full string as message.
- **P0** `test_kickoff_completed_aggregates_tokens_cost_duration` — pre-seed `_total_tokens_by_crew`/`_total_cost_by_crew`; `CrewKickoffCompletedEvent(output=...)` → `crew.output`, `crew.total_tokens`, `crew.total_cost`, `crew.duration_ms>0`, `crew.status='ok'`, `trace.finish()`. ⬆ `test_kickoff_completed_clears_crew_span`.
- **P1** `test_train_failed_string_error_sets_mode_train` — `CrewTrainFailedEvent` with no prior `train_started` → `crew.mode='train'` + error attrs, no raise.
- **P1** `test_full_test_lifecycle_started_result_completed` — real `CrewTestStartedEvent`→`CrewTestResultEvent`→`CrewTestCompletedEvent`; asserts mode/n_iterations/eval_llm/inputs, quality_score/test_model/execution_duration_s, then `status='ok'` + teardown. ⬆ existing test/result tests (adds completion phase).
- **P1** `test_kickoff_snapshot_gating` — `capture_agent_snapshot=False, capture_crew_snapshot=False` → identity (`crew.id/name/process`) kept, bulky `*_snapshot`/`agent_roles` omitted.

### 3.2 Task (`_handlers_task.py`)
- **P0 🐛** `test_task_evaluation_real_event_extracts_nothing` *(or asserts correct after fix)* — `TaskEvaluationEvent(evaluation_type='score', task=...)` → `_extract_evaluation_attributes` returns no `task.evaluation_*`. **Documents finding #1.** ⬆ `TestTaskEvaluationScore` (which passes only via MagicMock).
- **P0** `test_task_started_task_id_from_baseevent_field` — `TaskStartedEvent(task_id='t1', task=...)` → `_task_spans['t1']`, `task.id='t1'`; documents id comes from inherited `BaseEvent.task_id`.
- **P0** `test_task_failed_parses_error_string` — `TaskFailedEvent(error='ValueError: x')` → `error.type='ValueError'`, `task.status='error'`, `set_status(ERROR)`.
- **P0** `test_task_started_completed_pairing_two_concurrent` — two real `TaskStartedEvent` (`t1`,`t2`); complete `t1`, assert only `t2` remains, no orphan. ⬆ `test_task_completed_removes_span`.
- **P1** `test_task_completed_extracts_TaskOutput_raw` — `TaskCompletedEvent(output=TaskOutput(raw='...'))` (real class) → `task.output` from `.raw`, `task.status='ok'`, `task.duration_ms`.
- **P1** `test_task_capture_inputs_false_omits_description_context` / `test_task_capture_outputs_false_omits_output` — privacy flags vs real events. ⬆ the two MagicMock gating tests.
- **P1** `test_task_eval_falls_back_to_crew_span_when_task_closed` — eval after task span closed → attaches to crew span (if/when #1 fixed).

### 3.3 Agent (`_handlers_agent.py`)
- **P0** `test_agent_started_full_identity` — real `AgentExecutionStartedEvent` with agent (role/goal/backstory/llm/tools) + task → asserts `agent.id/role/goal/backstory/llm_model/task_prompt/type='full'` + `available_tools.*`. ⬆ `TestAgentHandlers` (existence-only).
- **P0 🐛** `test_lite_agent_lifecycle_uses_agent_info_dict` — `LiteAgentExecutionStartedEvent(agent_info={'id':'lite-1','role':'Helper'}, ...)` → `agent.type='lite'`, identity sourced from `agent_info` dict (finding #15). No existing lite coverage.
- **P0 🐛** `test_agent_eval_completed_reads_score_not_passed_result` — `AgentEvaluationCompletedEvent(score=0.95, metric_category=..., iteration=1)` → `score` captured; `passed`/`result`/`model` absent (finding #16).
- **P1** `test_agent_completed_output_no_iterations` — `AgentExecutionCompletedEvent(output=...)` → `agent.output` set, `agent.iterations` *absent* (finding #14). ⬆ `test_agent_completed_removes_span`.
- **P1** `test_agent_error_exception_details` — `AgentExecutionErrorEvent(error=RuntimeError(...))` → `error.type/message`, `agent.status='error'`, `set_status(ERROR)`. ⬆ `test_agent_error_removes_span`.
- **P1 (behavioral)** `test_orphan_llm_tool_spans_closed_on_agent_finish` — open LLM+tool spans for an agent, complete the agent → children force-closed/finished, no leak.
- **P1** `test_agent_capture_flag_matrix` — inputs/outputs/snapshot flags gate `task_prompt`/`tool_names`/`goal`/`backstory`/`output`.
- **P1** `test_agent_span_duration_monotonic` — uses `duration_ms_monotonic`, ~sleep(0.1) → ~100ms.

### 3.4 LLM (`_handlers_llm.py`)
- **P0** `test_llm_started_core_attrs` — `LLMCallStartedEvent(call_id, model='claude-3-5-sonnet', messages=[sys,user], tools=[])` → `llm.call_id/model/provider='anthropic'/system_prompt/input_messages`.
- **P0** `test_llm_completed_response_tokens_cost` — `LLMCallCompletedEvent(usage={prompt_tokens,completion_tokens,total_tokens}, response=..., call_type=LLMCallType.LLM_CALL)` → `llm.response`, token attrs, `llm.cost.total>0`, `llm.call_type='LLM_CALL'`, finished. ⬆ `test_llm_completed_consumes_token_buffer`.
- **P0** `test_llm_failed_error_status` — `LLMCallFailedEvent(error='rate_limit')` (required field) → `set_status(ERROR)`, error attrs. ⬆ `test_llm_failed_removes_span`.
- **P1 (fallback)** `test_llm_sampling_params_from_source` — event lacks `temperature/max_tokens/top_p/seed`; read from source obj (finding §2D). ⬆ `TestLLMTemperature`.
- **P1** `test_llm_stream_chunk_buffers_and_joins` / `test_llm_thinking_chunk_buffers_and_joins` — real `LLMStreamChunkEvent`/`LLMThinkingChunkEvent` with `chunk=...`, `capture_streaming/thinking=True` → joined into `llm.streaming_response`/`llm.thinking`. ⬆ the two chunk tests.
- **P1** `test_llm_token_merge_layered` — response.usage incomplete + `event.usage` fills missing output tokens.
- **P1** `test_llm_parent_resolution_agent_then_task` — parent = agent span when `agent_id` present, else task span via `_task_to_crew_id`.
- **P2** `test_llm_cache_reasoning_tokens_from_usage` — `usage={cache_read_tokens, reasoning_tokens}` → provider-extra attrs.

### 3.5 Tool (`_handlers_tool.py`)
- **P0** `test_tool_started_all_attrs` — `ToolUsageStartedEvent(tool_name, tool_args, tool_class, agent_role, task_name, run_attempts=0, delegations=None)` → identity/correlation + `tool.input` JSON.
- **P0** `test_tool_run_id_correlation_via_event_id` — start keyed by `event_id`, finish via `started_event_id` → span removed. ⬆ `test_tool_finished_removes_span`.
- **P0** `test_tool_run_attempts_zero_not_swallowed` — `run_attempts=0` preserved (not dropped by `or`). ⬆ `test_tool_run_attempts_zero_captured`.
- **P1** `test_tool_finished_output_status` / `test_tool_error_sets_error_no_output` — `ToolUsageFinishedEvent(output=...)` vs `ToolUsageErrorEvent(error=ValueError(...))` → status ok/error, `set_status(ERROR)`. ⬆ `test_tool_error_removes_span`.
- **P1** `test_tool_validate_input_error_annotates_agent_span` / `test_tool_selection_error_includes_chosen` / `test_tool_execution_error_tool_vs_agent_span` — the orphan-routing branches for `ToolValidateInputErrorEvent`/`ToolSelectionErrorEvent`/`ToolExecutionErrorEvent`.
- **P1** `test_tool_capture_inputs_false_omits_args`.
- **P2 🐛** `test_tool_output_truncation` — feed a >8192-char output and assert truncation. **Will currently fail** because of `truncate_str` (finding §2B) — use as the driver to fix it, or `xfail`.

### 3.6 Memory (`_handlers_memory.py`)
- **P0** `test_memory_query_started_all_attrs` — `MemoryQueryStartedEvent(query, limit, score_threshold, type='short_term')` → `memory.operation='query'`, query/limit/threshold/op_id.
- **P0** `test_memory_query_completed_count_preview` — `MemoryQueryCompletedEvent(results=[...], started_event_id=op)` → `result_count`, `results_preview`, `status='ok'`, span removed.
- **P0** `test_memory_query_failed_error_status`, `test_memory_save_started/completed/failed`, `test_memory_retrieval_started/completed/failed` — full save & retrieval lifecycles (none currently tested). Retrieval-completed asserts `memory_content` (the only payload field) — finding §2D dead fallbacks documented.
- **P0** `test_memory_respects_capture_memory_false` — all memory handlers no-op.
- **P1** `test_memory_pairing_via_started_event_id` — correlation across started/completed (without it, span lives until trace end → wrong `duration_ms`).
- **P1** `test_memory_parent_agent_then_task`, `test_memory_completed_orphan_safe`, `test_memory_query_completed_string_results`, `test_memory_save_dict_value_json`.
- **P1 (concurrency)** `test_memory_concurrent_no_state_corruption` — 10 threads × 5 start/complete pairs, dict empties cleanly.

### 3.7 Flow (`_handlers_flow.py`)
- **P0** `test_flow_started_inputs_state`, `test_method_started_name_and_type` (infer `@start`/`@listen`/`@router` from Flow metadata), `test_method_finished_reads_result` (real field is `result`, not `output`).
- **P1 🐛** `test_flow_input_received_reads_response` — `FlowInputReceivedEvent(response='user input')` → `flow.input_value` from `response` (finding #10); `flow.input_field` absent.
- **P1 🐛** `test_flow_paused_reads_emit` — `FlowPausedEvent(emit=['continue','cancel'])` → `flow.pause_possible_outcomes` from `emit` (finding #11).
- **P1** `test_flow_input_requested_reads_message`, `test_human_feedback_requested_reads_message`, `test_human_feedback_received_reads_feedback`.
- **P1** `test_flow_nested_in_crew_attaches_to_crew_span` (trace=None for nested), `test_method_failed_error_status`, `test_flow_failed_error_status`. ⬆ `test_flow_started_creates_flow_span_entry`.

### 3.8 Knowledge (`_handlers_knowledge.py`) — currently **0 tests**, multiple §2A bugs
- **P0 🐛** `test_knowledge_query_started_captures_task_prompt` — real field `task_prompt` (finding #6). Currently `memory.query` never set.
- **P0 🐛** `test_knowledge_retrieval_completed_reads_retrieved_knowledge` — `retrieved_knowledge` (finding #7); assert `result_count`/preview.
- **P0** `test_knowledge_retrieval_started_opens_span` (type='knowledge', op_id), `test_knowledge_query_failed_error_status` (error is *string*, finding #9), `test_knowledge_op_pairing_via_started_event_id`.
- **P1** `test_knowledge_search_query_failed_annotates_open_span`, `test_knowledge_no_parent_when_agent_missing`, plus unit tests for `_enrich_results` (string vs list) and `_enrich_scores` (mixed/null scores).

### 3.9 Guardrail (`_handlers_guardrail.py`) — multiple §2A bugs
- **P0 🐛** `test_guardrail_completed_rejection_captured` — `LLMGuardrailCompletedEvent(success=False, result=..., error='disallowed', retry_count=2)` → asserts pass/fail + error captured + `status='error'`. **Primary regression test for findings #2-#4.** ⬆ `test_guardrail_completed_pairs_via_started_event_id` (which uses the *wrong* field `validation_success` and masks the bug).
- **P0** `test_guardrail_started_named_function_identity` — `LLMGuardrailStartedEvent(guardrail=<fn>, retry_count=0)` → `guardrail.name` from `__name__`, `guardrail.type`, agent role.
- **P1** `test_guardrail_started_to_completed_pairing`, `test_guardrail_retry_sequence_distinct_spans` (retry_count 0/1/2, success F/F/T), `test_guardrail_parent_fallback_to_agent`.

### 3.10 MCP (`_handlers_mcp.py`) — currently **0 tests**
- **P0** `test_mcp_connection_started_server_details`, `test_mcp_connection_failed_error_type`, `test_mcp_tool_execution_full_lifecycle` (started→completed via stable `mcp_key`).
- **P0** `test_mcp_config_redaction_sanitizes_credentials` — `_redact_config` must replace `api_key`/`Authorization`/`secret` with `<redacted>` before writing `mcp.config`. **Security-relevant.**
- **P1 🐛** `test_mcp_connection_completed_no_tools_field` — real event has no `tools`; assert `mcp.available_tools`/`tool_count` absent + duration from `start_t` (finding #18).
- **P1** `test_mcp_tool_execution_failed_error_type`, `test_mcp_config_fetch_failed_annotates_agent_span` (finding #19), `test_mcp_key_isolation_concurrent`.

### 3.11 A2A (`_handlers_a2a.py`) — 23 handlers, only 3 tests today
- **P0** `test_a2a_delegation_started_all_attrs`, `test_a2a_delegation_completed_ok`, `test_a2a_delegation_completed_failed` (status='failed' → ERROR; the live failure path, since `A2ADelegationFailedEvent` is dead — §2C). ⬆ `test_conversation_completed_failed_sets_error_status`.
- **P0** `test_a2a_conversation_started_separate_span_and_buffers`, `test_a2a_conversation_completed_success_flushes_streaming`, `test_a2a_conversation_completed_failed`.
- **P0 🐛** `test_a2a_polling_status_field_mismatch` — `A2APollingStatusEvent(state='pending', poll_count=2)` → assert `a2a.polling_status`/`a2a.polling_attempt` currently **None** (findings #12). Known-issue marker; flips to real assertions when fixed.
- **P0 🐛** `test_a2a_auth_failed_field_mismatch` — `A2AAuthenticationFailedEvent(auth_type='bearer')` → `a2a.auth_method` currently None (finding #13); `error.type='AuthenticationError'` still set.
- **P0** `test_a2a_connection_error_captures_type_endpoint`.
- **P1** message-sent/response-received buffering, streaming-chunk accumulate+flush-on-final, artifact metadata (non-image) ⬆ `test_artifact_received_image_calls_export_image`, chunked-image accumulation→`export_image`, server-task started/failed, context-lifecycle snapshot, composite-key coexistence (delegation + conversation share `context_id`).

### 3.12 Reasoning (`_handlers_reasoning.py`) — currently **0 tests** (only the gate is tested); 3 HIGH/MEDIUM §2A bugs
9 handlers: reasoning started/completed/failed, step-observation started/completed/failed, and 3 mid-reasoning annotations (plan_refinement, replan_triggered, goal_achieved_early — these annotate the open reasoning span, they do **not** open/close one).
- **P0** `test_reasoning_lifecycle_full_cycle` — `AgentReasoningStartedEvent(agent_role, task_id, attempt=2)` → span with `reasoning.id/attempt/agent.role/task.id`; then `AgentReasoningCompletedEvent(plan=..., ready=True)` → `reasoning.final_plan`, `reasoning.status='ok'`, `duration_ms>0`, span removed. **Asserts `reasoning.is_ready` is set from `ready`** — currently fails (finding #20).
- **P0** `test_step_observation_lifecycle_with_refinements` — `StepObservationStartedEvent(step_number=1, step_description=...)` → `step.number=1`/description; `StepObservationCompletedEvent(suggested_refinements=[...])` → `step.suggested_refinements` JSON, `step.status='ok'`, duration.
- **P0 🐛** `test_goal_achieved_early_reads_step_number` — `GoalAchievedEarlyEvent(steps_remaining=5, steps_completed=2, step_number=2)` → `reasoning.goal_achieved_early=True`, `reasoning.steps_remaining=5`; **asserts step correlation from `step_number`** (currently reads `step`/`current_step` → lost, finding #22).
- **P1 🐛** `test_replan_triggered_completed_steps_preserved` — `PlanReplanTriggeredEvent(replan_count=2, replan_reason=..., completed_steps_preserved=3)` → `reasoning.replan_count=2` (int), `reasoning.replan_reason`; **asserts `completed_steps_preserved` (int) captured** (currently reads `completed_steps` and JSON-treats it, finding #21). Span NOT closed (annotation only).
- **P1** `test_reasoning_failed_error_status` — `AgentReasoningFailedEvent(error='...')` → `reasoning.status='error'`, `set_status(ERROR)`, span removed.
- **P1** `test_plan_refinement_annotates_open_span` — `PlanRefinementEvent(refinements=[...], refined_step_count=3)` → `reasoning.refined_steps` JSON + count; span **not** closed.
- **P1** `test_capture_reasoning_false_gates_all_nine_handlers` — `capture_reasoning=False`; call all 9 with real events → `_reasoning_spans`/`_observation_spans` stay empty. ⬆ `test_capture_reasoning_false_skips_reasoning_handlers` (extends from 1 handler to all 9 with real events).

**Fix-or-baseline (HIGH):** `is_ready`→`ready` (`:195`), `completed_steps`→`completed_steps_preserved` + int handling (`:484`), `step`→`step_number` in goal-early (`:547`).

---

## 4. `crewai_utils.py` — pure-function unit tests (consolidated; currently 0 direct tests)

Dense, fast, deterministic — the cheapest high-value coverage. Consolidate the granular proposals into these groups (one parametrized test per group where sensible):

- **`safe_serialize` / `_safe_serialize_inner`** (P0): max-depth truncation marker; circular refs (dict/list/object `__dict__`); pydantic v2 `model_dump` and v1 `dict()`; custom `to_dict`; `__dict__` public-only (excludes `_private`/dunders); bytes/set → `str()`; outer exception → `<serialization_error:Type:msg>`.
- **`extract_token_usage` / `_probe_token`** (P0): OpenAI / Anthropic / Vertex / Bedrock / Watsonx / Google-genai shapes; **path precedence** (Anthropic `input_tokens` wins over OpenAI `prompt_tokens` when both present); computes `total` when missing; all-None on empty / `None` response; non-int values skipped safely.
- **`extract_finish_reason`** (P0): OpenAI `choices[0].finish_reason`, Anthropic `stop_reason`, Gemini `candidates[0].finish_reason`.
- **`extract_response_text`** (P0): OpenAI `choices[].message.content`, Anthropic content-list & content-string, Gemini `candidates[].content.parts[].text`.
- **`extract_system_prompt`** (P0): dict & object messages; multiple system msgs joined; case-insensitive role; Anthropic content-block list; empty → None.
- **`count_messages_by_role` / `messages_to_json`** (P0): dict vs object messages; role lowercasing; empty handling.
- **`serialise_tools_list` / `serialize_tool_schema` / `merge_available_tools_attributes`** (P0): dict tools, object tools (name/description), callables; merge sets count/names/descriptions/schemas.
- **`calculate_llm_cost`** (P0): known model → cost dict; unknown → `{}`; None tokens → 0.
- **`duration_ms` / `duration_ms_monotonic`** (P0): explicit end; negative clamped to 0; rounded 3dp; None end → now.
- **`resolve_agent_id` / `safe_getattr`** (P0): priority order; dotted-chain & dotted-string; dict `.get`; None-in-chain/missing → default.
- **P1 🐛 `test_truncate_str_is_noop`** — `truncate_str('x'*10000, 100)` returns length 10000. **Documents finding §2B.** Replace with a "truncates to ≤max_len" assertion if/when fixed.

`crewai_state.py` is exercised transitively by §3/§5; add direct tests only for `_migrate_legacy_prefixed_span_maps` (see §5).

---

## 5. Listener lifecycle / state / token-patch — gap tests

(`TestListenerInit/TokenPatch/ShutdownIdempotency/HandlersNeverRaise/NoOpWhenNotInitialized/CreateChildSpan/ConcurrentCrews` already exist — these fill the gaps.)

- **Token buffer LRU** (P1): FIFO eviction order (not random); `move_to_end` on duplicate write keeps entry; exact 512 boundary (no off-by-one); concurrent eviction stays ≤512 and uncorrupted.
- **Token patch lifecycle** (P1): interleaved shutdowns — patch stays applied until the *last* listener shuts down; idempotent apply under thread contention (no nested wrappers / lost original).
- **`_create_child_span` resolution** (P1): `task_id` hint → `_task_to_crew_id` picks correct crew with multiple open; explicit `crew_id` overrides task mapping; last-resort fallback works with exactly one open trace, returns **None** with 2+ (no contamination); `parent_span.trace_id` looked up in `_active_traces`.
- **Capture-flag → registration gating** (P1): `capture_memory/knowledge/a2a/mcp/flow/reasoning/guardrails=False` removes those handlers from `_handlers` *at registration time*; handler count varies by flag matrix.
- **Shutdown** (P1/P2): force-closes dangling crew/flow spans (`crew.shutdown_closed=True`) ; deregisters handlers from bus *before* force-close (no re-open race); `_is_active()` false after shutdown / no client; `_buffer_token_usage` no-op after shutdown.
- **Concurrency** (P1): 5 concurrent crews — no cross-crew span parenting; per-crew token totals isolated; `_accumulate_tokens` exact under 50×100 contention.
- **State migration** (P2): `_migrate_legacy_prefixed_span_maps` moves `rsn::id` (2 colons) to `_reasoning_spans` but leaves `rsn::method::id` (3 colons) in `_flow_method_spans`.

---

## 6. Bus-routing + end-to-end — the "does it actually work" tier

No existing test goes through the event bus; all call handlers directly. These are the strongest verification of real behavior. All `@pytest.mark.crewai`, using `client_with_mocked_transport` + a real `Trace`, and `crewai_event_bus.flush()` to drain the handler thread before asserting.

- **P0 (bus)** `test_listener_registers_all_handler_types` — after construction, `_handlers` has 60+ `(event_cls, handler)` pairs; emit a real `CrewKickoffStartedEvent` via `crewai_event_bus.emit(src, ev)` → `on_crew_kickoff_started` fired. Catches a broken `setup_listeners` the entire current suite would miss.
- **P0 (bus)** `test_bus_emits_crew_event_creates_span` — emit started+completed → `trace.spans` has `crewai.crew.<name>`, `status=ok`, `crew.id` attr.
- **P0 (bus)** `test_bus_crew_task_agent_llm_nesting` — emit the 4-event sequence → assert `parent_span_id` chain crew→task→agent→llm, all ok. Verifies parent resolution works with bus-delivered events (context propagation).
- **P1 (bus)** `test_bus_llm_tokens_reach_crew_span`, `test_bus_llm_failed_error_status`, `test_bus_tool_lifecycle`, `test_bus_routing_respects_capture_flags`.
- **P0 (e2e)** `test_e2e_minimal_crew_fake_llm` — real `Crew(1 agent, 1 task)` with a **fake LLM** (subclass `crewai.llms.base_llm.BaseLLM`, override `call()` to return a string, no network); attach listener; `crew.kickoff()`; `flush()`; assert full span tree (crew→task→agent→llm), all finished, `crew.total_tokens>=0`.
- **P1 (e2e)** `test_e2e_two_sequential_tasks_sibling_parenting` (tasks are siblings under crew, not nested; tokens accumulate), `test_e2e_task_error_propagates_to_crew` (FakeLLM raises → task & crew `status=error`).
- **P2 (e2e)** `test_e2e_shutdown_closes_dangling_spans`, `test_e2e_concurrent_crews_isolation`.

> **Fake-LLM mechanism — confirm before writing:** check `../crewAI/conftest.py` and `../crewAI/lib/crewai/tests/` for how CrewAI's own suite avoids real LLM calls (it pins `vcrpy`/cassettes; there may be a fake/stub LLM helper). Prefer subclassing `BaseLLM.call()` (network-free, deterministic) over cassettes for our unit-scope e2e. The e2e-bus analysis pass surfaced this — verify the exact API rather than guessing.

---

## 7. Dedup map (honor "don't duplicate existing tests")

New real-event tests **upgrade** (don't duplicate) these MagicMock tests — keep both, or replace the MagicMock one once the real-event version is green:

| Existing (MagicMock) | Upgraded by |
|---|---|
| `test_kickoff_failed_clears_crew_span` / `test_kickoff_completed_clears_crew_span` | §3.1 P0 error/aggregation tests |
| `TestCrewTestStartedInputs` / `TestCrewTestResult` | §3.1 full-test-lifecycle |
| `test_task_completed_removes_span` / `*_omits_*_when_capture_*_false` | §3.2 pairing & gating |
| `TestTaskEvaluationScore` | §3.2 🐛 real-event (reveals #1) |
| `TestAgentHandlers` existence checks | §3.3 identity/error/lite |
| `TestLLMHandlers` / `TestLLMTemperature` | §3.4 contract tests |
| `TestToolHandlers` / `TestToolRunAttempts` | §3.5 contract tests |
| `test_guardrail_completed_pairs_via_started_event_id` | §3.9 🐛 (reveals #2-#4) |
| `test_flow_started_creates_flow_span_entry` | §3.7 nested-in-crew |
| `TestA2AConversationAndServerTask` (3 tests) | §3.11 |

Net-new areas with **no** existing coverage to dedup against: **knowledge, mcp, reasoning**, most of **a2a**, all of **`crewai_utils.py`**, **bus-routing**, **e2e**.

---

## 8. Implementation notes

- **Reuse the harness** in `test_crewai_integration.py`: `_make_client`, `_make_listener`, `_make_span` (its `attributes` dict mirrors `set_attribute`), and `_make_rich_listener`/`_make_rich_trace` for attribute assertions. For real-event tests, swap the MagicMock event factories for real Pydantic constructors; you no longer need the manual `ev.field = None` nulling.
- **Constructing real events:** they're Pydantic `BaseEvent` subclasses — many fields are required (e.g. `LLMGuardrailCompletedEvent.success`, `A2APollingStatusEvent.state/poll_count`). Read the class in `../crewAI/.../events/types/*.py` for the required set. `BaseEvent` supplies `task_id/agent_id/event_id/started_event_id/...`.
- **Bus tests:** import `from crewai.events import crewai_event_bus`; register via the listener (constructor calls `setup_listeners`); emit with `crewai_event_bus.emit(source, event)`; **always `crewai_event_bus.flush()`** before asserting (handlers run on a worker thread).
- **Baseline-vs-fix policy:** for every 🐛, leave a comment linking to §2 and the finding number, and either `pytest.mark.xfail(reason=...)` (asserting the *correct* behavior) or assert the *current* (buggy) behavior with a `# KNOWN BUG` note. Match the repo's existing precedent (LangChain known bugs baselined, not fixed).
- **Markers:** real-event/bus/e2e → `@pytest.mark.crewai`; the e2e/bus may also warrant `@pytest.mark.integration`.

---

## 9. Suggested rollout order (highest ROI first)

1. **`crewai_utils.py` unit tests (§4)** — fast, deterministic, zero deps beyond `crewai` importable; immediately catches the `truncate_str` bug and locks token/serialization logic.
2. **§2 bug-baseline tests (🐛 across §3)** — turns silent data-loss into visible, tracked regressions; directly answers "does it actually work" (answer: several parts don't).
3. **Real-event contract tests for net-new families** — knowledge, mcp, a2a, reasoning (most missing data, most drift risk).
4. **Real-event upgrades** of crew/task/agent/llm/tool/flow/guardrail/memory.
5. **Lifecycle/LRU/concurrency gaps (§5).**
6. **Bus-routing + e2e (§6)** — the capstone proving end-to-end correctness.

---

*Generated from a source-verified cross-check of the integration handlers against the real CrewAI event classes. Findings in §2 are confirmed against the `../crewAI` checkout; confirm against your pinned CrewAI version before fixing vs. baselining.*
