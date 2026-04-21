"""
CrewAI end-to-end validation script for noveum-trace

Builds a single **multi-agent** crew (2 agents, 2 tasks, search tool, unified
memory) against a live LiteLLM-backed model, records every finished trace
from the SDK client, and **asserts** span coverage (hierarchy by name,
``task.context_tasks``, LLM message payload, tool I/O, tokens, memory ops,
``crew.available_agents`` / ``crew.available_agent_count`` on the crew span,
``agent.available_tools.*`` / ``agent.tool_names`` and ``llm.available_tools.*`` /
``llm.tools`` / ``llm.input_messages`` for the search tool (display name, common slug,
or ``Serper``); if the provider omits the verbose name on LLM spans, a matching
``crewai.tool`` span plus agent tool metadata is accepted.

**Optionally** runs a **two-agent A2A remote delegation** crew when
``NOVEUM_TRACE_A2A_AGENT_CARD_URL`` (or ``CREWAI_A2A_AGENT_CARD_URL``) is set, then a
**standalone Flow** with a stub LLM inside nested crews (cheap, deterministic) to
assert ``crewai.flow`` / ``crewai.flow.method`` spans, and a **zero-impact** micro-crew
(stub LLM) that must return identical text with the listener on vs after ``shutdown()``.

This is **not** a pytest module: run as::

    python docs/examples/crewai_e2e_test.py

Prerequisites
-------------
  pip install "noveum-trace[crewai]" sentence-transformers

  For the optional A2A step (``[2/4]``): ``pip install "crewai[a2a]"`` and a reachable
  agent card URL (see Environment).

  At least one LLM credential (same resolution order as ``crewai_integration_example``):
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, or local Ollama.

  Optional real web search (otherwise a mock tool is used):
    pip install crewai-tools            # SERPER_API_KEY
    pip install langchain-community duckduckgo-search

Environment
-----------
  NOVEUM_API_KEY   — required to *run* the script (checked in ``_init_noveum_trace()`` from
                     ``main()``; importing this module does not read it or initialise the SDK)
  NOVEUM_PROJECT   — optional (default: crewai-e2e)

  NOVEUM_TRACE_A2A_AGENT_CARD_URL or CREWAI_A2A_AGENT_CARD_URL — **optional** (full
  ``https://…`` URL to a remote A2A agent card). When set, step ``[2/4]`` runs the A2A
  delegation crew and asserts ``crewai.a2a.delegation`` spans; when unset, that step
  is skipped and the rest of the suite still runs.

  This script sets ``CREWAI_DISABLE_TELEMETRY=true`` by default (if unset). CrewAI's
  built-in OpenTelemetry hook records ``crew.memory`` as a span attribute; when
  you pass a ``Memory`` instance that value is not OTel-serializable and logs
  ``Invalid type Memory for attribute 'crew_memory'``. Disabling CrewAI telemetry
  avoids that noise; set ``CREWAI_DISABLE_TELEMETRY=false`` before running if you
  want CrewAI's anonymous telemetry on.

Exit code ``0`` means all assertions passed; ``1`` means a hard failure.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import traceback
from collections import defaultdict
from typing import Any, Callable
from urllib.parse import urlparse

# Before any import that loads ``crewai`` (including ``noveum_trace.integrations.crewai``).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from crewai import LLM, Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from crewai.llms.base_llm import BaseLLM
from crewai.memory.unified_memory import Memory
from crewai.memory.utils import sanitize_scope_name
from crewai.rag.embeddings.factory import build_embedder
from crewai.tools import BaseTool
from pydantic import BaseModel

import noveum_trace
from noveum_trace.core.trace import Trace
from noveum_trace.integrations.crewai import setup_crewai_tracing
from noveum_trace.integrations.crewai.crewai_constants import (
    ATTR_A2A_DELEGATING_AGENT,
    ATTR_A2A_RECEIVING_AGENT,
    ATTR_AGENT_TOOL_NAMES,
    ATTR_CREW_AVAILABLE_AGENT_COUNT,
    ATTR_CREW_AVAILABLE_AGENTS,
    ATTR_CREW_MEMORY,
    ATTR_CREW_STATUS,
    ATTR_STATUS_SUCCESS,
    ATTR_TOOL_NAME,
    SPAN_A2A_DELEGATION,
    SPAN_AGENT,
    SPAN_CREW,
    SPAN_FLOW,
    SPAN_FLOW_METHOD,
    SPAN_LLM,
    SPAN_MEMORY_QUERY,
    SPAN_MEMORY_RETRIEVAL,
    SPAN_MEMORY_SAVE,
    SPAN_TASK,
    SPAN_TOOL,
)

# ``noveum_trace.init()`` is deferred to ``_init_noveum_trace()`` inside ``main()`` so
# importing this module does not require ``NOVEUM_API_KEY`` or exit the process.


# ---------------------------------------------------------------------------
# 1. Tool resolution (Serper → DuckDuckGo → mock)
# ---------------------------------------------------------------------------

try:
    from crewai_tools import SerperDevTool

    _search_tool: Any = SerperDevTool()
except ImportError:
    try:
        from langchain_community.tools import (
            DuckDuckGoSearchRun,  # type: ignore[import]
        )

        _ddg_backend = DuckDuckGoSearchRun()

        class _WebSearchViaDuckDuckGo(BaseTool):
            """CrewAI ``BaseTool`` wrapping LangChain's DuckDuckGo search."""

            name: str = "web_search"
            description: str = (
                "Search the web for current information (DuckDuckGo). "
                "Pass a short query string."
            )

            def _run(self, query: str) -> str:  # type: ignore[override]
                return str(_ddg_backend.invoke(query))

        _search_tool = _WebSearchViaDuckDuckGo()
    except ImportError:

        class _MockSearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Search the web for current information."

            def _run(self, query: str) -> str:  # type: ignore[override]
                return f"[mock search for {query!r}]"

        _search_tool = _MockSearchTool()

# Resolved at import time: SerperDevTool uses a long default name, not ``web_search``.
E2E_SEARCH_TOOL_NAME = (
    str(getattr(_search_tool, "name", "") or "").strip() or "web_search"
)


def _e2e_search_tool_llm_match_tokens() -> frozenset[str]:
    """
    Substrings that may appear on ``crewai.llm`` spans.

    LiteLLM / provider tool schemas often use a slug (e.g. ``search_the_internet_with_serper``)
    rather than CrewAI's verbose ``tool.name`` string.
    """
    name = E2E_SEARCH_TOOL_NAME
    out: set[str] = set()
    for s in (name, "web_search"):
        if isinstance(s, str) and s.strip():
            out.add(s.strip())
    if name:
        slug = "_".join(
            "".join(c if c.isalnum() or c.isspace() else " " for c in name)
            .lower()
            .split()
        )
        if slug:
            out.add(slug)
    if "serper" in name.lower():
        out.update(("Serper", "serper"))
    return frozenset(t for t in out if isinstance(t, str) and len(t) >= 2)


# ---------------------------------------------------------------------------
# 2. LLM factory
# ---------------------------------------------------------------------------


def _build_default_llm() -> LLM:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLM(
            model="anthropic/claude-sonnet-4-6",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            stream=True,
        )
    if os.environ.get("OPENAI_API_KEY"):
        return LLM(model="openai/gpt-4o-mini", stream=True)
    if os.environ.get("GROQ_API_KEY"):
        return LLM(model="groq/llama-3.1-70b-versatile", stream=True)
    if os.environ.get("GEMINI_API_KEY"):
        return LLM(model="gemini/gemini-1.5-flash", stream=True)
    return LLM(model="ollama/llama3.2", stream=True)


default_llm = _build_default_llm()

E2E_CREW_NAME = "e2e-memory-multi"
E2E_A2A_CREW_NAME = "e2e-a2a-remote"
_ZERO_IMPACT_TEXT = "E2E stub: Paris is the capital of France."


def _optional_a2a_agent_card_url() -> str | None:
    """Return agent card URL if set and valid; ``None`` to skip the A2A crew."""
    url = (
        os.environ.get("NOVEUM_TRACE_A2A_AGENT_CARD_URL", "").strip()
        or os.environ.get("CREWAI_A2A_AGENT_CARD_URL", "").strip()
    )
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(
            f"A2A agent card URL must be http(s): got scheme={parsed.scheme!r}"
        )
    return url


class _FlowStubLLM(BaseLLM):
    """Deterministic LLM for nested crews inside the Flow (no external API)."""

    llm_type: str = "noveum_e2e_flow_stub"

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs,
    ):
        return (
            "1. Introduction\n2. Market trends\n3. Risks\n4. Outlook\n"
            "Summary paragraph for E2E flow (fixed output)."
        )


class _ZeroImpactStubLLM(BaseLLM):
    """Deterministic LLM for zero-impact comparison."""

    llm_type: str = "noveum_e2e_zero_impact_stub"

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        from_task=None,
        from_agent=None,
        response_model=None,
        **kwargs,
    ):
        return _ZERO_IMPACT_TEXT


# ---------------------------------------------------------------------------
# 3. Trace recorder (wraps client._export_trace — same object CrewAI populated)
# ---------------------------------------------------------------------------


def _install_trace_recorder() -> list[Trace]:
    finished: list[Trace] = []

    client = noveum_trace.get_client()
    orig_export = client._export_trace

    def _wrapped(trace: Trace) -> None:
        finished.append(trace)
        orig_export(trace)

    client._export_trace = _wrapped  # type: ignore[method-assign]
    return finished


# ---------------------------------------------------------------------------
# 4. Scenarios
# ---------------------------------------------------------------------------


def run_memory_multi_agent_crew() -> str:
    """2 agents, 2 tasks, search tool, unified memory — the combined E2E crew."""

    embedder_spec = {
        "provider": "sentence-transformer",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
        },
    }
    crew_memory = Memory(
        embedder=build_embedder(embedder_spec),
        llm=default_llm,
        root_scope=f"/crew/{sanitize_scope_name(E2E_CREW_NAME)}",
    )

    researcher = Agent(
        role="E2E Researcher",
        goal="Find concise facts on the given topic using the search tool once.",
        backstory="You research carefully and cite tool results.",
        tools=[_search_tool],
        llm=default_llm,
        verbose=False,
        allow_delegation=False,
    )
    writer = Agent(
        role="E2E Writer",
        goal="Turn research notes into a short summary.",
        backstory="You write tight executive summaries.",
        llm=default_llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Use the search tool at most once to gather 3 bullet facts on {topic}. "
            "Keep each bullet under 25 words."
        ),
        expected_output="Exactly 3 bullet lines starting with '- '.",
        agent=researcher,
    )
    write_task = Task(
        description=(
            "Using only the prior task output as context, write a 90-word "
            "executive summary of {topic} for a busy CTO."
        ),
        expected_output="One paragraph, <= 90 words.",
        agent=writer,
        context=[research_task],
    )

    crew = Crew(
        name=E2E_CREW_NAME,
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        memory=crew_memory,
        embedder=embedder_spec,
        verbose=False,
    )
    return str(crew.kickoff(inputs={"topic": "edge AI inference trends"}))


def run_a2a_remote_delegation_crew(agent_card_url: str) -> str:
    """
    Two-agent sequential crew: local briefing, then coordinator with ``A2AClientConfig``.

    Emits ``crewai.a2a.delegation`` when CrewAI performs remote A2A delegation
    (requires ``crewai[a2a]`` and a reachable agent card URL).
    """
    try:
        from crewai.a2a import A2AClientConfig  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("A2A step requires: pip install 'crewai[a2a]'") from exc

    briefing_llm = _build_default_llm()
    coordinator_llm = _build_default_llm()

    briefing_analyst = Agent(
        role="E2E Briefing Analyst",
        goal="Produce a short local draft for the A2A coordinator.",
        backstory="Two sentences from general knowledge; no tools.",
        tools=[],
        llm=briefing_llm,
        verbose=False,
        allow_delegation=False,
    )
    a2a_coordinator = Agent(
        role="E2E A2A Coordinator",
        goal="Delegate specialist work through the configured remote A2A connection.",
        backstory=textwrap.dedent(
            """\
            When the task requires A2A delegation, use your remote A2A agent
            (agent card). Return the remote agent's substantive reply.
        """
        ),
        tools=[],
        llm=coordinator_llm,
        a2a=A2AClientConfig(
            endpoint=agent_card_url,
            timeout=180,
            max_turns=8,
            fail_fast=False,
        ),
        verbose=False,
        allow_delegation=False,
    )

    draft_task = Task(
        description=(
            "Write exactly two sentences on what '{topic}' means for edge AI "
            "inference (general knowledge only)."
        ),
        expected_output="Two sentences.",
        agent=briefing_analyst,
    )
    delegate_task = Task(
        description=(
            "MANDATORY A2A: use your **remote A2A agent** to obtain **one factual sentence** "
            "about: {topic}. Prior task output is injected as context — forward useful "
            "parts to the remote agent if appropriate. Final output must reflect the "
            "remote agent's response."
        ),
        expected_output="One sentence from the remote A2A agent.",
        agent=a2a_coordinator,
        context=[draft_task],
    )
    crew = Crew(
        name=E2E_A2A_CREW_NAME,
        agents=[briefing_analyst, a2a_coordinator],
        tasks=[draft_task, delegate_task],
        process=Process.sequential,
        memory=False,
        verbose=False,
    )
    return str(crew.kickoff(inputs={"topic": "edge AI inference tracing"}))


# --- Minimal Flow (standalone trace) --------------------------------------


class _FlowState(BaseModel):
    topic: str = "E2E flow topic"


class E2EMiniFlow(Flow[_FlowState]):
    """Two-step flow so ``crewai.flow`` + ``crewai.flow.method`` spans appear."""

    @start()
    def draft_outline(self) -> dict:
        stub = _FlowStubLLM(
            model="e2e-flow-stub",
            provider="noveum",
            temperature=0.0,
        )
        agent = Agent(
            role="Flow Outline Bot",
            goal="Emit a fixed outline.",
            backstory="Stub-driven.",
            llm=stub,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=f"Produce a short outline about {self.state.topic}.",
            expected_output="Numbered outline plus one summary sentence.",
            agent=agent,
        )
        inner = Crew(agents=[agent], tasks=[task], verbose=False)
        return {"text": str(inner.kickoff())}

    @listen(draft_outline)
    def finalize(self, prev: dict) -> str:
        return (prev.get("text") or "").strip()


def run_standalone_flow() -> str:
    flow = E2EMiniFlow()
    return str(flow.kickoff(inputs={"topic": "supply chain resilience"}))


def _run_zero_impact_micro_crew() -> str:
    stub = _ZeroImpactStubLLM(
        model="e2e-zero-impact-stub",
        provider="noveum",
        temperature=0.0,
    )
    agent = Agent(
        role="E2E Micro",
        goal="Answer in one sentence.",
        backstory="Stub.",
        llm=stub,
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description="What is the capital of France? One sentence.",
        expected_output="One sentence.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return str(crew.kickoff()).strip()


def run_zero_impact(listener: Any) -> None:
    with_listener = _run_zero_impact_micro_crew()
    assert with_listener == _ZERO_IMPACT_TEXT.strip()
    listener.shutdown()
    without = _run_zero_impact_micro_crew()
    assert without == _ZERO_IMPACT_TEXT.strip(), (
        "Zero-impact failed: output differed with listener removed "
        f"(with={with_listener!r}, without={without!r})"
    )


# ---------------------------------------------------------------------------
# 5. Assertions on recorded traces
# ---------------------------------------------------------------------------


def _count_spans_by_name(trace: Trace) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for span in trace.spans:
        counts[span.name] += 1
    return dict(counts)


def _find_crew_root_span(trace: Trace) -> Any:
    for span in trace.spans:
        if span.name == SPAN_CREW and not span.parent_span_id:
            return span
    for span in trace.spans:
        if span.name == SPAN_CREW:
            return span
    return None


def _names_attr_contains_tool_name(raw: Any, tool_name: str) -> bool:
    if not tool_name:
        return False
    if isinstance(raw, list):
        return tool_name in raw or tool_name in [str(x) for x in raw]
    if isinstance(raw, str):
        return tool_name in raw
    return False


def _agent_tool_names_json_includes(attrs: dict[str, Any], tool_name: str) -> bool:
    raw = attrs.get(ATTR_AGENT_TOOL_NAMES)
    if not isinstance(raw, str) or not tool_name:
        return False
    try:
        names = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(names, list):
        return False
    return tool_name in [str(x) for x in names]


def _agent_span_shows_search_tool(attrs: dict[str, Any], tool_name: str) -> bool:
    if _names_attr_contains_tool_name(
        attrs.get("agent.available_tools.names"), tool_name
    ):
        return True
    schemas = attrs.get("agent.available_tools.schemas")
    if isinstance(schemas, str) and tool_name in schemas:
        return True
    return _agent_tool_names_json_includes(attrs, tool_name)


def _haystack_matches_any_token(haystack: str, tokens: frozenset[str]) -> bool:
    if not haystack:
        return False
    return any(t in haystack for t in tokens)


def _llm_span_matches_search_tool_tokens(
    attrs: dict[str, Any], tokens: frozenset[str]
) -> bool:
    """True if LLM span attributes mention the search tool (names, schemas, tools, messages)."""
    parts: list[str] = []
    n = attrs.get("llm.available_tools.names")
    if isinstance(n, list):
        parts.extend(str(x) for x in n)
    elif isinstance(n, str):
        parts.append(n)
    for key in ("llm.available_tools.schemas", "llm.tools", "llm.input_messages"):
        v = attrs.get(key)
        if isinstance(v, str):
            parts.append(v)
    return _haystack_matches_any_token("\n".join(parts), tokens)


def _tool_span_matches_search_tool(
    attrs: dict[str, Any], tokens: frozenset[str]
) -> bool:
    """True if ``tool.name`` matches the configured search tool (display name or slug token)."""
    raw = attrs.get(ATTR_TOOL_NAME)
    if not isinstance(raw, str):
        return False
    name = raw.strip()
    if not name:
        return False
    if name == E2E_SEARCH_TOOL_NAME:
        return True
    return _haystack_matches_any_token(name, tokens)


def _assert_crew_available_agents(
    crew_attrs: dict[str, Any], expected_role_labels: frozenset[str]
) -> None:
    """Assert ``crew.available_agents`` JSON and ``crew.available_agent_count``."""
    raw = crew_attrs.get(ATTR_CREW_AVAILABLE_AGENTS)
    if not raw:
        raise AssertionError(
            f"Expected {ATTR_CREW_AVAILABLE_AGENTS!r} on crew span; got {raw!r}"
        )
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"crew.available_agents is not valid JSON: {raw!r}"
        ) from exc
    if not isinstance(parsed, list):
        raise AssertionError(
            f"crew.available_agents must be a JSON list, got {type(parsed).__name__}"
        )
    found = {str(x) for x in parsed}
    missing = expected_role_labels - found
    if missing:
        raise AssertionError(
            f"crew.available_agents missing labels {sorted(missing)!r}; "
            f"have {sorted(found)!r}"
        )
    count = crew_attrs.get(ATTR_CREW_AVAILABLE_AGENT_COUNT)
    if count != len(expected_role_labels):
        raise AssertionError(
            f"Expected {ATTR_CREW_AVAILABLE_AGENT_COUNT}={len(expected_role_labels)}, "
            f"got {count!r}"
        )


def assert_memory_crew_trace(trace: Trace) -> None:
    if not trace.name.startswith(f"crewai.{E2E_CREW_NAME}"):
        raise AssertionError(
            f"Expected trace name crewai.{E2E_CREW_NAME}.*, got {trace.name!r}"
        )

    counts = _count_spans_by_name(trace)
    for need in (SPAN_CREW, SPAN_TASK, SPAN_AGENT, SPAN_LLM):
        if counts.get(need, 0) < 1:
            raise AssertionError(f"Missing span {need!r} in {trace.trace_id}: {counts}")

    if counts.get(SPAN_TOOL, 0) < 1:
        raise AssertionError(
            f"Expected at least one {SPAN_TOOL} span (tool trace); got {counts}"
        )

    mem_ops = (
        counts.get(SPAN_MEMORY_QUERY, 0)
        + counts.get(SPAN_MEMORY_SAVE, 0)
        + counts.get(SPAN_MEMORY_RETRIEVAL, 0)
    )
    if mem_ops < 1:
        raise AssertionError(
            "Expected at least one crewai.memory.* span when memory is enabled; "
            f"got {counts}"
        )

    crew_span = _find_crew_root_span(trace)
    if crew_span is None:
        raise AssertionError("No crewai.crew span found")
    attrs = crew_span.attributes
    if attrs.get(ATTR_CREW_STATUS) != ATTR_STATUS_SUCCESS:
        raise AssertionError(
            f"crew.status expected {ATTR_STATUS_SUCCESS!r}, got {attrs.get(ATTR_CREW_STATUS)!r}"
        )
    if not attrs.get(ATTR_CREW_MEMORY):
        raise AssertionError("crew.memory should be truthy on memory-enabled crew")

    _assert_crew_available_agents(
        attrs,
        frozenset({"E2E Researcher", "E2E Writer"}),
    )

    # task.context_tasks on the writer task (second task)
    ctx_found = False
    token_found = False
    cost_found = False
    sys_prompt_found = False
    msgs_found = False
    tool_io = False
    agent_search_tool_ok = False
    llm_search_tool_ok = False
    tool_span_search_ok = False
    tool_name = E2E_SEARCH_TOOL_NAME
    llm_tool_tokens = _e2e_search_tool_llm_match_tokens()

    for span in trace.spans:
        a = span.attributes
        if span.name == SPAN_AGENT and _agent_span_shows_search_tool(a, tool_name):
            agent_search_tool_ok = True
        if span.name == SPAN_LLM and _llm_span_matches_search_tool_tokens(
            a, llm_tool_tokens
        ):
            llm_search_tool_ok = True
        if span.name == SPAN_TOOL and _tool_span_matches_search_tool(
            a, llm_tool_tokens
        ):
            tool_span_search_ok = True
        if span.name == SPAN_TASK:
            raw = a.get("task.context_tasks")
            if raw:
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    parsed = []
                if isinstance(parsed, list) and len(parsed) >= 1:
                    ctx_found = True
        if span.name == SPAN_LLM:
            if int(a.get("llm.total_tokens") or a.get("llm.input_tokens") or 0) > 0:
                token_found = True
            # Cost may be zero on some providers; accept either total cost or currency line
            if a.get("llm.cost.total") is not None or a.get("llm.cost.currency"):
                cost_found = True
            sp = a.get("llm.system_prompt")
            if isinstance(sp, str) and sp.strip():
                sys_prompt_found = True
            im = a.get("llm.input_messages")
            if isinstance(im, str) and len(im) > 50:
                msgs_found = True
        if span.name == SPAN_TOOL:
            if a.get("tool.input") and (a.get("tool.output") is not None):
                tool_io = True

    if not ctx_found:
        raise AssertionError(
            "Expected task.context_tasks JSON with >=1 upstream description on a task span"
        )
    if not token_found:
        raise AssertionError(
            "Expected llm.total_tokens (or input_tokens) > 0 on at least one crewai.llm span"
        )
    if not cost_found:
        raise AssertionError("Expected llm.cost.* attributes on at least one LLM span")
    if not sys_prompt_found:
        raise AssertionError("Expected non-empty llm.system_prompt on an LLM span")
    if not msgs_found:
        raise AssertionError("Expected llm.input_messages JSON on an LLM span")
    if not tool_io:
        raise AssertionError("Expected tool.input and tool.output on a tool span")
    if not agent_search_tool_ok:
        raise AssertionError(
            "Expected agent.available_tools / agent.tool_names to include the search "
            f"tool name {tool_name!r} on a crewai.agent span (researcher carries the tool)"
        )
    if not llm_search_tool_ok and agent_search_tool_ok and tool_span_search_ok:
        # Provider payloads often slug tool names; ``tool.name`` on ``crewai.tool`` still
        # reflects the CrewAI tool identity when LLM JSON does not repeat the long name.
        llm_search_tool_ok = True
    if not llm_search_tool_ok:
        raise AssertionError(
            "Expected an LLM span to mention the search tool (names/schemas/tools/messages "
            f"matching any of {sorted(llm_tool_tokens)!r}), or agent+tool spans to show the "
            f"tool (display name {tool_name!r}) when the LLM payload uses a different label"
        )


def assert_a2a_e2e_trace(trace: Trace) -> None:
    if not trace.name.startswith(f"crewai.{E2E_A2A_CREW_NAME}"):
        raise AssertionError(
            f"Expected trace name crewai.{E2E_A2A_CREW_NAME}.*, got {trace.name!r}"
        )
    counts = _count_spans_by_name(trace)
    for need in (SPAN_CREW, SPAN_TASK, SPAN_AGENT, SPAN_LLM):
        if counts.get(need, 0) < 1:
            raise AssertionError(f"Missing span {need!r} in A2A trace: {counts}")
    if counts.get(SPAN_A2A_DELEGATION, 0) < 1:
        raise AssertionError(
            f"Expected at least one {SPAN_A2A_DELEGATION} span; got {counts}"
        )

    a2a_crew = _find_crew_root_span(trace)
    if a2a_crew is None:
        raise AssertionError("No crewai.crew span found for A2A trace")
    _assert_crew_available_agents(
        a2a_crew.attributes,
        frozenset({"E2E Briefing Analyst", "E2E A2A Coordinator"}),
    )

    deleg_ok = False
    for span in trace.spans:
        if span.name != SPAN_A2A_DELEGATION:
            continue
        a = span.attributes
        da = a.get(ATTR_A2A_DELEGATING_AGENT)
        ra = a.get(ATTR_A2A_RECEIVING_AGENT)
        if da and ra and str(da).strip() and str(ra).strip():
            deleg_ok = True
            break
    if not deleg_ok:
        raise AssertionError(
            "Expected a2a.delegating_agent and a2a.receiving_agent on a delegation span"
        )


def assert_flow_trace(trace: Trace) -> None:
    if not trace.name.startswith("crewai.flow."):
        raise AssertionError(f"Expected flow trace name prefix, got {trace.name!r}")
    counts = _count_spans_by_name(trace)
    if counts.get(SPAN_FLOW, 0) < 1:
        raise AssertionError(f"Missing {SPAN_FLOW}: {counts}")
    if counts.get(SPAN_FLOW_METHOD, 0) < 1:
        raise AssertionError(f"Missing {SPAN_FLOW_METHOD}: {counts}")


def _pick_trace(traces: list[Trace], predicate: Callable[[Trace], bool]) -> Trace:
    for t in reversed(traces):
        if predicate(t):
            return t
    raise AssertionError("No trace matched predicate in recorded exports")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------


def _init_noveum_trace() -> None:
    """Require ``NOVEUM_API_KEY`` and initialise the Noveum SDK (call from ``main()`` only)."""
    if not os.environ.get("NOVEUM_API_KEY"):
        print("NOVEUM_API_KEY is required.", file=sys.stderr)
        raise SystemExit(1)
    noveum_trace.init(
        api_key=os.environ["NOVEUM_API_KEY"],
        project=os.environ.get("NOVEUM_PROJECT", "crewai-e2e"),
    )


def main() -> int:
    _init_noveum_trace()

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print(
            "sentence-transformers is required for memory embedder "
            "(pip install sentence-transformers).",
            file=sys.stderr,
        )
        return 1

    recorded = _install_trace_recorder()
    listener = setup_crewai_tracing(
        capture_inputs=True,
        capture_outputs=True,
        capture_llm_messages=True,
        capture_tool_schemas=True,
        capture_agent_snapshot=True,
        capture_crew_snapshot=True,
        capture_memory=True,
        capture_flow=True,
        capture_a2a=True,
        verbose=False,
    )

    print("[1/4] Multi-agent memory crew …")
    try:
        out = run_memory_multi_agent_crew()
        print(f"      kickoff ok, output chars={len(out)}")
    except Exception as exc:
        print(
            f"[FAIL] Memory crew raised: {exc}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        listener.shutdown()
        return 1

    try:
        mem_trace = _pick_trace(
            recorded, lambda t: t.name.startswith(f"crewai.{E2E_CREW_NAME}")
        )
        assert_memory_crew_trace(mem_trace)
        print(f"      trace {mem_trace.trace_id!r} passed structural assertions")
    except Exception as exc:
        print(f"[FAIL] Memory trace assertions: {exc}", file=sys.stderr)
        listener.shutdown()
        return 1

    noveum_trace.flush()

    try:
        a2a_url = _optional_a2a_agent_card_url()
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        listener.shutdown()
        return 1

    if a2a_url:
        host = urlparse(a2a_url).netloc or a2a_url
        print(f"[2/4] A2A remote delegation crew (agent card host={host!r}) …")
        try:
            a2a_out = run_a2a_remote_delegation_crew(a2a_url)
            print(f"      A2A kickoff ok, output chars={len(a2a_out)}")
        except Exception as exc:
            print(
                f"[FAIL] A2A crew raised: {exc}\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            listener.shutdown()
            return 1
        noveum_trace.flush()
        try:
            a2a_trace = _pick_trace(
                recorded, lambda t: t.name.startswith(f"crewai.{E2E_A2A_CREW_NAME}")
            )
            assert_a2a_e2e_trace(a2a_trace)
            print(f"      trace {a2a_trace.trace_id!r} passed A2A assertions")
        except Exception as exc:
            print(f"[FAIL] A2A trace assertions: {exc}", file=sys.stderr)
            listener.shutdown()
            return 1
    else:
        print(
            "[2/4] A2A remote delegation: skipped (set NOVEUM_TRACE_A2A_AGENT_CARD_URL or "
            "CREWAI_A2A_AGENT_CARD_URL to run ``crewai[a2a]`` delegation + assertions). "
            "See https://docs.crewai.com/en/learn/a2a-agent-delegation"
        )

    print("[3/4] Standalone Flow (stub LLM inside nested crew) …")
    try:
        flow_out = run_standalone_flow()
        print(f"      flow ok, output chars={len(flow_out)}")
    except Exception as exc:
        print(f"[FAIL] Flow raised: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        listener.shutdown()
        return 1

    try:
        flow_trace = _pick_trace(recorded, lambda t: t.name.startswith("crewai.flow."))
        assert_flow_trace(flow_trace)
        print(f"      trace {flow_trace.trace_id!r} passed flow assertions")
    except Exception as exc:
        print(f"[FAIL] Flow trace assertions: {exc}", file=sys.stderr)
        listener.shutdown()
        return 1

    print("[4/4] Zero-impact stub micro-crew (listener shutdown inside) …")
    try:
        run_zero_impact(listener)
        print("      zero-impact OK")
    except Exception as exc:
        print(f"[FAIL] Zero-impact: {exc}\n{traceback.format_exc()}", file=sys.stderr)
        return 1

    noveum_trace.flush()
    print("\nAll CrewAI E2E checks passed. Traces were exported to Noveum.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
