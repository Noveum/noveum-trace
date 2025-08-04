"""
Integration tests for real LLM scenarios with actual API calls.

This module tests real-world LLM scenarios including chat completions,
streaming responses, function calling, multi-turn conversations, and
complex agent workflows with actual LLM provider APIs.

NO MOCKING - All tests use real API calls and validate traces via GET requests.
"""

import os
import time
from typing import Any, Optional

import pytest
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

import noveum_trace

# Test endpoints - configurable via environment
ENDPOINT = os.environ.get("NOVEUM_ENDPOINT", "https://api.noveum.ai/api")
API_KEY = os.environ.get("NOVEUM_API_KEY", "test-api-key")

# LLM Provider API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Import real clients>
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False


def is_valid_api_key(key: Optional[str]) -> bool:
    """Check if API key is valid (not None, empty, or placeholder)."""
    if not key:
        return False

    invalid_keys = [
        "",
        "your-openai-api-key-here",
        "your-anthropic-api-key-here",
        "test-key",
        "sk-test",
        "sk-fake",
    ]
    return key not in invalid_keys and len(key) > 10


def should_test_openai() -> bool:
    """Check if OpenAI tests should run."""
    return OPENAI_AVAILABLE and is_valid_api_key(OPENAI_API_KEY)


def should_test_anthropic() -> bool:
    """Check if Anthropic tests should run."""
    return ANTHROPIC_AVAILABLE and is_valid_api_key(ANTHROPIC_API_KEY)


def get_openai_client() -> openai.OpenAI:
    """Get real OpenAI client."""
    if not should_test_openai():
        pytest.skip("OpenAI API key not available or invalid")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def get_anthropic_client() -> anthropic.Anthropic:
    """Get real Anthropic client."""
    if not should_test_anthropic():
        pytest.skip("Anthropic API key not available or invalid")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def validate_trace_via_api(trace_id: str, timeout: int = 120) -> dict[str, Any]:
    """
    Validate that a trace was successfully sent by fetching it via GET API.

    Args:
        trace_id: The trace ID to fetch
        timeout: Maximum time to wait for trace to appear (seconds)

    Returns:
        The trace data from the API

    Raises:
        AssertionError: If trace is not found or validation fails
    """
    # Calculate retry sleep interval proportional to timeout (min 0.5s, max 2.0s)
    # For default timeout=120s: RETRY_SLEEP = 6.0s -> capped at 2.0s
    # Unified across all retry paths to ensure consistent back-off behavior
    RETRY_SLEEP = max(0.5, min(2.0, timeout / 20))

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    url = f"{ENDPOINT.rstrip('/')}/v1/traces/{trace_id}"

    # Retry for up to timeout seconds
    start_time = time.time()
    retry_count = 0
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                trace_data = response.json()

                # Check if we have actual trace data
                if "trace" in trace_data:
                    trace = trace_data["trace"]
                    if trace:  # Ensure trace is not None
                        assert (
                            trace["trace_id"] == trace_id
                        ), f"Trace ID mismatch: expected {trace_id}, got {trace['trace_id']}"

                        print(
                            f"✅ Successfully validated trace {trace_id} after {retry_count} retries"
                        )
                        return trace_data

                # Handle new API response format with success/data wrapper
                elif (
                    "success" in trace_data
                    and trace_data["success"]
                    and "data" in trace_data
                ):
                    trace = trace_data["data"]
                    if trace:  # Ensure trace is not None
                        assert (
                            trace["trace_id"] == trace_id
                        ), f"Trace ID mismatch: expected {trace_id}, got {trace['trace_id']}"

                        print(
                            f"✅ Successfully validated trace {trace_id} after {retry_count} retries"
                        )
                        # Normalize the response format to have "trace" key
                        return {"trace": trace}

                # If we get status OK but no trace data, the trace might still be processing
                elif "status" in trace_data and trace_data["status"] == "ok":
                    print(
                        f"⏳ Trace {trace_id} acknowledged but data not yet available (attempt {retry_count})"
                    )
                    retry_count += 1
                    time.sleep(RETRY_SLEEP)
                    continue

                # If response format is unexpected, log and retry
                else:
                    print(f"⚠️  Unexpected response format: {trace_data}")
                    retry_count += 1
                    time.sleep(RETRY_SLEEP)
                    continue

            elif response.status_code == 404:
                # Trace not found yet, wait and retry
                retry_count += 1
                if retry_count % 5 == 0:  # Log every 5 retries
                    print(f"⏳ Waiting for trace {trace_id}... (attempt {retry_count})")
                time.sleep(RETRY_SLEEP)
                continue
            elif response.status_code == 401:
                pytest.fail(
                    f"Authentication failed - check API key. Status: {response.status_code}"
                )
            elif response.status_code == 403:
                pytest.fail(
                    f"Access forbidden - check API key permissions. Status: {response.status_code}"
                )
            else:
                pytest.fail(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

        except requests.RequestException as e:
            if "timeout" in str(e).lower():
                print(f"⚠️  Request timeout, retrying... ({retry_count})")
                time.sleep(RETRY_SLEEP)
                continue
            pytest.fail(f"API request failed: {e}")

    pytest.fail(
        f"❌ Trace {trace_id} not found within {timeout} seconds after {retry_count} retries"
    )


# Test models configuration - easily extensible for future models
LLM_MODELS = [
    pytest.param(
        "openai",
        "gpt-4o-mini",
        get_openai_client,
        marks=pytest.mark.skipif(
            not should_test_openai(), reason="OpenAI API key not available"
        ),
        id="openai-gpt4o-mini",
    ),
    pytest.param(
        "openai",
        "gpt-3.5-turbo",
        get_openai_client,
        marks=pytest.mark.skipif(
            not should_test_openai(), reason="OpenAI API key not available"
        ),
        id="openai-gpt35-turbo",
    ),
    pytest.param(
        "anthropic",
        "claude-3-haiku-20240307",
        get_anthropic_client,
        marks=pytest.mark.skipif(
            not should_test_anthropic(), reason="Anthropic API key not available"
        ),
        id="anthropic-claude3-haiku",
    ),
]


@pytest.fixture(autouse=True)
def setup_noveum_trace():
    """Setup noveum trace for each test."""
    # Ensure clean state
    noveum_trace.shutdown()

    # Initialize with optimized transport settings for integration tests
    noveum_trace.init(
        project="noveum-trace-python",
        api_key=API_KEY,
        endpoint=ENDPOINT,
        environment="git-integ-test",
        transport_config={
            "batch_size": 1,  # Send traces immediately
            "batch_timeout": 0.1,  # Very short timeout for faster tests
            "timeout": 10,  # Reduced timeout for test environment
        },
    )

    yield

    # Cleanup - shutdown completely to ensure clean state for next test
    noveum_trace.shutdown()
    # Removed sleep - shutdown should handle cleanup properly


@pytest.mark.integration
@pytest.mark.disable_transport_mocking
class TestRealLLMBasicScenarios:
    """Test basic LLM scenarios with real API calls."""

    @pytest.mark.parametrize("provider,model_name,client_factory", LLM_MODELS)
    def test_simple_chat_completion(
        self, provider: str, model_name: str, client_factory
    ):
        """Test simple chat completion with real LLM providers."""
        client = client_factory()

        @noveum_trace.trace_llm(provider=provider, metadata={"model": model_name})
        def call_llm(prompt: str) -> str:
            current_span = noveum_trace.get_current_span()

            if provider == "openai":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                )
                result = response.choices[0].message.content

                # Add detailed attributes
                if current_span:
                    current_span.set_attribute(
                        "llm.usage.input_tokens", response.usage.prompt_tokens
                    )
                    current_span.set_attribute(
                        "llm.usage.output_tokens", response.usage.completion_tokens
                    )
                    current_span.set_attribute(
                        "llm.usage.total_tokens", response.usage.total_tokens
                    )

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                )
                result = response.content[0].text

                # Add detailed attributes
                if current_span:
                    current_span.set_attribute(
                        "llm.usage.input_tokens", response.usage.input_tokens
                    )
                    current_span.set_attribute(
                        "llm.usage.output_tokens", response.usage.output_tokens
                    )
                    current_span.set_attribute(
                        "llm.usage.total_tokens",
                        response.usage.input_tokens + response.usage.output_tokens,
                    )

            if current_span:
                current_span.set_attribute("llm.response", result)
                current_span.set_attribute("llm.model", model_name)
                current_span.set_attribute("llm.provider", provider)

            return result

        # Execute the test and capture the trace ID
        prompt = "What is the capital of France? Answer in one sentence."

        # Capture trace ID before calling the function
        trace_id = None

        # Create a manual trace to ensure we have a trace context
        with noveum_trace.start_trace(
            f"test_llm_{provider}_{model_name}"
        ) as test_trace:
            trace_id = test_trace.trace_id
            result = call_llm(prompt)

            # Validate response
            assert result is not None, "LLM response should not be None"
            assert len(result.strip()) > 0, "LLM response should not be empty"
            assert (
                "paris" in result.lower()
            ), f"Expected Paris in response, got: {result}"

        # Flush and validate trace
        noveum_trace.flush()

        # Give backend time to process the trace
        print(f"⏱️  Waiting 5 seconds for backend to process {trace_id}...")
        time.sleep(5)

        # Validate trace using the captured ID
        assert trace_id is not None, "Trace ID should have been captured"
        trace_data = validate_trace_via_api(trace_id)

        # Validate trace structure
        trace = trace_data["trace"]

        assert (
            trace["status"] == "ok"
        ), f"Trace status should be 'ok', got: {trace['status']}"
        assert (
            trace["span_count"] >= 1
        ), f"Should have at least 1 span, got: {trace['span_count']}"

        # Validate span attributes
        spans = trace["spans"]
        llm_span = None
        for span in spans:
            if span["attributes"].get("llm.provider") == provider:
                llm_span = span
                break

        assert (
            llm_span is not None
        ), f"Should have LLM span with provider {provider}. Available spans: {[s.get('name', 'unnamed') for s in spans]}"
        assert llm_span["attributes"]["llm.model"] == model_name
        assert llm_span["attributes"]["llm.provider"] == provider
        assert "llm.usage.total_tokens" in llm_span["attributes"]


@pytest.mark.integration
@pytest.mark.disable_transport_mocking
class TestRealLLMFunctionCalling:
    """Test function calling scenarios with real LLM providers."""

    @pytest.mark.skipif(not should_test_openai(), reason="OpenAI API key not available")
    def test_openai_function_calling(self):
        """Test OpenAI function calling with real API."""
        client = get_openai_client()

        @noveum_trace.trace_tool(tool_name="get_weather", tool_type="api_tool")
        def get_weather(location: str) -> dict[str, Any]:
            """Get weather information for a location."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("tool.location", location)
                current_span.set_attribute("tool.api_call", "weather_service")

            # Simulate weather API response
            weather_data = {
                "location": location,
                "temperature": 22,
                "condition": "sunny",
                "humidity": 65,
                "timestamp": "2024-01-15T12:00:00Z",
            }

            if current_span:
                import json

                current_span.set_attribute("tool.result", json.dumps(weather_data))
                current_span.set_attribute("tool.success", True)

            return weather_data

        @noveum_trace.trace_llm(
            provider="openai",
            metadata={"model": "gpt-4o-mini", "function_calling": True},
        )
        def llm_with_function_calling(user_message: str) -> str:
            """LLM call that can use function calling."""
            current_span = noveum_trace.get_current_span()

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather information for a specific location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]

            messages = [{"role": "user", "content": user_message}]

            # First LLM call
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
            )

            message = response.choices[0].message

            if current_span:
                current_span.set_attribute(
                    "llm.has_tool_calls", bool(message.tool_calls)
                )
                if message.tool_calls:
                    current_span.set_attribute(
                        "llm.tool_calls_count", len(message.tool_calls)
                    )

            # Handle tool calls
            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    if tool_call.function.name == "get_weather":
                        import json

                        function_args = json.loads(tool_call.function.arguments)
                        function_response = get_weather(function_args["location"])

                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "content": json.dumps(function_response),
                            }
                        )

                # Second LLM call with function results
                second_response = client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages
                )

                if current_span:
                    current_span.set_attribute(
                        "llm.total_tokens",
                        response.usage.total_tokens
                        + second_response.usage.total_tokens,
                    )

                return second_response.choices[0].message.content

            return message.content

        # Execute function calling test
        user_query = "What's the weather like in San Francisco?"

        # Create a manual trace to ensure we have a trace context
        trace_id = None
        with noveum_trace.start_trace("test_openai_function_calling") as test_trace:
            trace_id = test_trace.trace_id
            result = llm_with_function_calling(user_query)

            # Validate response includes weather information
            assert result is not None, "Function calling response should not be None"
            assert (
                len(result.strip()) > 0
            ), "Function calling response should not be empty"
            assert any(
                word in result.lower()
                for word in ["weather", "temperature", "sunny", "francisco"]
            ), f"Response should contain weather information: {result}"

        # Flush and validate traces
        noveum_trace.flush()

        # Give backend time to process function calling traces with multiple spans
        print(f"⏱️  Waiting 8 seconds for backend to process {trace_id}...")
        time.sleep(8)

        # Validate trace using the captured ID
        assert trace_id is not None, "Trace ID should have been captured"
        trace_data = validate_trace_via_api(trace_id)

        trace = trace_data["trace"]
        assert trace["status"] == "ok"

        # Should have both LLM and tool spans
        spans = trace["spans"]

        # If spans array is empty but span_count > 0, this is a backend issue - FAIL the test with details
        if trace["span_count"] > 0 and len(spans) == 0:
            print("\n🚨 BACKEND API FAILURE DETECTED:")
            print("    Model: openai gpt-4o-mini (function calling)")
            print(f"    Trace ID: {trace_id}")
            print(f"    Expected spans: {trace['span_count']}")
            print(f"    Actual spans returned: {len(spans)}")
            print("    Full backend response:")
            import json

            print(json.dumps(trace_data, indent=2, default=str))

            pytest.fail(
                f"BACKEND API BUG: span_count={trace['span_count']} but spans array is empty! "
                f"Backend failed to return stored spans for trace {trace_id}"
            )
        llm_spans = [
            s for s in spans if s["attributes"].get("llm.provider") == "openai"
        ]
        tool_spans = [
            s for s in spans if s["attributes"].get("tool.name") == "get_weather"
        ]

        assert len(llm_spans) >= 1, "Should have at least one LLM span"
        assert len(tool_spans) >= 1, "Should have at least one tool span"

        # Validate tool span
        tool_span = tool_spans[0]
        assert tool_span["attributes"]["tool.name"] == "get_weather"
        assert "San Francisco" in tool_span["attributes"]["tool.location"]


@pytest.mark.integration
@pytest.mark.disable_transport_mocking
class TestRealAgentScenarios:
    """Test real agent workflow scenarios."""

    @pytest.mark.parametrize("provider,model_name,client_factory", LLM_MODELS)
    def test_multi_agent_research_workflow(
        self, provider: str, model_name: str, client_factory
    ):
        """Test multi-agent research workflow with real LLM calls."""
        client = client_factory()

        @noveum_trace.trace_agent(
            agent_id="research_coordinator", agent_type="coordinator"
        )
        def research_coordinator(research_topic: str) -> dict[str, Any]:
            """Coordinate research workflow across multiple agents."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("research.topic", research_topic)
                current_span.set_attribute("agent.workflow", "multi_agent_research")

            # Execute research workflow
            search_results = search_agent(research_topic)
            analysis = analysis_agent(search_results, research_topic)
            summary = summary_agent(analysis, research_topic)

            workflow_result = {
                "topic": research_topic,
                "search_results": search_results,
                "analysis": analysis,
                "summary": summary,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "coordinator_agent": "research_coordinator",
            }

            if current_span:
                current_span.set_attribute("agent.completed_steps", 3)
                import json

                current_span.set_attribute(
                    "research.result_size", len(json.dumps(workflow_result))
                )

            return workflow_result

        @noveum_trace.trace_agent(agent_id="search_agent", agent_type="researcher")
        def search_agent(topic: str) -> list[dict[str, Any]]:
            """Simulate research search agent."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("agent.search_topic", topic)

            # Use search tools
            web_results = web_search_tool(topic)
            academic_results = academic_search_tool(topic)

            combined_results = web_results + academic_results

            if current_span:
                current_span.set_attribute("agent.results_count", len(combined_results))
                current_span.set_attribute("agent.sources", "web,academic")

            return combined_results

        @noveum_trace.trace_tool(tool_name="web_search", tool_type="search_engine")
        def web_search_tool(query: str) -> list[dict[str, Any]]:
            """Simulate web search tool."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("tool.query", query)
                current_span.set_attribute("tool.search_engine", "simulated_google")

            # Simulate realistic search results
            results = [
                {
                    "title": f"Understanding {query}: A Comprehensive Guide",
                    "url": f"https://example.com/{query.replace(' ', '-').lower()}",
                    "snippet": f"Learn about {query} with detailed explanations and examples...",
                    "relevance_score": 0.92,
                    "source_type": "web",
                },
                {
                    "title": f"{query} Best Practices and Applications",
                    "url": f"https://tech-blog.com/{query.replace(' ', '-').lower()}-guide",
                    "snippet": f"Discover the best practices and real-world applications of {query}...",
                    "relevance_score": 0.87,
                    "source_type": "web",
                },
            ]

            if current_span:
                current_span.set_attribute("tool.results_count", len(results))
                current_span.set_attribute(
                    "tool.avg_relevance",
                    sum(r["relevance_score"] for r in results) / len(results),
                )

            return results

        @noveum_trace.trace_tool(
            tool_name="academic_search", tool_type="academic_database"
        )
        def academic_search_tool(query: str) -> list[dict[str, Any]]:
            """Simulate academic search tool."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("tool.query", query)
                current_span.set_attribute("tool.database", "simulated_arxiv")

            # Simulate academic search results
            results = [
                {
                    "title": f"Recent Advances in {query}: A Comprehensive Survey",
                    "authors": ["Dr. Jane Smith", "Dr. John Doe", "Dr. Alice Johnson"],
                    "abstract": f"This paper provides a comprehensive survey of recent advances in {query}, covering theoretical foundations and practical applications...",
                    "arxiv_id": f"2024.{hash(query) % 10000:04d}",
                    "citations": 42,
                    "source_type": "academic",
                }
            ]

            if current_span:
                current_span.set_attribute("tool.papers_found", len(results))
                current_span.set_attribute(
                    "tool.total_citations", sum(r["citations"] for r in results)
                )

            return results

        @noveum_trace.trace_agent(agent_id="analysis_agent", agent_type="analyzer")
        def analysis_agent(
            search_results: list[dict[str, Any]], topic: str
        ) -> dict[str, Any]:
            """Agent that analyzes search results using LLM."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("agent.input_sources", len(search_results))
                current_span.set_attribute("analysis.topic", topic)

            # Create analysis prompt from search results
            results_summary = "\n".join(
                [
                    f"- {result['title']}: {result.get('snippet', result.get('abstract', 'No description'))}"
                    for result in search_results[:3]  # Limit to top 3 results
                ]
            )

            analysis_prompt = f"""Analyze the following research results about '{topic}':

{results_summary}

Provide a structured analysis including:
1. Key findings (2-3 bullet points)
2. Main themes (2-3 themes)
3. Confidence assessment (0.0-1.0)

Keep the response concise and factual."""

            # Use LLM for analysis
            analysis_result = llm_analysis_call(analysis_prompt, topic)

            # Structure the analysis result
            analysis_data = {
                "topic": topic,
                "input_sources": len(search_results),
                "llm_analysis": analysis_result,
                "confidence_score": 0.85,  # Could be extracted from LLM response
                "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            if current_span:
                current_span.set_attribute(
                    "agent.analysis_confidence", analysis_data["confidence_score"]
                )
                current_span.set_attribute(
                    "agent.analysis_length", len(analysis_result)
                )

            return analysis_data

        @noveum_trace.trace_llm(
            provider=provider, metadata={"model": model_name, "task": "analysis"}
        )
        def llm_analysis_call(prompt: str, topic: str) -> str:
            """LLM call for analysis task."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("llm.task", "research_analysis")
                current_span.set_attribute("llm.topic", topic)
                current_span.set_attribute("llm.prompt_length", len(prompt))

            if provider == "openai":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert research analyst. Provide clear, structured analysis.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=100,
                    temperature=0.3,
                )
                result = response.choices[0].message.content

                if current_span:
                    current_span.set_attribute(
                        "llm.usage.total_tokens", response.usage.total_tokens
                    )

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": f"As an expert research analyst, {prompt}",
                        }
                    ],
                    max_tokens=100,
                )
                result = response.content[0].text

                if current_span:
                    current_span.set_attribute(
                        "llm.usage.total_tokens",
                        response.usage.input_tokens + response.usage.output_tokens,
                    )

            if current_span:
                current_span.set_attribute("llm.response_length", len(result))

            return result

        @noveum_trace.trace_agent(agent_id="summary_agent", agent_type="summarizer")
        def summary_agent(analysis: dict[str, Any], topic: str) -> str:
            """Agent that creates executive summary."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("summary.topic", topic)
                current_span.set_attribute(
                    "summary.input_confidence", analysis.get("confidence_score", 0)
                )

            summary_prompt = f"""Create a concise executive summary for research on '{topic}' based on this analysis:

{analysis['llm_analysis']}

Provide a 2-3 sentence executive summary that captures the key insights."""

            summary = llm_summary_call(summary_prompt, topic)

            if current_span:
                current_span.set_attribute("agent.summary_length", len(summary))

            return summary

        @noveum_trace.trace_llm(
            provider=provider, metadata={"model": model_name, "task": "summarization"}
        )
        def llm_summary_call(prompt: str, topic: str) -> str:
            """LLM call for summarization task."""
            current_span = noveum_trace.get_current_span()
            if current_span:
                current_span.set_attribute("llm.task", "summarization")
                current_span.set_attribute("llm.topic", topic)

            if provider == "openai":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating concise executive summaries.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=50,
                    temperature=0.2,
                )
                result = response.choices[0].message.content

            elif provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": f"As an expert summarizer, {prompt}",
                        }
                    ],
                    max_tokens=50,
                )
                result = response.content[0].text

            return result

        # Execute the full research workflow
        research_topic = "artificial intelligence in healthcare"

        # Create a manual trace to ensure we have a trace context
        trace_id = None
        with noveum_trace.start_trace(
            f"test_research_workflow_{provider}_{model_name}"
        ) as test_trace:
            trace_id = test_trace.trace_id
            workflow_result = research_coordinator(research_topic)

            # Validate workflow results
            assert workflow_result["topic"] == research_topic
            assert "search_results" in workflow_result
            assert "analysis" in workflow_result
            assert "summary" in workflow_result
            assert len(workflow_result["search_results"]) >= 2
            assert len(workflow_result["summary"]) > 0

        # Flush and validate comprehensive trace
        noveum_trace.flush()

        # Give backend additional time to process complex traces with multiple spans
        print(f"⏱️  Waiting 10 seconds for backend to process {trace_id}...")
        time.sleep(10)

        # Validate trace using the captured ID
        assert trace_id is not None, "Trace ID should have been captured"
        trace_data = validate_trace_via_api(trace_id)

        trace = trace_data["trace"]
        assert trace["status"] == "ok"
        assert trace["span_count"] >= 6  # Should have multiple agent and tool spans

        spans = trace["spans"]

        # If spans array is empty but span_count > 0, this is a backend issue - FAIL the test with details
        if trace["span_count"] > 0 and len(spans) == 0:
            print("\n🚨 BACKEND API FAILURE DETECTED:")
            print(f"    Model: {provider} {model_name}")
            print(f"    Trace ID: {trace_id}")
            print(f"    Expected spans: {trace['span_count']}")
            print(f"    Actual spans returned: {len(spans)}")
            print("    Full backend response:")
            import json

            print(json.dumps(trace_data, indent=2, default=str))

            pytest.fail(
                f"BACKEND API BUG: span_count={trace['span_count']} but spans array is empty! "
                f"Backend failed to return stored spans for trace {trace_id}"
            )

        # Validate agent spans exist
        agent_spans = [s for s in spans if "agent.id" in s["attributes"]]

        assert (
            len(agent_spans) >= 3
        ), f"Should have at least 3 agent spans, got {len(agent_spans)}"

        # Validate specific agents
        agent_ids = [s["attributes"]["agent.id"] for s in agent_spans]
        expected_agents = [
            "research_coordinator",
            "search_agent",
            "analysis_agent",
            "summary_agent",
        ]
        for expected_agent in expected_agents:
            assert expected_agent in agent_ids, f"Missing agent: {expected_agent}"

        # Validate LLM spans
        llm_spans = [
            s for s in spans if s["attributes"].get("llm.provider") == provider
        ]
        assert len(llm_spans) >= 2, f"Should have at least 2 LLM spans for {provider}"

        # Validate tool spans
        tool_spans = [s for s in spans if "tool.name" in s["attributes"]]
        assert len(tool_spans) >= 2, "Should have at least 2 tool spans"


# @pytest.mark.integration
# @pytest.mark.asyncio
# @pytest.mark.disable_transport_mocking
# class TestAsyncRealScenarios:
#     """Test async real-world scenarios."""

#     @pytest.mark.parametrize("provider,model_name,client_factory", LLM_MODELS)
#     async def test_concurrent_agent_processing(
#         self, provider: str, model_name: str, client_factory
#     ):
#         """Test concurrent agent processing with real async patterns."""
#         client = client_factory()

#         @noveum_trace.trace_agent(
#             agent_id="parallel_coordinator", agent_type="async_coordinator"
#         )
#         async def parallel_processing_coordinator(tasks: List[str]) -> Dict[str, Any]:
#             """Coordinate parallel processing of multiple tasks."""
#             current_span = noveum_trace.get_current_span()
#             if current_span:
#                 current_span.set_attribute("coordinator.task_count", len(tasks))
#                 current_span.set_attribute("coordinator.processing_type", "parallel")

#             # Process multiple tasks concurrently
#             start_time = time.time()
#             results = await asyncio.gather(
#                 *[process_single_task(task, i) for i, task in enumerate(tasks)]
#             )
#             processing_time = time.time() - start_time

#             coordinator_result = {
#                 "total_tasks": len(tasks),
#                 "results": results,
#                 "processing_time_seconds": round(processing_time, 2),
#                 "parallel_processing": True,
#             }

#             if current_span:
#                 current_span.set_attribute("coordinator.completed_tasks", len(results))
#                 current_span.set_attribute(
#                     "coordinator.processing_time", processing_time
#                 )

#             return coordinator_result

#         @noveum_trace.trace_agent(agent_id="task_processor", agent_type="async_worker")
#         async def process_single_task(task: str, task_id: int) -> Dict[str, Any]:
#             """Process a single task asynchronously."""
#             current_span = noveum_trace.get_current_span()
#             if current_span:
#                 current_span.set_attribute("worker.task_id", task_id)
#                 current_span.set_attribute("worker.task_description", task)

#             # Simulate async processing with real LLM call
#             result = await async_llm_processing(task, task_id)

#             # Simulate additional async work
#             await asyncio.sleep(0.1)

#             task_result = {
#                 "task_id": task_id,
#                 "task": task,
#                 "result": result,
#                 "status": "completed",
#                 "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
#             }

#             if current_span:
#                 current_span.set_attribute("worker.processing_complete", True)
#                 current_span.set_attribute("worker.result_length", len(result))

#             return task_result

#         @noveum_trace.trace_llm(
#             provider=provider, metadata={"model": model_name, "async": True}
#         )
#         async def async_llm_processing(task: str, task_id: int) -> str:
#             """Process task using LLM with async simulation."""
#             current_span = noveum_trace.get_current_span()
#             if current_span:
#                 current_span.set_attribute("llm.task_id", task_id)
#                 current_span.set_attribute("llm.async_processing", True)

#             # Simulate network delay
#             await asyncio.sleep(0.05)

#             prompt = (
#                 f"Process this task efficiently and provide a brief summary: {task}"
#             )

#             if provider == "openai":
#                 response = client.chat.completions.create(
#                     model=model_name,
#                     messages=[
#                         {
#                             "role": "system",
#                             "content": "You are an efficient task processor. Provide brief, actionable summaries.",
#                         },
#                         {"role": "user", "content": prompt},
#                     ],
#                     max_tokens=100,
#                 )
#                 result = response.choices[0].message.content

#                 if current_span:
#                     current_span.set_attribute(
#                         "llm.async_tokens", response.usage.total_tokens
#                     )

#             elif provider == "anthropic":
#                 response = client.messages.create(
#                     model=model_name,
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": f"As an efficient task processor, {prompt}",
#                         }
#                     ],
#                     max_tokens=100,
#                 )
#                 result = response.content[0].text

#                 if current_span:
#                     current_span.set_attribute(
#                         "llm.async_tokens",
#                         response.usage.input_tokens + response.usage.output_tokens,
#                     )

#             return result

#         # Execute concurrent processing test
#         tasks = [
#             "Analyze market trends for Q1 2024",
#             "Generate product recommendations for e-commerce",
#             "Summarize customer feedback from last quarter",
#             "Create performance report for development team",
#         ]

#         # Create a manual trace to ensure we have a trace context
#         trace_id = None
#         with noveum_trace.start_trace(
#             f"test_concurrent_processing_{provider}_{model_name}"
#         ) as test_trace:
#             trace_id = test_trace.trace_id
#             result = await parallel_processing_coordinator(tasks)

#             # Validate concurrent processing results
#             assert result["total_tasks"] == 4
#             assert len(result["results"]) == 4
#             assert result["parallel_processing"] is True
#             assert result["processing_time_seconds"] > 0

#             # Validate all tasks completed successfully
#             for task_result in result["results"]:
#                 assert task_result["status"] == "completed"
#                 assert len(task_result["result"]) > 0
#                 assert task_result["task_id"] in [0, 1, 2, 3]

#         # Flush and validate async traces
#         noveum_trace.flush()
#         time.sleep(3)

#         # Validate trace using the captured ID
#         assert trace_id is not None, "Trace ID should have been captured"
#         trace_data = validate_trace_via_api(trace_id)

#         trace = trace_data["trace"]
#         assert trace["status"] == "ok"

#         spans = trace["spans"]

#         # Should have coordinator, worker, and LLM spans
#         coordinator_spans = [
#             s
#             for s in spans
#             if s["attributes"].get("agent.id") == "parallel_coordinator"
#         ]
#         worker_spans = [
#             s for s in spans if s["attributes"].get("agent.id") == "task_processor"
#         ]
#         llm_spans = [
#             s for s in spans if s["attributes"].get("llm.async_processing") is True
#         ]

#         assert len(coordinator_spans) >= 1, "Should have coordinator span"
#         assert len(worker_spans) >= 4, "Should have worker spans for each task"
#         assert len(llm_spans) >= 4, "Should have LLM spans for each task"

#         # Validate async processing attributes
#         for llm_span in llm_spans:
#             assert llm_span["attributes"]["llm.provider"] == provider
#             assert llm_span["attributes"]["llm.async_processing"] is True
#             assert "llm.async_tokens" in llm_span["attributes"]


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
