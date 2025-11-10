"""
Utility functions for LangChain integration.

This module provides pure utility functions used by the NoveumTraceCallbackHandler
to extract metadata, build attributes, and generate operation names.
"""

import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_call_site_info(skip_frames: int = 0) -> Optional[dict[str, Any]]:
    """
    Extract call site information from the call stack.

    Walks up the call stack to find the first frame that's NOT in:
    - LangChain code (langchain, langchain_core, langchain_openai, etc.)
    - Noveum Trace code (noveum_trace)
    - Python standard library

    Args:
        skip_frames: Number of frames to skip from the top (default: 0)

    Returns:
        Dict with call site info: {
            "call_site.file": str,
            "call_site.line": int,
            "call_site.function": str,
            "call_site.module": str,
            "call_site.code_context": Optional[str]  # The actual line of code
        }
        Returns None if no user code frame found
    """
    try:
        stack = inspect.stack()

        # Skip the current frame (this function) and any requested frames
        start_idx = 1 + skip_frames

        # Patterns to skip (internal libraries)
        skip_patterns = [
            "langchain",
            "noveum_trace",
            "site-packages",
            "<frozen",
            "importlib",
        ]

        for frame_info in stack[start_idx:]:
            filename = frame_info.filename

            # Skip if this frame is in a library we want to ignore
            if any(pattern in filename for pattern in skip_patterns):
                continue

            # Skip if this is a builtin or internal Python code
            if filename.startswith("<") or "site-packages" in filename:
                continue

            # Found user code! Extract information
            try:
                code_context = None
                if frame_info.code_context and len(frame_info.code_context) > 0:
                    code_context = frame_info.code_context[0].strip()

                module_name = "unknown"
                try:
                    frame = frame_info.frame
                    module_name = frame.f_globals.get("__name__", "unknown")
                except Exception:
                    pass

                return {
                    "call_site.file": filename,
                    "call_site.line": frame_info.lineno,
                    "call_site.function": frame_info.function,
                    "call_site.module": module_name,
                    "call_site.code_context": code_context,
                }
            except Exception:
                # If extraction fails, continue to next frame
                continue

        # No user code frame found
        return None

    except Exception:
        # If stack inspection fails, return None (fail gracefully)
        return None


def extract_noveum_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Extract metadata.noveum configuration.

    Args:
        metadata: LangChain metadata dict

    Returns:
        Dict with 'name' and 'parent_name' keys if present
    """
    if not metadata:
        return {}

    noveum_config = metadata.get("noveum", {})
    if not isinstance(noveum_config, dict):
        return {}

    # Only extract 'name' and 'parent_name'
    result = {}
    if "name" in noveum_config:
        result["name"] = noveum_config["name"]
    if "parent_name" in noveum_config:
        result["parent_name"] = noveum_config["parent_name"]

    return result


def get_operation_name(
    event_type: str,
    serialized: Optional[dict[str, Any]],
    langgraph_metadata: Optional[dict[str, Any]] = None,
) -> str:
    """
    Generate standardized operation names with LangGraph support.

    Args:
        event_type: Type of event (llm_start, chain_start, etc.)
        serialized: Serialized object dict (may be None for LangGraph)
        langgraph_metadata: Optional LangGraph metadata dict

    Returns:
        Operation name string (e.g., "graph.node.research" or "chain.unknown")
    """
    # For chain_start, check LangGraph first (works even with None serialized)
    if event_type == "chain_start":
        if langgraph_metadata and langgraph_metadata.get("is_langgraph"):
            return get_langgraph_operation_name(langgraph_metadata, "unknown")

    # For other event types or non-LangGraph chains, need serialized
    if serialized is None:
        return f"{event_type}.unknown"

    name = serialized.get("name", "unknown")

    if event_type == "llm_start":
        # Use model name instead of class name for better readability
        model_name = extract_model_name(serialized)
        return f"llm.{model_name}"
    elif event_type == "chain_start":
        # Regular chain naming (LangGraph case handled above)
        return f"chain.{name}"
    elif event_type == "agent_start":
        return f"agent.{name}"
    elif event_type == "retriever_start":
        return f"retrieval.{name}"
    elif event_type == "tool_start":
        return f"tool.{name}"

    return f"{event_type}.{name}"


def get_langgraph_operation_name(
    langgraph_metadata: dict[str, Any], fallback_name: str
) -> str:
    """
    Generate LangGraph-aware operation names.

    Args:
        langgraph_metadata: LangGraph metadata dict
        fallback_name: Fallback name if no LangGraph info available

    Returns:
        Operation name string (e.g., "graph.node.research")
    """
    # Check if we have a node name (most specific)
    node_name = langgraph_metadata.get("node_name")
    if node_name:
        return f"graph.node.{node_name}"

    # Check if we have a graph name
    graph_name = langgraph_metadata.get("graph_name")
    if graph_name:
        return f"graph.{graph_name}"

    # Check if we have a step number
    step = langgraph_metadata.get("step")
    if step is not None:
        return f"graph.node.step_{step}"

    # Fallback to generic graph naming
    if fallback_name and fallback_name != "unknown":
        return f"graph.{fallback_name}"

    # Ultimate fallback
    return "graph.unknown"


def extract_model_name(serialized: dict[str, Any]) -> str:
    """Extract model name from serialized LLM data."""
    if not serialized:
        return "unknown"

    # Try to get model name from kwargs
    kwargs = serialized.get("kwargs", {})
    model = kwargs.get("model")
    if model:
        return model

    # Fallback to provider name
    id_path = serialized.get("id", [])
    if len(id_path) >= 2:
        # e.g., "openai" from ["langchain", "chat_models", "openai", "ChatOpenAI"]
        return id_path[-2]

    # Final fallback to class name
    return serialized.get("name", "unknown")


def extract_agent_type(serialized: dict[str, Any]) -> str:
    """Extract agent type from serialized agent data."""
    if not serialized:
        return "unknown"

    # Get agent category from ID path
    id_path = serialized.get("id", [])
    if len(id_path) >= 2:
        # e.g., "react" from ["langchain", "agents", "react", "ReActAgent"]
        return id_path[-2]

    return "unknown"


def extract_agent_capabilities(serialized: dict[str, Any]) -> str:
    """Extract agent capabilities from tools in serialized data."""
    if not serialized:
        return "unknown"

    capabilities = []
    kwargs = serialized.get("kwargs", {})
    tools = kwargs.get("tools", [])

    if tools:
        capabilities.append("tool_usage")

        # Extract specific tool types
        tool_types = set()
        for tool in tools:
            if isinstance(tool, dict):
                tool_name = tool.get("name", "").lower()
                if "search" in tool_name or "web" in tool_name:
                    tool_types.add("web_search")
                elif "calc" in tool_name or "math" in tool_name:
                    tool_types.add("calculation")
                elif "file" in tool_name or "read" in tool_name:
                    tool_types.add("file_operations")
                elif "api" in tool_name or "request" in tool_name:
                    tool_types.add("api_calls")
                else:
                    tool_types.add(
                        tool.get("name", "other") if tool.get("name") else "other"
                    )

        if tool_types:
            capabilities.extend(tool_types)

    # Add default capabilities
    if not capabilities:
        capabilities = ["reasoning"]

    return ",".join(capabilities)


def extract_tool_function_name(serialized: dict[str, Any]) -> str:
    """Extract function name from serialized tool data."""
    if not serialized:
        return "unknown"

    kwargs = serialized.get("kwargs", {})
    func_name = kwargs.get("name")
    if func_name:
        return func_name

    # Fallback to class name
    return serialized.get("name", "unknown")


def extract_langgraph_metadata(
    metadata: Optional[dict[str, Any]],
    tags: Optional[list[str]],
    serialized: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract LangGraph-specific metadata from callback parameters.

    This method safely extracts LangGraph metadata from various sources
    with comprehensive fallbacks to ensure it never breaks regular LangChain usage.

    Args:
        metadata: LangChain metadata dict (may contain LangGraph keys)
        tags: List of tags (may contain graph indicators)
        serialized: Serialized object dict (contains type info, may be None)

    Returns:
        Dict with LangGraph metadata, all fields are optional and safe
    """
    result: dict[str, Any] = {
        "is_langgraph": False,
        "node_name": None,
        "step": None,
        "graph_name": None,
        "checkpoint_ns": None,
        "execution_type": None,
    }

    # 1. Try to extract from metadata (primary source)
    try:
        if metadata and isinstance(metadata, dict):
            # LangGraph-specific metadata keys
            result["node_name"] = metadata.get("langgraph_node")
            result["step"] = metadata.get("langgraph_step")
            result["graph_name"] = metadata.get("langgraph_graph_name")
            result["checkpoint_ns"] = metadata.get("langgraph_checkpoint_ns")

            # Extract path information if available
            langgraph_path = metadata.get("langgraph_path")
            if langgraph_path and isinstance(langgraph_path, (list, tuple)):
                # Path format: ('__pregel_pull', 'node_name', ...)
                # Extract the actual node name (skip internal nodes)
                for part in langgraph_path:
                    if isinstance(part, str) and not part.startswith("__"):
                        if not result["node_name"]:
                            result["node_name"] = part
                        break
    except Exception:
        # Silent fallback - metadata extraction failed
        pass

    # 2. Try to extract from tags (secondary source)
    try:
        if tags and isinstance(tags, list):
            # Look for LangGraph indicators in tags
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("langgraph:"):
                    # Extract node name from tag like "langgraph:node_name"
                    parts = tag.split(":", 1)
                    if len(parts) == 2 and not result["node_name"]:
                        result["node_name"] = parts[1]
    except Exception:
        # Silent fallback - tag extraction failed
        pass

    # 3. Try to extract from serialized dict (tertiary source)
    # Note: LangGraph often passes None for serialized, so this is optional
    try:
        if serialized and isinstance(serialized, dict):
            # Check if this is a LangGraph type
            id_path = serialized.get("id", [])
            if isinstance(id_path, list) and any(
                "langgraph" in str(part).lower() for part in id_path
            ):
                # Look for langgraph in the ID path
                result["is_langgraph"] = True

                # Try to extract graph name from serialized name
                name = serialized.get("name", "")
                if not result["graph_name"] and name and isinstance(name, str):
                    result["graph_name"] = name
    except Exception:
        # Silent fallback - serialized extraction failed
        pass

    # 4. Determine if this is LangGraph execution
    # Only mark as LangGraph if we found clear indicators
    result["is_langgraph"] = bool(
        result["node_name"]
        or result["step"] is not None
        or result["checkpoint_ns"]
        or result["is_langgraph"]  # Set by serialized check
    )

    # 5. Determine execution type
    if result["is_langgraph"]:
        if result["node_name"]:
            result["execution_type"] = "node"
        elif result["graph_name"]:
            result["execution_type"] = "graph"
        else:
            result["execution_type"] = "unknown"

    return result


def build_langgraph_attributes(langgraph_metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Build span attributes from LangGraph metadata.

    Only includes attributes that have actual values to avoid
    cluttering spans with None values.

    Args:
        langgraph_metadata: Dict from extract_langgraph_metadata()

    Returns:
        Dict of span attributes to add (may be empty)
    """
    attributes: dict[str, Any] = {}

    # Only add attributes if we have LangGraph data
    if not langgraph_metadata.get("is_langgraph"):
        return attributes

    # Add LangGraph indicator
    attributes["langgraph.is_graph"] = True

    # Add node name if available
    if langgraph_metadata.get("node_name"):
        attributes["langgraph.node_name"] = langgraph_metadata["node_name"]

    # Add step number if available
    if langgraph_metadata.get("step") is not None:
        attributes["langgraph.step"] = langgraph_metadata["step"]

    # Add graph name if available
    if langgraph_metadata.get("graph_name"):
        attributes["langgraph.graph_name"] = langgraph_metadata["graph_name"]

    # Add checkpoint namespace if available
    if langgraph_metadata.get("checkpoint_ns"):
        attributes["langgraph.checkpoint_ns"] = langgraph_metadata["checkpoint_ns"]

    # Add execution type if available
    if langgraph_metadata.get("execution_type"):
        attributes["langgraph.execution_type"] = langgraph_metadata["execution_type"]

    return attributes


def build_routing_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Build routing span attributes from payload.

    Captures all fields provided by the user, with special handling
    for known routing fields.

    Args:
        payload: Routing decision data from user

    Returns:
        Dictionary of span attributes
    """
    attributes = {}

    # Core routing attributes (always present)
    attributes["routing.source_node"] = payload.get("source_node", "unknown")
    attributes["routing.target_node"] = payload.get("target_node", "unknown")
    attributes["routing.decision"] = payload.get(
        "decision", payload.get("target_node", "unknown")
    )
    attributes["routing.type"] = "conditional_edge"

    # Optional but common attributes
    if "reason" in payload:
        attributes["routing.reason"] = str(payload["reason"])

    if "confidence" in payload:
        attributes["routing.confidence"] = float(payload["confidence"])

    # Tool/option scores (expanded into individual attributes)
    if "tool_scores" in payload:
        tool_scores = payload["tool_scores"]
        # Store as JSON string for full data
        attributes["routing.tool_scores"] = str(tool_scores)
        # Also store individual scores as separate attributes
        if isinstance(tool_scores, dict):
            for tool, score in tool_scores.items():
                attributes[f"routing.score.{tool}"] = float(score)

    # Alternatives
    if "alternatives" in payload:
        alternatives = payload["alternatives"]
        attributes["routing.alternatives"] = str(alternatives)
        if isinstance(alternatives, list):
            attributes["routing.alternatives_count"] = len(alternatives)

    # State snapshot (if provided)
    if "state_snapshot" in payload:
        state_snapshot = payload["state_snapshot"]
        attributes["routing.state_snapshot"] = str(state_snapshot)

    # Capture ANY other fields provided by the user
    # This ensures we don't lose any custom data
    known_fields = {
        "source_node",
        "target_node",
        "decision",
        "reason",
        "confidence",
        "tool_scores",
        "alternatives",
        "state_snapshot",
    }

    for key, value in payload.items():
        if key not in known_fields:
            # Prefix with "routing." and convert to string
            attr_key = f"routing.{key}"
            attributes[attr_key] = str(value)

    return attributes
