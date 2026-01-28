"""
Utility functions for LangChain integration.

This module provides pure utility functions used by the NoveumTraceCallbackHandler
to extract metadata, build attributes, and generate operation names.
"""

import inspect
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache for project root to avoid repeated lookups
_project_root_cache: Optional[Path] = None
_project_root_cache_lock = threading.Lock()


def _is_library_directory(path: Path) -> bool:
    """
    Check if a directory is a known library location (not user code).

    Args:
        path: Directory path to check

    Returns:
        True if this is a library directory, False otherwise
    """
    try:
        # Normalize the path (resolve to absolute, handle symlinks)
        normalized_path = path.resolve()
        path_str = str(normalized_path).lower()
        path_parts = [part.lower() for part in normalized_path.parts]

        # Library directory names that must match exactly
        # This prevents false positives like "frontend" or "inventory"
        # being treated as library directories
        library_dir_names = {
            "site-packages",
            "dist-packages",
            "venv",
            ".venv",
            "env",
            ".env",
            "virtualenv",
        }

        # Check if any directory component exactly matches a library directory name
        if any(part in library_dir_names for part in path_parts):
            return True

        # Patterns that require substring matching (e.g., "lib/python3.9", "python3.11")
        library_substrings: tuple[str, ...] = (
            "lib/python",
            "python3.",
            "python2.",
            "frozen",
            "importlib",
        )
        return any(substring in path_str for substring in library_substrings)
    except Exception:
        # If path resolution fails, fall back to string matching
        path_str = str(path).lower()
        fallback_substrings: tuple[str, ...] = (
            "site-packages",
            "dist-packages",
            "venv",
            ".venv",
            "/env/",
            "/.env/",
            "virtualenv",
            "lib/python",
            "python3.",
            "python2.",
            "frozen",
            "importlib",
        )
        return any(substring in path_str for substring in fallback_substrings)


def _find_project_root(file_path: str) -> Optional[Path]:
    """
    Find the project root directory by identifying the first user code directory.

    This method finds the first directory that:
    1. Contains the file
    2. Is NOT in a known library location (site-packages, venv, etc.)
    3. Is likely the user's project root

    This works in production environments where code might be deployed
    without traditional markers like .git or pyproject.toml.

    Args:
        file_path: Absolute path to a file

    Returns:
        Path to project root if found, None otherwise
    """
    global _project_root_cache

    # Use cached value if available (thread-safe check)
    with _project_root_cache_lock:
        if _project_root_cache is not None:
            return _project_root_cache

    try:
        path = Path(file_path).resolve()

        # If it's a file, start from its parent directory
        if path.is_file():
            current = path.parent
        else:
            current = path

        # Walk up the directory tree
        while current != current.parent:  # Stop at filesystem root
            # Check if this directory is NOT a library location
            if not _is_library_directory(current):
                with _project_root_cache_lock:
                    # This looks like user code! Use it as project root
                    _project_root_cache = current
                    return current

            # Move up one level
            current = current.parent

        # If we reached filesystem root without finding user code, return None
        return None
    except Exception:
        # If anything fails, return None
        return None


def _make_path_relative(file_path: str) -> str:
    """
    Convert an absolute file path to a relative path from project root.

    If project root cannot be determined, returns just the filename.

    Args:
        file_path: Absolute file path

    Returns:
        Relative path from project root, or just filename if root not found
    """
    try:
        project_root = _find_project_root(file_path)
        if project_root is None:
            # If we can't find project root, just return the filename
            return os.path.basename(file_path)

        abs_path = Path(file_path).resolve()
        try:
            relative_path = abs_path.relative_to(project_root)
            return str(relative_path)
        except ValueError:
            # Path is not under project root, return filename
            return os.path.basename(file_path)
    except Exception:
        # If anything fails, return filename
        return os.path.basename(file_path)


def extract_code_location_info(skip_frames: int = 0) -> Optional[dict[str, Any]]:
    """
    Extract code location information from the call stack.

    Walks up the call stack to find the first frame that's in user code,
    skipping library code and standard library.

    This function intelligently identifies user code by:
    - Skipping files in known library locations (site-packages, venv, etc.)
    - Skipping Python standard library code
    - Skipping LangChain internal code (but not user code that uses LangChain)
    - NOT skipping noveum_trace if it's part of the user's codebase
    - Finding the actual calling code in the user's application

    This approach works in production environments where the SDK might be
    installed in various ways (pip, local development, etc.).

    Args:
        skip_frames: Number of frames to skip from the top (default: 0)

    Returns:
        Dict with code location info: {
            "code.file": str,  # Relative path from project root
            "code.line": int,
            "code.function": str,
            "code.module": str,
            "code.context": Optional[str]  # The actual line of code
        }
        Returns None if no user code frame found
    """
    try:
        stack = inspect.stack()

        # Skip the current frame (this function) and any requested frames
        start_idx = 1 + skip_frames

        for frame_info in stack[start_idx:]:
            filename = frame_info.filename

            # Skip if this is a builtin or internal Python code
            if filename.startswith("<"):
                continue

            # Skip if this frame is in a known library location
            # We check the directory, not just the filename, to be more accurate
            try:
                file_path = Path(filename).resolve()
                # Check if the file is in a library directory
                if _is_library_directory(file_path.parent):
                    continue
            except Exception:
                # If we can't resolve the path, fall back to string matching
                filename_lower = filename.lower()
                library_patterns = [
                    "site-packages",
                    "dist-packages",
                    "venv",
                    ".venv",
                    "env/",
                    "virtualenv",
                    "lib/python",
                    "frozen",
                    "importlib",
                ]
                if any(pattern in filename_lower for pattern in library_patterns):
                    continue

            # Skip LangChain internal code (but not user code that uses LangChain)
            # Only skip if it's clearly in the langchain package directory
            if "langchain" in filename.lower():
                # Check if it's in site-packages/langchain or similar
                try:
                    if _is_library_directory(Path(filename).parent):
                        continue
                except Exception:
                    # If we can't check the path, skip langchain files in library locations
                    if any(
                        pattern in filename.lower()
                        for pattern in ["site-packages", "dist-packages"]
                    ):
                        continue

            # Found user code! Extract information
            try:
                code_context = None
                if frame_info.code_context and len(frame_info.code_context) > 0:
                    code_context = frame_info.code_context[0].strip()

                module_name = "unknown"
                function_def_info = None
                try:
                    frame = frame_info.frame
                    module_name = frame.f_globals.get("__name__", "unknown")

                    # Try to get function definition info from the frame
                    func_name = frame_info.function
                    if func_name and func_name != "<module>":
                        # Try to get the function object from the frame
                        func_obj = frame.f_locals.get(func_name) or frame.f_globals.get(
                            func_name
                        )
                        if func_obj and callable(func_obj):
                            # Extract function definition info
                            function_def_info = extract_function_definition_info(
                                func_obj
                            )
                except Exception:
                    pass

                # Make path relative to project root
                relative_file = _make_path_relative(filename)

                result = {
                    "code.file": relative_file,
                    "code.line": frame_info.lineno,
                    "code.function": frame_info.function,
                    "code.module": module_name,
                    "code.context": code_context,
                }

                # Add function definition info if available
                if function_def_info:
                    result.update(function_def_info)

                return result
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
        Dict with:
        - 'name': Custom span name (if present)
        - 'parent_name': Parent span name (if present)
        - 'metadata': Dict of all other custom fields to add as span attributes
    """
    if not metadata:
        return {}

    noveum_config = metadata.get("noveum", {})
    if not isinstance(noveum_config, dict):
        return {}

    result = {}
    custom_metadata = {}

    for key, value in noveum_config.items():
        if key == "name":
            result["name"] = value
        elif key == "parent_name":
            result["parent_name"] = value
        else:
            custom_metadata[key] = value

    if custom_metadata:
        result["metadata"] = custom_metadata

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
        return f"{event_type}.node"

    name = serialized.get("name", "node")

    if event_type == "llm_start":
        # Use model name instead of class name for better readability
        model_name = extract_model_name(serialized)
        # If extract_model_name returns "unknown" for an empty dict, use "node" instead
        # to match the pattern used by other event types
        if model_name == "unknown" and not serialized:
            model_name = "node"
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


def extract_function_definition_info(func: Any) -> Optional[dict[str, Any]]:
    """
    Extract function definition information (file, start line, end line).

    Args:
        func: Function object

    Returns:
        Dict with function definition info:
        {
            "function.definition.file": str,  # Relative path from project root
            "function.definition.start_line": int,
            "function.definition.end_line": int,
        }
        Returns None if extraction fails
    """
    try:
        if not callable(func):
            return None

        # Get source lines
        source_lines, start_line = inspect.getsourcelines(func)
        end_line = start_line + len(source_lines) - 1

        # Get file path and make it relative
        file_path = inspect.getfile(func)
        relative_file = _make_path_relative(file_path)

        return {
            "function.definition.file": relative_file,
            "function.definition.start_line": start_line,
            "function.definition.end_line": end_line,
        }
    except (OSError, TypeError, AttributeError):
        # Function might be builtin, from C extension, or not have source
        return None


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


def extract_available_tools(
    serialized: Optional[dict[str, Any]],
    metadata: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Extract available tools from agent serialized data or metadata.

    Priority:
    1. Manual injection via metadata.noveum.available_tools (PRIMARY - most reliable)
    2. Auto-detection from serialized.kwargs.tools (FALLBACK - convenience)

    Args:
        serialized: LangChain serialized agent/chain data
        metadata: LangChain metadata dict

    Returns:
        List of dicts with full tool schema:
        [
            {
                "name": str,
                "description": str,
                "args_schema": dict (JSON schema) or None,
            }
        ]
    """
    # Priority 1: Check metadata for manually injected tools (RECOMMENDED)
    if metadata:
        noveum_config = metadata.get("noveum", {})
        if isinstance(noveum_config, dict) and (
            manual_tools := noveum_config.get("available_tools")
        ):
            tools_list = _convert_tools_to_dict_list(manual_tools)
            if tools_list:
                logger.debug(
                    f"Extracted {len(tools_list)} tools from metadata (manual injection)"
                )
                return tools_list

    # Priority 2: Fallback to auto-detection from serialized data
    if serialized:
        kwargs = serialized.get("kwargs", {})
        if auto_tools := kwargs.get("tools"):
            tools_list = _convert_tools_to_dict_list(auto_tools)
            if tools_list:
                logger.debug(
                    f"Extracted {len(tools_list)} tools from serialized data (auto-detection)"
                )
                return tools_list

    return []


def _tool_to_dict(tool: Any) -> dict[str, Any]:
    """
    Convert a single tool to a standardized dict format.

    Tries conversion strategies in order:
    1. Pydantic model serialization (.model_dump(), .dict())
    2. OpenAI function calling format {type: "function", function: {...}}
    3. Built-in dict() conversion (works for most objects)

    Args:
        tool: A tool in any format

    Returns:
        Dict with keys: name, description, args_schema
    """
    tool_data = None

    try:
        # Strategy 1: Pydantic serialization (Pydantic v2)
        if hasattr(tool, "model_dump"):
            tool_data = tool.model_dump()

        # Strategy 1: Pydantic serialization (Pydantic v1)
        elif hasattr(tool, "dict"):
            tool_data = tool.dict()

        # Strategy 2: OpenAI function calling format (from invocation_params)
        elif (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and "function" in tool
        ):
            tool_data = tool["function"]

        # Strategy 3: Convert to dict
        else:
            tool_data = dict(tool)

    except Exception as e:
        logger.debug(f"Error converting tool to dict: {e}")
        return {"name": "unknown", "description": "No description", "args_schema": None}

    # Normalize keys (handle different naming conventions)
    return {
        "name": tool_data.get("name", "unknown"),
        "description": tool_data.get("description", "No description"),
        "args_schema": tool_data.get("args_schema")
        or tool_data.get("parameters")
        or tool_data.get("args"),
    }


def _convert_tools_to_dict_list(tools: Any) -> list[dict[str, Any]]:
    """
    Convert tools to a list of standardized dicts.

    Handles any tool format by trying multiple conversion strategies.

    Args:
        tools: List of tools in any format, or a single tool

    Returns:
        List of dicts with keys: name, description, args_schema
    """
    if not tools:
        return []

    result: list[dict[str, Any]] = []

    try:
        # Handle both list and single tool
        tools_iter = tools if isinstance(tools, (list, tuple)) else [tools]

        logger.debug(f"Converting {len(tools_iter)} tools to dict list")

        for tool in tools_iter:
            try:
                tool_dict = _tool_to_dict(tool)

                # Only add if we got a valid name
                if tool_dict.get("name") and tool_dict["name"] != "unknown":
                    result.append(tool_dict)
                    logger.debug(f"✅ Added tool: {tool_dict['name']}")
                else:
                    logger.debug(
                        f"❌ Skipped tool with invalid name: {tool_dict.get('name')}"
                    )

            except Exception as e:
                logger.debug(f"Error converting tool to dict: {e}")
                continue

    except Exception as e:
        logger.error(f"Error processing tools list: {e}")
        return []

    return result
