# LangChain Code Tracing Explained

This document explains how the code tracing system works in the Noveum Trace SDK for LangChain integration, including function name tracing, line numbers, and other details.

## Overview

When you use LangChain with Noveum Trace, the SDK automatically captures detailed information about where LLM and tool calls are made in your code. This includes:

1. **Call Site Information**: Where the call was made (file, line, function)
2. **Function Definition Information**: Where the function is defined (file, start line, end line)
3. **Code Context**: The actual line of code that made the call

## How It Works

### Step 1: Call Stack Inspection

When an LLM or tool is called, the callback handler triggers `extract_call_site_info()`, which uses Python's `inspect.stack()` to walk up the call stack.

```python
# Example call stack when you call llm.invoke():
# Frame 0: extract_call_site_info() [this function - skipped]
# Frame 1: on_llm_start() [callback handler - skipped with skip_frames=1]
# Frame 2: langchain internal code [skipped - library location]
# Frame 3: Your code! [CAPTURED - this is what we want]
```

### Step 2: Finding User Code

The system intelligently skips library code and finds your actual code:

```python
def extract_call_site_info(skip_frames: int = 0):
    stack = inspect.stack()
    start_idx = 1 + skip_frames
    
    for frame_info in stack[start_idx:]:
        filename = frame_info.filename
        
        # Skip standard library
        if filename.startswith("<"):
            continue
        
        # Skip library locations (site-packages, venv, etc.)
        try:
            file_path = Path(filename).resolve()
            if _is_library_directory(file_path.parent):
                continue
        except Exception:
            # Fallback to string matching
            if any(pattern in filename.lower() for pattern in ["site-packages", "dist-packages", "venv", ...]):
                continue
        
        # Skip LangChain internal code (but not your code using LangChain)
        if "langchain" in filename.lower():
            try:
                if _is_library_directory(Path(filename).parent):
                    continue
            except Exception:
                if any(pattern in filename.lower() for pattern in ["site-packages", "dist-packages"]):
                    continue
        
        # Found user code! Extract information
        code_context = frame_info.code_context[0].strip() if frame_info.code_context else None
        module_name = frame_info.frame.f_globals.get("__name__", "unknown")
        relative_file = _make_path_relative(filename)
        
        # Try to get function definition info from the frame (if available)
        function_def_info = None
        func_name = frame_info.function
        if func_name and func_name != "<module>":
            func_obj = frame_info.frame.f_locals.get(func_name) or frame_info.frame.f_globals.get(func_name)
            if func_obj and callable(func_obj):
                function_def_info = extract_function_definition_info(func_obj)
        
        result = {
            "code.file": relative_file,
            "code.line": frame_info.lineno,
            "code.function": frame_info.function,
            "code.module": module_name,
            "code.context": code_context,
        }
        
        # Add function definition info if available
        # (for tools, this is also extracted separately from the tool object)
        if function_def_info:
            result.update(function_def_info)
        
        return result
```

### Step 3: Extracting Information

For each user code frame found, we extract:

#### 1. File Path (`code.file`)
- **What**: Relative path from project root
- **How**: Uses `_make_path_relative()` which:
  1. Finds project root by walking up until finding a non-library directory
  2. Converts absolute path to relative path
  3. Falls back to filename if project root can't be determined

**Example**:
```python
# Absolute: /Users/you/project/src/agent.py
# Relative: src/agent.py
```

#### 2. Line Number (`code.line`)
- **What**: Exact line number where the call was made
- **How**: From `frame_info.lineno` in the call stack

**Example**:
```python
# Line 47 in your file
response = llm.invoke("Hello")  # <- This line
```

#### 3. Function Name (`code.function`)
- **What**: Name of the function where the call was made
- **How**: From `frame_info.function` in the call stack

**Example**:
```python
def process_query():  # <- Function name: "process_query"
    response = llm.invoke("Hello")
```

#### 4. Module Name (`code.module`)
- **What**: Python module name (e.g., `myapp.agents`)
- **How**: From `frame.f_globals.get("__name__")`

**Example**:
```python
# File: myapp/agents.py
# Module: "myapp.agents"
```

#### 5. Code Context (`code.context`)
- **What**: The actual line of code that made the call
- **How**: From `frame_info.code_context[0]` (the source line)

**Example**:
```python
# code.context: "response = llm.invoke(\"Hello\")"
```

### Step 4: Function Definition Information

For tools, we also extract where the function is **defined** (not just where it's called):

```python
def extract_function_definition_info(func):
    # Get source lines of the function
    source_lines, start_line = inspect.getsourcelines(func)
    end_line = start_line + len(source_lines) - 1
    
    # Get file path
    file_path = inspect.getfile(func)
    relative_file = _make_path_relative(file_path)
    
    return {
        "function.definition.file": relative_file,
        "function.definition.start_line": start_line,
        "function.definition.end_line": end_line,
    }
```

**Example**:
```python
# Tool function defined at lines 10-20 in tools.py
@tool
def search_web(query: str):  # <- Line 10
    # ... function body ...
    return results  # <- Line 20

# When called from main.py line 50:
# - code.file: "main.py"
# - code.line: 50
# - function.definition.file: "tools.py"
# - function.definition.start_line: 10
# - function.definition.end_line: 20
```

## Complete Example

Let's trace through a complete example:

### Your Code

```python
# File: docs/examples/noveum_support_agent/main.py
# Line 324

def answer_question(question: str):
    """Answer a question using RAG and web search."""
    # Line 330
    result = agent.invoke({"input": question})  # <- LLM call here
    return result
```

### What Gets Captured

When `agent.invoke()` triggers an LLM call, the callback handler captures:

```python
{
    # Call site information (where the call was made)
    "code.file": "docs/examples/noveum_support_agent/main.py",
    "code.line": 330,
    "code.function": "answer_question",
    "code.module": "__main__",  # or "main" if imported
    "code.context": "result = agent.invoke({\"input\": question})",
    
    # Standard LangChain attributes
    "langchain.run_id": "abc-123",
    "llm.model": "gpt-4",
    "llm.provider": "openai",
    # ... other LLM attributes
}
```

### For Tool Calls

If you have a tool:

```python
# File: docs/examples/noveum_support_agent/tools.py
# Lines 45-60

@tool
def rag_search_tool(query: str, k: int = 3) -> str:
    """Search the RAG database."""
    vector_store = get_vector_store()
    results = search_rag(query, vector_store, k=k)
    return json.dumps(results, indent=2)
```

When this tool is called, we capture:

```python
{
    # Call site (where tool was invoked)
    "code.file": "docs/examples/noveum_support_agent/main.py",
    "code.line": 330,
    "code.function": "answer_question",
    "code.module": "__main__",
    "code.context": "result = agent.invoke({\"input\": question})",
    
    # Function definition (where tool function is defined)
    "function.definition.file": "docs/examples/noveum_support_agent/tools.py",
    "function.definition.start_line": 45,
    "function.definition.end_line": 60,
    
    # Standard tool attributes
    "tool.name": "rag_search_tool",
    "tool.operation": "rag_search_tool",
    # ... other tool attributes
}
```

## How Project Root Detection Works

The system finds your project root by walking up the directory tree until it finds a directory that's **not** in a library location:

```python
def _find_project_root(file_path: str):
    path = Path(file_path).resolve()
    current = path.parent
    
    while current != current.parent:  # Walk up
        # Check if this is NOT a library directory
        if not _is_library_directory(current):
            return current  # Found project root!
        
        current = current.parent  # Move up
    
    return None  # Couldn't find project root
```

**Library locations** (skipped):
- `site-packages/`
- `dist-packages/`
- `venv/`, `.venv/`
- `env/`, `.env/`
- `virtualenv/`
- `lib/python3.x/`
- Standard library paths

**User code locations** (captured):
- Your project directory
- Any directory not in the above list

## Benefits

1. **Debugging**: Quickly find where LLM/tool calls originate
2. **Performance Analysis**: Identify hotspots in your code
3. **Code Navigation**: Jump directly to the calling code
4. **Production-Friendly**: Works regardless of deployment structure
5. **Flexible**: Works whether SDK is installed via pip or part of your codebase

## Edge Cases Handled

1. **No Project Root Found**: Falls back to just filename
2. **Path Resolution Fails**: Falls back to string matching
3. **Function Not Found**: Gracefully skips function definition info
4. **Library Code**: Intelligently skipped
5. **Nested Calls**: Finds the actual user code, not intermediate wrappers

## Summary

The tracing system:
1. ✅ Walks the call stack to find user code
2. ✅ Skips library code intelligently
3. ✅ Extracts file, line, function, module, and code context
4. ✅ For tools, also extracts function definition location
5. ✅ Makes paths relative to project root
6. ✅ Works in production environments

This gives you complete visibility into where and how your LLM and tool calls are made!

