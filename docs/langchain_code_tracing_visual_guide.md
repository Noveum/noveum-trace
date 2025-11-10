# Code Tracing Visual Guide

This guide shows exactly what information is captured when you use Noveum Trace with LangChain.

## Visual Flow

```
Your Code
    ↓
LLM/Tool Call
    ↓
LangChain Callback Handler
    ↓
extract_call_site_info()
    ↓
Walk Call Stack
    ↓
Find User Code Frame
    ↓
Extract Information
    ↓
Add to Span Attributes
```

## Example 1: LLM Call

### Your Code

```python
# File: docs/examples/noveum_support_agent/main.py
# Line 324

def answer_question(self, question: str) -> str:
    """Answer a question using RAG and web search."""
    # ... setup code ...
    
    # Line 330 - LLM call
    response = self.llm.invoke(prompt)  # ← This line triggers tracing
    return response.content
```

### Call Stack When `llm.invoke()` is Called

```
Frame 0: extract_call_site_info()          [SKIPPED - this function]
Frame 1: on_llm_start()                     [SKIPPED - callback handler]
Frame 2: langchain/chat_models/base.py      [SKIPPED - library code]
Frame 3: main.py:330 (answer_question)     [CAPTURED! ← Your code]
```

### What Gets Captured

```python
{
    # ===== CALL SITE INFORMATION =====
    # Where the call was made
    "code.file": "docs/examples/noveum_support_agent/main.py",
    "code.line": 330,
    "code.function": "answer_question",
    "code.module": "__main__",
    "code.context": "response = self.llm.invoke(prompt)",
    
    # ===== STANDARD LLM ATTRIBUTES =====
    "langchain.run_id": "abc-123-def-456",
    "llm.model": "gpt-4",
    "llm.provider": "openai",
    "llm.operation": "completion",
    "llm.input.prompts": ["You are a helpful assistant..."],
    "llm.input.prompt_count": 1,
    # ... other LLM attributes
}
```

## Example 2: Tool Call

### Your Code

```python
# File: docs/examples/noveum_support_agent/tools.py
# Lines 37-50

@tool
def rag_search_tool(query: str, k: int = 3) -> str:
    """
    Search the RAG database for relevant information.
    """
    vector_store = get_vector_store()
    results = search_rag(query, vector_store, k=k)
    return json.dumps(results, indent=2)
```

### When Tool is Called from Agent

```python
# File: docs/examples/noveum_support_agent/main.py
# Line 280

# Agent calls the tool internally
result = agent.invoke({"input": question})
```

### Call Stack When Tool is Executed

```
Frame 0: extract_call_site_info()          [SKIPPED]
Frame 1: on_tool_start()                    [SKIPPED]
Frame 2: langchain/tools/base.py            [SKIPPED - library]
Frame 3: langchain/agents/agent.py          [SKIPPED - library]
Frame 4: main.py:280 (answer_question)      [CAPTURED! ← Your code]
```

### What Gets Captured

```python
{
    # ===== CALL SITE INFORMATION =====
    # Where the tool was invoked (from agent)
    "code.file": "docs/examples/noveum_support_agent/main.py",
    "code.line": 280,
    "code.function": "answer_question",
    "code.module": "__main__",
    "code.context": "result = agent.invoke({\"input\": question})",
    
    # ===== FUNCTION DEFINITION INFORMATION =====
    # Where the tool function is defined
    "function.definition.file": "docs/examples/noveum_support_agent/tools.py",
    "function.definition.start_line": 37,
    "function.definition.end_line": 50,
    
    # ===== STANDARD TOOL ATTRIBUTES =====
    "langchain.run_id": "xyz-789-abc-123",
    "tool.name": "rag_search_tool",
    "tool.operation": "rag_search_tool",
    "tool.input.query": "How to use Noveum Trace?",
    "tool.input.k": 3,
    # ... other tool attributes
}
```

## Information Breakdown

### 1. Call Site Information (`code.*`)

**Purpose**: Shows where the call was made in your code.

| Attribute | Example | Description |
|-----------|---------|-------------|
| `code.file` | `docs/examples/noveum_support_agent/main.py` | Relative path from project root |
| `code.line` | `330` | Exact line number where call was made |
| `code.function` | `answer_question` | Function name where call was made |
| `code.module` | `__main__` | Python module name |
| `code.context` | `response = self.llm.invoke(prompt)` | The actual line of code |

**How it works**:
1. Walks up the call stack
2. Skips library code (site-packages, venv, etc.)
3. Finds first user code frame
4. Extracts file, line, function, module, and code context

### 2. Function Definition Information (`function.definition.*`)

**Purpose**: Shows where the function is defined (for tools).

| Attribute | Example | Description |
|-----------|---------|-------------|
| `function.definition.file` | `docs/examples/noveum_support_agent/tools.py` | File where function is defined |
| `function.definition.start_line` | `37` | First line of function definition |
| `function.definition.end_line` | `50` | Last line of function definition |

**How it works**:
1. Gets the function object from the tool
2. Uses `inspect.getsourcelines()` to get source lines
3. Calculates start and end line numbers
4. Makes file path relative to project root

### 3. Project Root Detection

**Purpose**: Makes paths relative and portable.

**How it works**:
```
Absolute path: /Users/you/project/docs/examples/noveum_support_agent/main.py
                ↓
Walk up directory tree
                ↓
Find first directory NOT in library location
                ↓
Project root: /Users/you/project
                ↓
Relative path: docs/examples/noveum_support_agent/main.py
```

**Library locations** (skipped):
- `site-packages/`
- `venv/`, `.venv/`
- `env/`, `.env/`
- `dist-packages/`
- `lib/python3.x/`

**User code locations** (captured):
- Your project directory
- Any directory not in library locations

## Real-World Example

### Scenario: Customer Support Agent

```python
# main.py
def answer_question(self, question: str):
    # Line 280: Agent invokes tools
    result = agent.invoke({"input": question})
    return result

# tools.py
@tool
def rag_search_tool(query: str) -> str:
    # Tool implementation
    return search_rag(query)
```

### What You See in Noveum Dashboard

**LLM Span**:
```
Span: llm.gpt-4
├─ code.file: docs/examples/noveum_support_agent/main.py
├─ code.line: 330
├─ code.function: answer_question
├─ code.module: __main__
├─ code.context: response = self.llm.invoke(prompt)
└─ llm.model: gpt-4
```

**Tool Span**:
```
Span: tool.rag_search_tool
├─ code.file: docs/examples/noveum_support_agent/main.py
├─ code.line: 280
├─ code.function: answer_question
├─ code.context: result = agent.invoke({"input": question})
├─ function.definition.file: docs/examples/noveum_support_agent/tools.py
├─ function.definition.start_line: 37
├─ function.definition.end_line: 50
└─ tool.name: rag_search_tool
```

## Benefits

1. **Quick Debugging**: Click on `code.file` and `code.line` to jump to exact location
2. **Performance Analysis**: See which functions make most calls
3. **Code Navigation**: Find where functions are defined and called
4. **Production-Friendly**: Relative paths work across environments
5. **Complete Visibility**: See both call site and definition location

## Summary

The tracing system captures:
- ✅ **Where** the call was made (`code.*`)
- ✅ **What** function made the call (`code.function`)
- ✅ **Which** line of code (`code.line`)
- ✅ **What** code was executed (`code.context`)
- ✅ **Where** the function is defined (`function.definition.*`)

All with relative paths that work in production! 🎉

