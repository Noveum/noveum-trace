# LangChain Agent with Enhanced Memory

A LangChain agent with document retrieval capabilities, advanced memory management, custom tools, and comprehensive tracing.

## Features

- **Document Ingestion**: Automatically loads PDF and TXT files from the `docs/` directory
- **Vector Search**: Uses ChromaDB for semantic search over documents
- **Enhanced Memory System**: 
  - Conversation state tracking
  - Document context memory
  - Search history and topic extraction
  - Context summarization
  - Memory persistence (save/load)
- **Custom Tools**: 
  - `next_turn`: Passes control to user for additional input
  - `escalate_to_human`: Escalates complex queries to human support
  - `end_conversation`: Ends the conversation when query is resolved
  - `langchain_retriever`: Searches the knowledge base
- **Memory Commands**:
  - `memory` or `mem`: Show memory summary + LangChain built-in memory
  - `clear` or `reset`: Clear conversation memory
  - `save`: Save conversation context to file
  - `load`: Load conversation context from file
- **Tracing**: Comprehensive tracing with Noveum Trace SDK

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export NOVEUM_API_KEY="your-noveum-api-key"  # Optional
```

3. Add documents to the `docs/` directory (PDF or TXT files)

## Usage

### As a script:
```bash
python agent.py
```

### As a module:
```python
from langchain_agent.agent import LangChainAgent

# Initialize agent
agent = LangChainAgent()

# Start conversation
agent.start()
```

## Example Conversation

```
🤖 LangChain Agent with Enhanced Memory Started!
============================================================
Available tools:
  - next_turn: Use this tool when you need the user to provide more information
  - escalate_to_human: Use this tool when the query is too complex
  - end_conversation: Use this tool when the query is resolved
  - langchain_retriever: Use this tool to search through the knowledge base
  - memory_access: Use this tool to access conversation memory and context
============================================================
Special commands:
  - 'memory' or 'mem': Show memory summary
  - 'clear' or 'reset': Clear conversation memory
  - 'save': Save conversation context to file
  - 'load': Load conversation context from file
  - 'quit' or 'exit': End the conversation
============================================================

You: What are the key features of this system?
Agent: Let me search the knowledge base for information about the key features...

[Agent uses langchain_retriever tool]

Agent: Based on the documentation, the key features include:
1. Document ingestion from PDF and TXT files
2. ChromaDB vector store for semantic search
3. Custom tools for conversation control
4. Enhanced memory system with state tracking
5. Comprehensive tracing with Noveum Trace

You: memory
Memory Summary:
- Turn Count: 2
- Last Tool Used: langchain_retriever
- Escalated: False
- Conversation Ended: False
- Recent Topics: document, ingestion, vector, search, tools, memory, tracing
- Search History: 1 searches
- Context: Turn 1: What are the key features of this system?...

You: Can you explain how the retrieval works?
Agent: [Uses retrieval tool again and provides detailed explanation]

You: save
Conversation context saved!

You: This is too complex, I need human help
Agent: [Uses escalate_to_human tool]
ESCALATED TO HUMAN SUPPORT. Please wait for a human customer support representative to get back to you.
```

## Architecture

The agent uses LangChain's standard ReAct (Reasoning and Acting) approach:
1. **Understanding**: LangChain agent analyzes user input
2. **Tool Selection**: Agent automatically decides which tools to use
3. **Tool Execution**: LangChain handles tool calling automatically
4. **Reasoning**: Agent processes tool results
5. **Response**: Agent provides final answer or takes further action

The agent leverages LangChain's built-in memory system and automatic tool calling, making it much simpler and more reliable than manual tool detection.

## Tracing

All operations are automatically traced with Noveum Trace:
- LLM calls and responses
- Tool executions
- Retrieval operations
- Agent reasoning steps
- Memory operations

Check your Noveum dashboard to see detailed traces of all agent operations.
