"""
Langchain Integration Example for Noveum Trace SDK.

This example demonstrates how to trace Langchain workflows including:
- LLM chains and prompts
- Document processing
- Agent-based workflows with tools
- Memory and conversation tracking
"""

import os
import time
from typing import Any

# Load environment variables (install python-dotenv if needed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "python-dotenv not installed. Environment variables will be read from system only."
    )
    pass

import noveum_trace


class MockDocument:
    """Mock document class for demonstration."""

    def __init__(self, page_content: str, metadata: dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class TracedLangchainWorkflow:
    """Example class demonstrating Langchain integration with Noveum Trace."""

    def __init__(self):
        # Sample documents for RAG
        self.documents = self._create_sample_documents()
        self.conversation_history = []
        self.vector_store = None

    def _create_sample_documents(self) -> list[MockDocument]:
        """Create sample documents for demonstration."""
        sample_texts = [
            "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence.",
            "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language.",
            "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.",
            "Computer Vision is a field of AI that trains computers to interpret and understand visual information from the world around them.",
        ]

        documents = []
        for i, text in enumerate(sample_texts):
            doc = MockDocument(
                page_content=text, metadata={"source": f"document_{i}", "topic": "AI"}
            )
            documents.append(doc)

        return documents

    @noveum_trace.trace
    def setup_vector_store(self) -> None:
        """Set up vector store for document retrieval."""

        with noveum_trace.trace_context(name="text_splitting"):
            # Mock text splitting
            texts = self._mock_text_splitting(self.documents)

        with noveum_trace.trace_context(name="embedding_creation"):
            # Mock the embedding creation
            self.vector_store = self._mock_vector_store(texts)

    def _mock_text_splitting(self, documents: list[MockDocument]) -> list[MockDocument]:
        """Mock text splitting for demonstration."""
        time.sleep(0.1)  # Simulate processing time
        return documents  # Return as-is for simplicity

    def _mock_vector_store(self, texts: list[MockDocument]) -> dict[str, Any]:
        """Mock vector store for demonstration."""
        return {
            "documents": texts,
            "embeddings": ["mock_embedding"] * len(texts),
            "index": "mock_faiss_index",
        }

    @noveum_trace.trace_llm
    def simple_llm_chain(self, question: str) -> str:
        """Demonstrate a simple LLM chain."""

        with noveum_trace.trace_context(name="prompt_formatting"):
            # Mock prompt template
            formatted_prompt = f"You are a helpful AI assistant. Answer the following question: {question}"

        with noveum_trace.trace_context(name="llm_call"):
            # Mock LLM response since we don't have real OpenAI API
            response = self._mock_llm_response(formatted_prompt)

        return response

    def _mock_llm_response(self, prompt: str) -> str:
        """Mock LLM response for demonstration."""
        time.sleep(0.2)  # Simulate API call delay

        if "AI" in prompt or "artificial intelligence" in prompt.lower():
            return "Artificial Intelligence is a fascinating field that involves creating intelligent machines capable of performing tasks that typically require human intelligence."
        elif "machine learning" in prompt.lower():
            return "Machine Learning is a powerful subset of AI that enables systems to automatically learn and improve from experience."
        else:
            return (
                "I'm here to help answer your questions about AI and technology topics."
            )

    @noveum_trace.trace_retrieval
    def retrieval_qa_chain(self, question: str) -> dict[str, Any]:
        """Demonstrate retrieval-augmented generation (RAG)."""

        if not self.vector_store:
            self.setup_vector_store()

        with noveum_trace.trace_context(name="document_retrieval"):
            relevant_docs = self._retrieve_documents(question)

        with noveum_trace.trace_context(name="context_preparation"):
            context = self._prepare_context(relevant_docs)

        with noveum_trace.trace_context(name="answer_generation"):
            answer = self._generate_answer_with_context(question, context)

        return {
            "question": question,
            "answer": answer,
            "source_documents": relevant_docs,
            "context_length": len(context),
        }

    def _retrieve_documents(self, query: str) -> list[MockDocument]:
        """Mock document retrieval."""
        time.sleep(0.1)  # Simulate retrieval delay

        # Simple keyword-based retrieval for demo
        relevant_docs = []
        for doc in self.documents:
            if any(
                keyword.lower() in doc.page_content.lower() for keyword in query.split()
            ):
                relevant_docs.append(doc)

        return relevant_docs[:3]  # Return top 3 documents

    def _prepare_context(self, documents: list[MockDocument]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i + 1}: {doc.page_content}")

        return "\n\n".join(context_parts)

    def _generate_answer_with_context(self, question: str, context: str) -> str:
        """Generate answer using context."""
        time.sleep(0.3)  # Simulate generation delay

        # Mock answer generation
        if "machine learning" in question.lower():
            return "Based on the provided context, Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It's a powerful approach that allows systems to automatically improve their performance on a specific task."
        elif "AI" in question or "artificial intelligence" in question.lower():
            return "According to the context, Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. It encompasses various subfields including machine learning, natural language processing, and computer vision."
        else:
            return f"Based on the available context, I can provide information about AI-related topics. The context contains information about: {context[:100]}..."

    @noveum_trace.trace_agent(agent_id="langchain_agent")
    def agent_workflow(self, user_input: str) -> dict[str, Any]:
        """Demonstrate an agent-based workflow with tools."""

        # Define tools for the agent
        tools = self._create_agent_tools()

        with noveum_trace.trace_context(name="agent_planning"):
            plan = self._create_agent_plan(user_input, tools)

        with noveum_trace.trace_context(name="tool_execution"):
            results = self._execute_agent_plan(plan, tools)

        with noveum_trace.trace_context(name="response_synthesis"):
            final_response = self._synthesize_agent_response(user_input, results)

        return {
            "user_input": user_input,
            "plan": plan,
            "tool_results": results,
            "final_response": final_response,
            "tools_used": [tool["name"] for tool in plan],
        }

    def _create_agent_tools(self) -> list[dict[str, Any]]:
        """Create tools for the agent."""
        tools = [
            {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base for information about AI topics",
                "function": self._search_knowledge_base,
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "function": self._calculate,
            },
            {
                "name": "get_current_time",
                "description": "Get the current time",
                "function": self._get_current_time,
            },
        ]
        return tools

    def _create_agent_plan(
        self, user_input: str, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create an execution plan for the agent."""
        plan = []

        if "time" in user_input.lower():
            plan.append({"name": "get_current_time", "args": {}})

        if any(
            keyword in user_input.lower()
            for keyword in ["AI", "machine learning", "artificial intelligence"]
        ):
            plan.append(
                {"name": "search_knowledge_base", "args": {"query": user_input}}
            )

        if any(
            keyword in user_input.lower()
            for keyword in ["calculate", "math", "compute"]
        ):
            plan.append({"name": "calculate", "args": {"expression": user_input}})

        if not plan:
            plan.append(
                {"name": "search_knowledge_base", "args": {"query": user_input}}
            )

        return plan

    def _execute_agent_plan(
        self, plan: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute the agent's plan."""
        results = []

        tool_map = {tool["name"]: tool["function"] for tool in tools}

        for step in plan:
            tool_name = step["name"]
            tool_args = step["args"]

            if tool_name in tool_map:
                result = tool_map[tool_name](**tool_args)
                results.append({"tool": tool_name, "args": tool_args, "result": result})

        return results

    @noveum_trace.trace_tool
    def _search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base."""
        time.sleep(0.1)

        for doc in self.documents:
            if any(
                keyword.lower() in doc.page_content.lower() for keyword in query.split()
            ):
                return doc.page_content

        return "No relevant information found in the knowledge base."

    @noveum_trace.trace_tool
    def _calculate(self, expression: str) -> str:
        """Perform calculations."""
        time.sleep(0.05)

        # Simple calculation for demo
        try:
            # Extract numbers for basic operations
            if "+" in expression:
                parts = expression.split("+")
                if len(parts) == 2:
                    result = float(parts[0].strip()) + float(parts[1].strip())
                    return f"The result of {expression} is {result}"
            elif "*" in expression:
                parts = expression.split("*")
                if len(parts) == 2:
                    result = float(parts[0].strip()) * float(parts[1].strip())
                    return f"The result of {expression} is {result}"
        except (ValueError, IndexError, ZeroDivisionError):
            pass

        return "I can perform basic arithmetic operations like addition and multiplication."

    @noveum_trace.trace_tool
    def _get_current_time(self) -> str:
        """Get current time."""
        import datetime

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"The current time is {current_time}"

    def _synthesize_agent_response(
        self, user_input: str, results: list[dict[str, Any]]
    ) -> str:
        """Synthesize final response from tool results."""
        if not results:
            return "I couldn't find any relevant information to answer your question."

        response_parts = []
        for result in results:
            response_parts.append(f"Using {result['tool']}: {result['result']}")

        return " ".join(response_parts)

    @noveum_trace.trace
    def conversation_with_memory(self, messages: list[str]) -> list[str]:
        """Demonstrate conversation with memory."""
        responses = []

        for i, message in enumerate(messages):
            with noveum_trace.trace_context(name=f"conversation_turn_{i + 1}"):
                # Add to memory
                self.conversation_history.append({"role": "user", "content": message})

                # Generate response
                conversation_context = self._get_conversation_context()
                response = self._generate_contextual_response(
                    message, conversation_context
                )

                # Add response to memory
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )

                responses.append(response)

        return responses

    def _get_conversation_context(self) -> str:
        """Get conversation context from history."""
        context_parts = []
        for entry in self.conversation_history[-6:]:  # Last 6 messages
            role = entry["role"]
            content = entry["content"]
            context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)

    def _generate_contextual_response(
        self, message: str, conversation_history: str
    ) -> str:
        """Generate response considering conversation history."""
        time.sleep(0.2)

        # Mock contextual response
        if "hello" in message.lower():
            return "Hello! I'm here to help you with AI and technology questions."
        elif "thank" in message.lower():
            return "You're welcome! Is there anything else you'd like to know?"
        elif conversation_history and "previous" in message.lower():
            return "Based on our previous conversation, I can provide more details on that topic."
        else:
            return f"I understand you're asking about: {message}. Let me help you with that."


def main():
    """Main function demonstrating Langchain integration."""

    # Initialize Noveum Trace SDK
    noveum_trace.init(
        api_key=os.getenv("NOVEUM_API_KEY"),
        project="langchain_integration_demo",
        environment="development",
    )

    print("🔗 Starting Langchain Integration Demo")
    print("=" * 50)

    # Create workflow instance
    workflow = TracedLangchainWorkflow()

    # Demo 1: Simple LLM Chain
    print("\n1️⃣ Simple LLM Chain Demo")
    with noveum_trace.trace_context(name="simple_llm_demo"):
        question = "What is artificial intelligence?"
        answer = workflow.simple_llm_chain(question)
        print(f"Q: {question}")
        print(f"A: {answer}")

    # Demo 2: Retrieval QA
    print("\n2️⃣ Retrieval QA Demo")
    with noveum_trace.trace_context(name="retrieval_qa_demo"):
        rag_question = "Tell me about machine learning"
        rag_result = workflow.retrieval_qa_chain(rag_question)
        print(f"Q: {rag_result['question']}")
        print(f"A: {rag_result['answer']}")
        print(f"Sources: {len(rag_result['source_documents'])} documents")

    # Demo 3: Agent Workflow
    print("\n3️⃣ Agent Workflow Demo")
    with noveum_trace.trace_context(name="agent_workflow_demo"):
        agent_input = "What time is it and tell me about AI?"
        agent_result = workflow.agent_workflow(agent_input)
        print(f"Input: {agent_result['user_input']}")
        print(f"Tools Used: {', '.join(agent_result['tools_used'])}")
        print(f"Response: {agent_result['final_response']}")

    # Demo 4: Conversation with Memory
    print("\n4️⃣ Conversation with Memory Demo")
    with noveum_trace.trace_context(name="conversation_demo"):
        conversation = [
            "Hello, I'm interested in learning about AI",
            "Can you tell me more about machine learning?",
            "Thank you for the explanation",
        ]
        responses = workflow.conversation_with_memory(conversation)

        for i, (msg, resp) in enumerate(zip(conversation, responses)):
            print(f"Turn {i + 1}:")
            print(f"  User: {msg}")
            print(f"  AI: {resp}")

    # Flush traces
    client = noveum_trace.get_client()
    client.flush()

    print("\n✅ Langchain integration demo completed!")
    print(
        "🔍 Check your Noveum dashboard to view the complete Langchain workflow traces."
    )
    print(
        "📊 Traces include: LLM calls, retrieval operations, agent actions, and conversation flows."
    )


if __name__ == "__main__":
    main()
