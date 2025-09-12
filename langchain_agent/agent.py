"""
LangChain Agent with Retrieval and Custom Tools

This agent provides:
- Document ingestion from docs/ directory (PDF and TXT files)
- ChromaDB vector store for retrieval
- Custom tools: next_turn, escalate_to_human, end_convo
- Conversation memory between turns
- Comprehensive tracing with Noveum Trace
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for noveum_trace imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import noveum_trace
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler
except ImportError:
    print("Warning: noveum_trace not available. Running without tracing.")
    noveum_trace = None
    NoveumTraceCallbackHandler = None

try:
    from langchain.agents import AgentType, initialize_agent, Tool
    from langchain.memory import ConversationBufferMemory
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.tools import BaseTool
except ImportError as e:
    print(f"Error importing LangChain dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EscalateToHumanTool(BaseTool):
    """Tool to escalate the conversation to human customer support."""

    name: str = "escalate_to_human"
    description: str = "Use this tool when the query is too complex or you cannot resolve the user's issue. This will escalate to human customer support."

    def _run(self, reason: str = "Complex query requiring human assistance") -> str:
        """Execute the escalation tool."""
        return f"ESCALATED TO HUMAN SUPPORT. Reason: {reason}. Please wait for a human customer support representative to get back to you."

    async def _arun(self, reason: str = "Complex query requiring human assistance") -> str:
        """Async version of the tool."""
        return self._run(reason)


# class EndConversationTool(BaseTool):
#     """Tool to end the conversation when the query is resolved."""

#     name: str = "end_conversation"
#     description: str = "Use this tool when the user's query has been fully resolved and the conversation can be ended. Do not use this tool until the user hints that the conversation should end."

#     def _run(self, summary: str = "Query resolved successfully") -> str:
#         """Execute the end conversation tool."""
#         return f"CONVERSATION ENDED. Summary: {summary}"

#     async def _arun(self, summary: str = "Query resolved successfully") -> str:
#         """Async version of the tool."""
#         return self._run(summary)


class UserInputTool(BaseTool):
    """Tool to get additional input from the user."""

    name: str = "user_input"
    description: str = "Use this tool when you need additional information or clarification from the user. This will prompt the user for more details."

    def _run(self, prompt: str = "Please provide more information") -> str:
        """Execute the user input tool."""
        user_response = input(f"Agent: {prompt}\nYou: ")
        return f"User response: {user_response}"

    async def _arun(self, prompt: str = "Please provide more information") -> str:
        """Async version of the tool."""
        return self._run(prompt)


class LangChainAgent:
    """LangChain Agent with retrieval capabilities and custom tools."""

    def __init__(self,
                 docs_dir: str = "docs",
                 openai_api_key: Optional[str] = None,
                 noveum_api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo"):
        """Initialize the agent."""
        # Use absolute path to avoid working directory issues
        if not os.path.isabs(docs_dir):
            # If relative path, make it relative to the script's directory
            script_dir = Path(__file__).parent
            self.docs_dir = script_dir / docs_dir
        else:
            self.docs_dir = Path(docs_dir)
        self.model_name = model_name
        self.vectorstore = None
        self.retriever = None
        self.agent = None
        self.memory = None
        self.callback_handler = None

        # Initialize tracing
        self._setup_tracing(noveum_api_key)

        # Initialize LLM
        self._setup_llm(openai_api_key)

        # Initialize memory
        self._setup_memory()

        # Load and process documents
        self._load_documents()

        # Setup retrieval
        self._setup_retrieval()

        # Create custom tools
        self._create_tools()

        # Initialize agent
        self._create_agent()

    def _setup_tracing(self, noveum_api_key: Optional[str] = None):
        """Setup Noveum Trace for comprehensive tracing."""
        if noveum_trace and NoveumTraceCallbackHandler:
            try:
                # Initialize noveum trace
                noveum_trace.init(
                    api_key=noveum_api_key or os.getenv("NOVEUM_API_KEY"),
                    transport_config={"batch_size": 1, "batch_timeout": 5.0},
                    endpoint="https://noveum.free.beeceptor.com"
                )

                # Create callback handler
                self.callback_handler = NoveumTraceCallbackHandler()
                logger.info("Noveum Trace initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize Noveum Trace: %s", e)
                self.callback_handler = None
        else:
            logger.warning("Noveum Trace not available")

    def _setup_llm(self, openai_api_key: Optional[str] = None):
        """Setup the OpenAI LLM."""
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it to the constructor.")

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0.7,
            api_key=api_key,
            callbacks=[self.callback_handler] if self.callback_handler else []
        )
        logger.info("LLM initialized: %s", self.model_name)

    def _setup_memory(self):
        """Setup conversation memory with enhanced features."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # Additional memory for agent state
        self.conversation_state = {
            "turn_count": 0,
            "escalated": False,
            "conversation_ended": False,
            "last_tool_used": None,
            "user_preferences": {},
            "context_summary": ""
        }

        # Memory for document context
        self.document_context = {
            "last_retrieved_docs": [],
            "relevant_topics": set(),
            "search_history": []
        }

        logger.info("Enhanced conversation memory initialized")

    def _load_documents(self):
        """Load and process documents from the docs directory."""
        if not self.docs_dir.exists():
            logger.warning(
                "Docs directory %s does not exist. Creating it.", self.docs_dir)
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            return

        documents = []

        # Load PDF files
        for pdf_file in self.docs_dir.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                documents.extend(docs)
                logger.info("Loaded PDF: %s", pdf_file.name)
            except Exception as e:
                logger.error("Error loading PDF %s: %s", pdf_file, e)

        # Load text files
        for txt_file in self.docs_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(txt_file))
                docs = loader.load()
                documents.extend(docs)
                logger.info("Loaded TXT: %s", txt_file.name)
            except Exception as e:
                logger.error("Error loading TXT %s: %s", txt_file, e)

        if not documents:
            logger.warning("No documents found in docs directory")
            return

        logger.info(f"Found {len(documents)} documents to process")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.documents = text_splitter.split_documents(documents)
        logger.info("Processed %d document chunks", len(self.documents))

    def _setup_retrieval(self):
        """Setup ChromaDB vector store and retriever."""
        if not hasattr(self, 'documents') or not self.documents:
            logger.warning("No documents available for retrieval setup")
            return

        logger.info(
            f"Setting up retrieval with {len(self.documents)} document chunks")

        try:
            # Clear existing ChromaDB for fresh start (demo mode)
            chroma_dir = Path("./chroma_db")
            if chroma_dir.exists():
                import shutil
                shutil.rmtree(chroma_dir)
                logger.info(
                    "Cleared existing ChromaDB directory for fresh start")

            # Create embeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("OpenAI embeddings created successfully")

            # Create ChromaDB vector store
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            logger.info("ChromaDB vector store created successfully")

            # Create retriever using modern LangChain interface
            # This follows the pattern: vectorstore.as_retriever() -> retriever.invoke(query)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            logger.info("Retriever created successfully")

            logger.info("ChromaDB vector store and retriever initialized")
        except Exception as e:
            logger.error("Error setting up retrieval: %s", e)
            self.retriever = None

    def _create_tools(self):
        """Create custom tools for the agent."""
        self.custom_tools = [
            EscalateToHumanTool(),
            UserInputTool()
        ]
        logger.info("Created %d custom tools", len(self.custom_tools))

    def _retrieval_func(self, query: str) -> str:
        """Function to perform retrieval with memory integration using modern LangChain interface."""
        try:
            # Store search query in memory
            self.document_context["search_history"].append(query)

            # Use the modern invoke method instead of get_relevant_documents
            docs = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."

            # Store retrieved docs in memory
            self.document_context["last_retrieved_docs"] = docs

            # Extract topics from retrieved documents
            for doc in docs:
                # Simple topic extraction (can be enhanced)
                words = doc.page_content.lower().split()
                for word in words:
                    if len(word) > 4 and word.isalpha():
                        self.document_context["relevant_topics"].add(word)

            result = "Retrieved information:\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"{i}. {doc.page_content}\n\n"

            return result
        except Exception as e:
            logger.error("Error in retrieval: %s", e)
            return f"Error during retrieval: {str(e)}"

    def _create_agent(self):
        """Create the LangChain agent."""
        try:
            # Create tools list with proper Tool objects
            tools = []

            # Add custom tools as Tool objects
            for tool in self.custom_tools:
                tools.append(Tool(
                    name=tool.name,
                    func=tool._run,
                    description=tool.description
                ))

            # Add retrieval tool if available
            if self.retriever:
                tools.append(Tool(
                    name="langchain_retriever",
                    func=self._retrieval_func,
                    description="Use this tool to search through the knowledge base for relevant information. Input should be a search query."
                ))
                logger.info("Added langchain_retriever tool to agent")
            else:
                logger.warning(
                    "No retriever available - langchain_retriever tool not added")

            logger.info(f"Total tools available: {len(tools)}")
            for tool in tools:
                logger.info(f"  - {tool.name}: {tool.description}")

            # Create agent exactly like the example
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                callbacks=[
                    self.callback_handler] if self.callback_handler else [],
                verbose=True,
                handle_parsing_errors=True
            )
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error("Error creating agent: %s", e)
            raise

    def start(self):
        """Start the agent conversation loop."""
        if not self.agent:
            raise RuntimeError("Agent not initialized properly")

        print("🤖 LangChain Agent with Enhanced Memory Started!")
        print("=" * 60)
        print("Available tools:")
        for tool in self.custom_tools:
            print(f"  - {tool.name}: {tool.description}")
        if self.retriever:
            print("  - langchain_retriever: Search through the knowledge base")
        print("=" * 60)
        print("Special commands:")
        print("  - 'memory' or 'mem': Show memory summary + LangChain memory")
        print("  - 'clear' or 'reset': Clear conversation memory")
        print("  - 'save': Save conversation context to file")
        print("  - 'load': Load conversation context from file")
        print("  - 'quit' or 'exit': End the conversation")
        print("=" * 60)
        print()

        # Main conversation loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! 👋")
                    # Save conversation context before exiting
                    self.save_conversation_context()
                    break

                if user_input.lower() in ['memory', 'mem']:
                    print(f"\n{self.get_memory_summary()}")
                    # Also show LangChain's built-in memory
                    if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
                        print("\nLangChain Memory (Recent Messages):")
                        for i, msg in enumerate(self.memory.chat_memory.messages[-5:], 1):
                            role = "Human" if msg.type == "human" else "AI"
                            content = msg.content[:100] + \
                                "..." if len(
                                    msg.content) > 100 else msg.content
                            print(f"  {i}. {role}: {content}")
                    continue

                if user_input.lower() in ['clear', 'reset']:
                    self.clear_memory()
                    print("Memory cleared! Starting fresh conversation.")
                    continue

                if user_input.lower() in ['save']:
                    self.save_conversation_context()
                    print("Conversation context saved!")
                    continue

                if user_input.lower() in ['load']:
                    self.load_conversation_context()
                    print("Conversation context loaded!")
                    continue

                if not user_input:
                    continue

                # Update conversation state
                self.update_conversation_state(user_input=user_input)

                # Run agent - let LangChain handle tool calling automatically
                response = self.agent.run(input=user_input)

                # Check if the response indicates conversation should end
                # if "conversation ended" in str(response).lower():
                #     print(f"\nAgent: {response}")
                #     print("Conversation ended. Goodbye! 👋")
                #     self.save_conversation_context()
                #     break

                # Check if escalated to human
                if "escalated to human" in str(response).lower():
                    print(f"\nAgent: {response}")
                    print("Conversation ended. Goodbye! 👋")
                    self.save_conversation_context()
                    break

                # Regular response
                print(f"\nAgent: {response}")
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                self.save_conversation_context()
                break
            except Exception as e:
                logger.error("Error in conversation loop: %s", e)
                print(f"Sorry, an error occurred: {e}")
                print("Please try again.")

    def get_agent(self):
        """Get the underlying LangChain agent for direct use."""
        return self.agent

    def update_conversation_state(self, tool_name: str = None, user_input: str = None):
        """Update conversation state based on current interaction."""
        self.conversation_state["turn_count"] += 1

        if tool_name:
            self.conversation_state["last_tool_used"] = tool_name

        if tool_name == "escalate_to_human":
            self.conversation_state["escalated"] = True
        elif tool_name == "end_conversation":
            self.conversation_state["conversation_ended"] = True

        # Update context summary
        if user_input:
            # Simple context summarization
            if len(self.conversation_state["context_summary"]) > 500:
                self.conversation_state["context_summary"] = self.conversation_state["context_summary"][-250:]

            self.conversation_state["context_summary"] += f" Turn {self.conversation_state['turn_count']}: {user_input[:100]}... "

    def get_memory_summary(self) -> str:
        """Get a summary of current memory state."""
        summary = f"""
Memory Summary:
- Turn Count: {self.conversation_state['turn_count']}
- Last Tool Used: {self.conversation_state['last_tool_used'] or 'None'}
- Escalated: {self.conversation_state['escalated']}
- Conversation Ended: {self.conversation_state['conversation_ended']}
- Recent Topics: {', '.join(list(self.document_context['relevant_topics'])[:10])}
- Search History: {len(self.document_context['search_history'])} searches
- Context: {self.conversation_state['context_summary'][:200]}...
        """
        return summary.strip()

    def clear_memory(self):
        """Clear all memory and reset conversation state."""
        self.memory.clear()
        self.conversation_state = {
            "turn_count": 0,
            "escalated": False,
            "conversation_ended": False,
            "last_tool_used": None,
            "user_preferences": {},
            "context_summary": ""
        }
        self.document_context = {
            "last_retrieved_docs": [],
            "relevant_topics": set(),
            "search_history": []
        }
        logger.info("Memory cleared and reset")

    def save_conversation_context(self, filepath: str = "conversation_context.json"):
        """Save conversation context to file."""
        import json

        context_data = {
            "conversation_state": self.conversation_state,
            "document_context": {
                "search_history": self.document_context["search_history"],
                "relevant_topics": list(self.document_context["relevant_topics"]),
                "last_retrieved_docs_count": len(self.document_context["last_retrieved_docs"])
            },
            "memory_messages": len(self.memory.chat_memory.messages) if hasattr(self.memory, 'chat_memory') else 0
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(context_data, f, indent=2)
            logger.info("Conversation context saved to %s", filepath)
        except Exception as e:
            logger.error("Error saving conversation context: %s", e)

    def load_conversation_context(self, filepath: str = "conversation_context.json"):
        """Load conversation context from file."""
        import json

        try:
            with open(filepath, 'r') as f:
                context_data = json.load(f)

            self.conversation_state = context_data.get(
                "conversation_state", self.conversation_state)

            doc_context = context_data.get("document_context", {})
            self.document_context["search_history"] = doc_context.get(
                "search_history", [])
            self.document_context["relevant_topics"] = set(
                doc_context.get("relevant_topics", []))

            logger.info("Conversation context loaded from %s", filepath)
        except Exception as e:
            logger.error("Error loading conversation context: %s", e)


def main():
    """Main function to run the agent."""
    try:
        # Initialize and start the agent
        agent = LangChainAgent()
        agent.start()
    except Exception as e:
        logger.error("Failed to start agent: %s", e)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
