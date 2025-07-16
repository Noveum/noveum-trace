#!/usr/bin/env python3
"""
Test LangChain integration with Noveum Trace.
"""
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables
from dotenv import load_dotenv

import noveum_trace

load_dotenv()


def test_langchain_openai():
    """Test LangChain with OpenAI integration."""
    print("🦜 Testing LangChain + OpenAI Integration...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        # Create OpenAI chat model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Test with messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of France?"),
        ]

        response = llm.invoke(messages)
        print(f"✅ LangChain OpenAI Response: {response.content}")

        # Test with simple invoke
        response2 = llm.invoke("Tell me a joke in one sentence.")
        print(f"✅ LangChain OpenAI Simple Response: {response2.content}")

    except ImportError as e:
        print(f"⚠️ LangChain OpenAI not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ LangChain OpenAI test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


def test_langchain_anthropic():
    """Test LangChain with Anthropic integration."""
    print("\n🤖 Testing LangChain + Anthropic Integration...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        # Create Anthropic chat model
        llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.7,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        # Test with messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the capital of Japan?"),
        ]

        response = llm.invoke(messages)
        print(f"✅ LangChain Anthropic Response: {response.content}")

        # Test with simple invoke
        response2 = llm.invoke("Tell me an interesting fact about space.")
        print(f"✅ LangChain Anthropic Simple Response: {response2.content}")

    except ImportError as e:
        print(f"⚠️ LangChain Anthropic not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ LangChain Anthropic test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


def test_langchain_chains():
    """Test LangChain chains."""
    print("\n🔗 Testing LangChain Chains...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        # Create OpenAI chat model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that explains concepts clearly.",
                ),
                ("user", "Explain {topic} in simple terms."),
            ]
        )

        # Create chain
        chain = prompt | llm | StrOutputParser()

        # Test the chain
        response = chain.invoke({"topic": "machine learning"})
        print(f"✅ LangChain Chain Response: {response[:200]}...")

    except ImportError as e:
        print(f"⚠️ LangChain components not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ LangChain chains test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


def test_langchain_agents():
    """Test LangChain agents."""
    print("\n🕵️ Testing LangChain Agents...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        # Create a simple tool
        @tool
        def get_current_weather(location: str) -> str:
            """Get the current weather for a given location."""
            return f"The weather in {location} is sunny with a temperature of 72°F."

        # Create OpenAI chat model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that can use tools to answer questions.",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create agent
        agent = create_tool_calling_agent(llm, [get_current_weather], prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[get_current_weather])

        # Test the agent
        response = agent_executor.invoke(
            {"input": "What's the weather like in San Francisco?"}
        )
        print(f"✅ LangChain Agent Response: {response.get('output', 'No output')}")

    except ImportError as e:
        print(f"⚠️ LangChain agents not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ LangChain agents test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


def test_langchain_retrieval():
    """Test LangChain retrieval (RAG)."""
    print("\n📚 Testing LangChain Retrieval (RAG)...")

    # Initialize tracer
    noveum_trace.init(
        project_id="test_project", file_logging=True, log_directory="test_traces"
    )

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # Create some sample documents
        docs = [
            Document(
                page_content="The capital of France is Paris.",
                metadata={"source": "geography"},
            ),
            Document(
                page_content="Python is a programming language.",
                metadata={"source": "programming"},
            ),
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"source": "ai"},
            ),
        ]

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # Create OpenAI chat model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Create RAG prompt
        prompt = ChatPromptTemplate.from_template(
            """
        Use the following context to answer the question:

        Context: {context}

        Question: {question}

        Answer:
        """
        )

        # Create RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Test the RAG chain
        response = rag_chain.invoke("What is the capital of France?")
        print(f"✅ LangChain RAG Response: {response}")

    except ImportError as e:
        print(f"⚠️ LangChain RAG components not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ LangChain RAG test failed: {e}")
        return False

    finally:
        noveum_trace.shutdown()

    return True


if __name__ == "__main__":
    print("🚀 Starting LangChain Integration Tests...")

    # Install required packages
    try:
        import subprocess

        packages = [
            "langchain",
            "langchain-openai",
            "langchain-anthropic",
            "langchain-community",
            "faiss-cpu",
        ]

        for package in packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                print(f"📦 Installing {package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package], check=True
                )
    except Exception as e:
        print(f"⚠️ Could not install packages: {e}")

    results = []
    results.append(test_langchain_openai())
    results.append(test_langchain_anthropic())
    results.append(test_langchain_chains())
    results.append(test_langchain_agents())
    results.append(test_langchain_retrieval())

    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    if all(results):
        print("🎉 All LangChain tests passed!")
    else:
        print("❌ Some LangChain tests failed")
        sys.exit(1)
