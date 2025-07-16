"""
Chatbot Example with Noveum Trace SDK

This example demonstrates how to trace a conversational AI chatbot
with session management, context tracking, and conversation flow analysis.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

import noveum_trace


class TracedChatbot:
    """A simple chatbot with comprehensive tracing."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

        # Initialize tracing
        noveum_trace.init(
            service_name="ai-chatbot",
            environment="demo",
            log_directory="./chatbot_traces",
        )

    def start_session(self, user_id: Optional[str] = None) -> str:
        """Start a new chat session."""
        session_id = str(uuid.uuid4())
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"

        with noveum_trace.get_tracer().start_span("session_start") as span:
            span.set_attribute("session.id", session_id)
            span.set_attribute("user.id", user_id)
            span.set_attribute("chatbot.model", self.model)

            self.sessions[session_id] = []

            span.add_event(
                "session_created",
                {
                    "session.id": session_id,
                    "user.id": user_id,
                    "timestamp": time.time(),
                },
            )

        return session_id

    def chat(self, session_id: str, message: str) -> str:
        """Process a chat message and return response."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        with noveum_trace.get_tracer().start_span("chat_turn") as turn_span:
            turn_span.set_attribute("session.id", session_id)
            turn_span.set_attribute("turn.number", len(self.sessions[session_id]) + 1)
            turn_span.set_attribute("input.length", len(message))

            # Add user message to session
            self.sessions[session_id].append(
                {"role": "user", "content": message, "timestamp": time.time()}
            )

            # Process message with context
            with noveum_trace.get_tracer().start_span(
                "context_preparation"
            ) as context_span:
                context = self._prepare_context(session_id)
                context_span.set_attribute("context.messages_count", len(context))
                context_span.set_attribute(
                    "context.total_tokens", self._estimate_tokens(context)
                )

            # Generate response (simulated LLM call)
            with noveum_trace.get_tracer().start_span("llm_generation") as llm_span:
                llm_span.set_attribute("gen_ai.system", "openai")
                llm_span.set_attribute("gen_ai.request.model", self.model)
                llm_span.set_attribute("gen_ai.operation.name", "chat")

                # Add input event
                llm_span.add_event(
                    "gen_ai.content.prompt",
                    {"gen_ai.prompt": self._format_prompt(context)},
                )

                # Simulate LLM processing time
                processing_time = 0.5 + (len(message) * 0.01)
                time.sleep(processing_time)

                # Generate response based on message content
                response = self._generate_response(message)

                # Add output event
                llm_span.add_event(
                    "gen_ai.content.completion", {"gen_ai.completion": response}
                )

                # Set usage metrics
                input_tokens = self._estimate_tokens([{"content": message}])
                output_tokens = self._estimate_tokens([{"content": response}])

                llm_span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                llm_span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                llm_span.set_attribute(
                    "gen_ai.usage.total_tokens", input_tokens + output_tokens
                )
                llm_span.set_attribute("llm.latency_ms", processing_time * 1000)

            # Add response to session
            self.sessions[session_id].append(
                {"role": "assistant", "content": response, "timestamp": time.time()}
            )

            # Set turn-level attributes
            turn_span.set_attribute("output.length", len(response))
            turn_span.set_attribute("turn.total_tokens", input_tokens + output_tokens)
            turn_span.set_attribute("turn.processing_time_ms", processing_time * 1000)

            return response

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a chat session and return summary."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        with noveum_trace.get_tracer().start_span("session_end") as span:
            span.set_attribute("session.id", session_id)

            session_data = self.sessions[session_id]

            # Calculate session metrics
            total_turns = len([msg for msg in session_data if msg["role"] == "user"])
            total_tokens = sum(self._estimate_tokens([msg]) for msg in session_data)
            session_duration = (
                session_data[-1]["timestamp"] - session_data[0]["timestamp"]
                if session_data
                else 0
            )

            span.set_attribute("session.total_turns", total_turns)
            span.set_attribute("session.total_tokens", total_tokens)
            span.set_attribute("session.duration_seconds", session_duration)

            # Create session summary
            summary = {
                "session_id": session_id,
                "total_turns": total_turns,
                "total_tokens": total_tokens,
                "duration_seconds": session_duration,
                "messages": session_data,
            }

            span.add_event("session_summary", summary)

            # Clean up session
            del self.sessions[session_id]

            return summary

    def _prepare_context(self, session_id: str) -> List[Dict[str, str]]:
        """Prepare conversation context for LLM."""
        return self.sessions[session_id][-10:]  # Last 10 messages

    def _format_prompt(self, context: List[Dict[str, str]]) -> str:
        """Format context into a prompt string."""
        prompt_parts = []
        for msg in context:
            role = msg["role"].title()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)

    def _generate_response(self, message: str) -> str:
        """Generate a response based on the input message (simulated)."""
        message_lower = message.lower()

        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! How can I help you today?"
        elif "weather" in message_lower:
            return "I don't have access to real-time weather data, but I'd recommend checking a weather app or website for current conditions."
        elif "time" in message_lower:
            return f"The current time is {time.strftime('%H:%M:%S')}."
        elif "help" in message_lower:
            return "I'm here to help! You can ask me questions about various topics, and I'll do my best to provide useful information."
        elif "bye" in message_lower or "goodbye" in message_lower:
            return "Goodbye! It was nice chatting with you. Have a great day!"
        else:
            return f"That's an interesting point about '{message}'. Could you tell me more about what specifically you'd like to know?"

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count for messages (simplified)."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return max(1, total_chars // 4)  # Rough approximation: 4 chars per token


def main():
    """Demonstrate chatbot tracing."""
    print("🤖 Traced Chatbot Demo")
    print("=" * 40)

    # Create chatbot
    chatbot = TracedChatbot()

    # Start a session
    session_id = chatbot.start_session("demo_user")
    print(f"Started session: {session_id}")

    # Simulate conversation
    conversation = [
        "Hello there!",
        "What's the weather like?",
        "Can you help me with Python programming?",
        "What time is it?",
        "Thanks for your help. Goodbye!",
    ]

    print("\n💬 Conversation:")
    for i, user_message in enumerate(conversation, 1):
        print(f"\nTurn {i}:")
        print(f"User: {user_message}")

        response = chatbot.chat(session_id, user_message)
        print(f"Bot: {response}")

        # Small delay between turns
        time.sleep(0.1)

    # End session and get summary
    summary = chatbot.end_session(session_id)

    print("\n📊 Session Summary:")
    print(f"Total turns: {summary['total_turns']}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Duration: {summary['duration_seconds']:.1f} seconds")

    # Flush traces
    noveum_trace.flush()
    print("\n✅ Traces saved to ./chatbot_traces/")

    # Shutdown
    noveum_trace.shutdown()

    print("\n🎉 Chatbot demo completed!")
    print("Check the trace files to see detailed conversation analytics.")


if __name__ == "__main__":
    main()
