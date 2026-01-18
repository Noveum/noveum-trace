"""
DROP-IN DIAGNOSTIC CODE

Add this to your actual application to debug why callbacks aren't firing.
"""

import threading
from langchain_core.callbacks import BaseCallbackHandler


class DiagnosticCallbackHandler(BaseCallbackHandler):
    """
    Add this handler alongside your NoveumTraceCallbackHandler to see
    if callbacks are being triggered at all.
    
    Usage:
        from debug_your_code import DiagnosticCallbackHandler
        
        diagnostic = DiagnosticCallbackHandler()
        noveum_handler = NoveumTraceCallbackHandler()
        
        result = graph.invoke(
            {"input": "..."},
            config={"callbacks": [noveum_handler, diagnostic]}
        )
        
        diagnostic.print_report()
    """
    
    def __init__(self):
        super().__init__()
        self.events = []
        self._lock = threading.Lock()
    
    def _record(self, event_name, **kwargs):
        with self._lock:
            self.events.append({
                "event": event_name,
                "thread": threading.current_thread().name,
                "run_id": str(kwargs.get("run_id", ""))[:8],
                "parent_run_id": str(kwargs.get("parent_run_id", ""))[:8] if kwargs.get("parent_run_id") else None,
            })
            print(f"📍 {event_name} | Thread: {threading.current_thread().name}")
    
    # LLM events
    def on_llm_start(self, serialized, prompts, **kwargs):
        self._record("on_llm_start", **kwargs)
    
    def on_llm_end(self, response, **kwargs):
        self._record("on_llm_end", **kwargs)
    
    def on_llm_error(self, error, **kwargs):
        self._record("on_llm_error", **kwargs)
    
    def on_llm_new_token(self, token, **kwargs):
        # Don't log every token, just count them
        pass
    
    # Chat model events
    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._record("on_chat_model_start", **kwargs)
    
    # Chain events
    def on_chain_start(self, serialized, inputs, **kwargs):
        name = serialized.get("name", "?") if serialized else "?"
        self._record(f"on_chain_start ({name})", **kwargs)
    
    def on_chain_end(self, outputs, **kwargs):
        self._record("on_chain_end", **kwargs)
    
    def on_chain_error(self, error, **kwargs):
        self._record("on_chain_error", **kwargs)
    
    # Tool events
    def on_tool_start(self, serialized, input_str, **kwargs):
        self._record("on_tool_start", **kwargs)
    
    def on_tool_end(self, output, **kwargs):
        self._record("on_tool_end", **kwargs)
    
    # Agent events
    def on_agent_action(self, action, **kwargs):
        self._record("on_agent_action", **kwargs)
    
    def on_agent_finish(self, finish, **kwargs):
        self._record("on_agent_finish", **kwargs)
    
    def print_report(self):
        print("\n" + "=" * 60)
        print("CALLBACK DIAGNOSTIC REPORT")
        print("=" * 60)
        
        if not self.events:
            print("❌ NO CALLBACKS WERE FIRED AT ALL!")
            print("\nPossible causes:")
            print("1. callbacks not passed to invoke/ainvoke")
            print("2. LLM not using the callback manager")
            print("3. Exception occurred before callbacks fired")
            return
        
        llm_events = [e for e in self.events if "llm" in e["event"].lower() or "chat" in e["event"].lower()]
        chain_events = [e for e in self.events if "chain" in e["event"].lower()]
        
        print(f"Total events: {len(self.events)}")
        print(f"LLM/Chat events: {len(llm_events)}")
        print(f"Chain events: {len(chain_events)}")
        
        print("\nAll events:")
        for i, e in enumerate(self.events, 1):
            print(f"  {i}. {e['event']:35} | {e['thread']:25}")
        
        if not llm_events:
            print("\n⚠️  WARNING: No LLM callbacks fired!")
            print("   This means the LLM invoke() is not triggering callbacks.")
            print("   Check if config is being passed to llm.invoke()")


# Quick test
if __name__ == "__main__":
    print("Testing DiagnosticCallbackHandler...")
    
    try:
        from langchain_community.llms.fake import FakeListLLM
        
        diag = DiagnosticCallbackHandler()
        llm = FakeListLLM(responses=["test"])
        
        # With callbacks
        result = llm.invoke("test", config={"callbacks": [diag]})
        diag.print_report()
        
    except ImportError:
        print("LangChain not installed")
