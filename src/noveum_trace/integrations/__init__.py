"""
Framework integrations for Noveum Trace SDK.

This package provides automatic instrumentation for popular
LLM frameworks and libraries.
"""

# Import available integrations
try:
    from noveum_trace.integrations.openai import patch_openai
except ImportError:
    patch_openai = None

try:
    from noveum_trace.integrations.anthropic import patch_anthropic
except ImportError:
    patch_anthropic = None

try:
    from noveum_trace.integrations.langchain import patch_langchain
except ImportError:
    patch_langchain = None

try:
    from noveum_trace.integrations.llamaindex import patch_llamaindex
except ImportError:
    patch_llamaindex = None


def auto_patch_all() -> list[str]:
    """
    Automatically patch all available integrations.

    This function will attempt to patch all supported frameworks
    that are currently installed.
    """
    patched = []

    if patch_openai:
        try:
            patch_openai()
            patched.append("openai")
        except Exception:
            pass

    if patch_anthropic:
        try:
            patch_anthropic()
            patched.append("anthropic")
        except Exception:
            pass

    if patch_langchain:
        try:
            patch_langchain()
            patched.append("langchain")
        except Exception:
            pass

    if patch_llamaindex:
        try:
            patch_llamaindex()
            patched.append("llamaindex")
        except Exception:
            pass

    return patched


__all__ = [
    "auto_patch_all",
    "patch_openai",
    "patch_anthropic",
    "patch_langchain",
    "patch_llamaindex",
]
