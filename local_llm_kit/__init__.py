"""
local_llm_kit - OpenAI-like interface for local LLMs
"""

__version__ = "0.1.0"

from .llm import LLM
from .chat import chat, complete
from .function_calling import add_function

__all__ = ["LLM", "chat", "complete", "add_function"] 