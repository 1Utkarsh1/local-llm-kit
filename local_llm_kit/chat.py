"""
High-level functions for chat and completion.
"""
from typing import List, Dict, Any, Optional, Union, Iterator

from .llm import LLM


def chat(
    messages: List[Dict[str, str]],
    model_path: str,
    backend: Optional[str] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    function_call: Union[str, Dict[str, str]] = "auto",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    format: Optional[str] = None,
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
    **backend_kwargs
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """
    Generate a chat completion for the provided messages.
    
    This is a convenience function that creates an LLM instance and calls its chat method.
    
    Args:
        messages: A list of message dictionaries with "role" and "content" keys
        model_path: Path or name of the model to load
        backend: Backend to use ("transformers", "llamacpp", or None for auto-detect)
        functions: List of function specifications (optional)
        function_call: Controls when functions are called ("auto", "none", or specific function)
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        format: Output format ("json" for structured JSON output)
        logprobs: Whether to return log probabilities
        top_logprobs: Number of top tokens to return logprobs for
        **backend_kwargs: Additional arguments to pass to the backend
        
    Returns:
        A completion object or iterator of completion chunks if streaming
    """
    llm = LLM(
        model_path=model_path,
        backend=backend,
        temperature=temperature,
        max_new_tokens=max_tokens or 512,
        backend_kwargs=backend_kwargs
    )
    
    return llm.chat(
        messages=messages,
        functions=functions,
        function_call=function_call,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        format=format,
        logprobs=logprobs,
        top_logprobs=top_logprobs
    )


def complete(
    prompt: str,
    model_path: str,
    backend: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    format: Optional[str] = None,
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
    **backend_kwargs
) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
    """
    Generate a completion for the provided prompt.
    
    This is a convenience function that creates an LLM instance and calls its complete method.
    
    Args:
        prompt: The prompt to complete
        model_path: Path or name of the model to load
        backend: Backend to use ("transformers", "llamacpp", or None for auto-detect)
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        format: Output format ("json" for structured JSON output)
        logprobs: Whether to return log probabilities
        top_logprobs: Number of top tokens to return logprobs for
        **backend_kwargs: Additional arguments to pass to the backend
        
    Returns:
        A completion object or iterator of completion chunks if streaming
    """
    llm = LLM(
        model_path=model_path,
        backend=backend,
        temperature=temperature,
        max_new_tokens=max_tokens or 512,
        backend_kwargs=backend_kwargs
    )
    
    return llm.complete(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        format=format,
        logprobs=logprobs,
        top_logprobs=top_logprobs
    ) 