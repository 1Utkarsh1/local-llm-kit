"""
Core LLM implementation with OpenAI-like API
"""
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, Tuple

from .backends.base import BaseBackend
from .backends.transformers import TransformersBackend
from .backends.llamacpp import LlamaCppBackend
from .prompt_formatting import get_prompt_formatter
from .memory import MessageHistory
from .function_calling import FunctionRegistry, FunctionCall, parse_function_calls

logger = logging.getLogger(__name__)

class LLM:
    """
    Main LLM class that provides an OpenAI-like interface for local language models.
    
    This class handles:
    - Chat and completion generation
    - Prompt formatting for different models
    - Function/tool calling
    - Memory and history management
    - Streaming support
    - Logprobs and token information
    - Structured output (JSON mode)
    """
    
    def __init__(
        self, 
        model_path: str,
        backend: Optional[str] = None,
        context_window: Optional[int] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the LLM with a local model.
        
        Args:
            model_path: Path or name of the model to load
            backend: Backend to use ("transformers", "llamacpp", or None for auto-detect)
            context_window: Max context window size (auto-detected if None)
            max_new_tokens: Maximum new tokens to generate per response
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            backend_kwargs: Additional arguments to pass to the backend
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.backend_kwargs = backend_kwargs or {}
        self.context_window = context_window
        
        # Initialize backend
        self.backend = self._init_backend(backend)
        
        # Auto-detect context window if not provided
        if not self.context_window:
            self.context_window = self.backend.get_context_window()
            logger.info(f"Auto-detected context window: {self.context_window}")
        
        # Initialize prompt formatter based on model name
        self.prompt_formatter = get_prompt_formatter(model_path)
        
        # Initialize memory for conversation history
        self.memory = MessageHistory(context_window=self.context_window)
        
        # Initialize function registry
        self.function_registry = FunctionRegistry()

    def _init_backend(self, backend_name: Optional[str]) -> BaseBackend:
        """Initialize the appropriate backend based on name or auto-detection."""
        if backend_name is None:
            # Auto-detect based on file extension or model path
            if ".gguf" in self.model_path.lower() or "ggml" in self.model_path.lower():
                backend_name = "llamacpp"
            else:
                backend_name = "transformers"

        if backend_name == "transformers":
            return TransformersBackend(self.model_path, **self.backend_kwargs)
        elif backend_name == "llamacpp":
            return LlamaCppBackend(self.model_path, **self.backend_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
    
    def add_function(self, name: str, schema: Dict[str, Any], implementation: Callable):
        """
        Register a function that can be called by the model.
        
        Args:
            name: Name of the function
            schema: JSON Schema describing the function parameters
            implementation: Python function to execute when called
        """
        self.function_registry.add_function(name, schema, implementation)
    
    def chat(
        self, 
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        format: Optional[str] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a chat completion for the provided messages.
        
        Args:
            messages: A list of message dictionaries with "role" and "content" keys
            functions: List of function specifications (optional)
            function_call: Controls when functions are called ("auto", "none", or specific function)
            temperature: Sampling temperature (overrides instance setting)
            max_tokens: Maximum number of tokens to generate (overrides instance setting)
            stream: Whether to stream the response
            format: Output format ("json" for structured JSON output)
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top tokens to return logprobs for
            
        Returns:
            A completion object or iterator of completion chunks if streaming
        """
        # Make local copies of messages and update memory
        message_list = list(messages)
        self.memory.add_messages(message_list)
        
        # Get functions from registry if not provided
        if functions is None and self.function_registry.has_functions():
            functions = self.function_registry.get_schema_list()
        
        # Prepare generation parameters
        gen_params = {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stream": stream,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs or 5 if logprobs else None,
        }
        
        # Format prompt according to model's template
        prompt = self.prompt_formatter.format_messages(
            message_list, 
            functions=functions, 
            function_call=function_call,
            json_mode=(format == "json")
        )
        
        if stream:
            return self._stream_chat_response(prompt, gen_params, format, functions, function_call)
        else:
            return self._generate_chat_response(prompt, gen_params, format, functions, function_call)
    
    def _generate_chat_response(
        self, 
        prompt: str, 
        gen_params: Dict[str, Any],
        format: Optional[str], 
        functions: Optional[List[Dict[str, Any]]], 
        function_call: Union[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """Generate a complete chat response."""
        # Generate response from the backend
        result = self.backend.generate(prompt, **gen_params)
        text = result["text"]
        
        # Handle JSON formatting
        if format == "json":
            text = self._ensure_json_output(text)
            
        # Handle function calling
        function_calls = None
        if functions and function_call != "none":
            try:
                function_calls = parse_function_calls(text)
                if function_calls and function_call != "none":
                    # Execute function if auto or specifically requested
                    if function_call == "auto" or (
                        isinstance(function_call, dict) and 
                        function_call.get("name") == function_calls[0].name
                    ):
                        func_result = self._execute_function_call(function_calls[0])
                        
                        # Add function result to messages
                        self.memory.add_messages([
                            {"role": "assistant", "content": None, "function_call": function_calls[0].to_dict()},
                            {"role": "function", "name": function_calls[0].name, "content": func_result}
                        ])
                        
                        # Generate a new response that includes the function result
                        new_prompt = self.prompt_formatter.format_messages(
                            self.memory.get_messages(),
                            functions=functions,
                            function_call=function_call,
                            json_mode=(format == "json")
                        )
                        result = self.backend.generate(new_prompt, **gen_params)
                        text = result["text"]
            except Exception as e:
                logger.warning(f"Error parsing function calls: {e}")
                
        # Prepare response object
        response = {
            "id": f"chatcmpl-{id(self)}-{id(prompt)}",
            "object": "chat.completion",
            "created": self.backend.get_timestamp(),
            "model": self.model_path,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }]
        }
        
        # Add function call to response if detected
        if function_calls:
            response["choices"][0]["message"]["function_call"] = function_calls[0].to_dict()
            response["choices"][0]["message"]["content"] = None
        
        # Add logprobs if requested
        if gen_params.get("logprobs"):
            response["choices"][0]["logprobs"] = result.get("logprobs", {})
            
        # Add usage information
        response["usage"] = {
            "prompt_tokens": self.backend.count_tokens(prompt),
            "completion_tokens": self.backend.count_tokens(text),
            "total_tokens": self.backend.count_tokens(prompt) + self.backend.count_tokens(text)
        }
            
        return response
        
    def _stream_chat_response(
        self, 
        prompt: str, 
        gen_params: Dict[str, Any],
        format: Optional[str], 
        functions: Optional[List[Dict[str, Any]]], 
        function_call: Union[str, Dict[str, str]]
    ) -> Iterator[Dict[str, Any]]:
        """Stream a chat response token by token."""
        response_id = f"chatcmpl-{id(self)}-{id(prompt)}"
        created = self.backend.get_timestamp()
        
        # Start generation stream
        stream = self.backend.generate_stream(prompt, **gen_params)
        accumulated_text = ""
        
        for chunk in stream:
            delta_text = chunk["text"]
            accumulated_text += delta_text
            
            # Prepare chunk response
            response_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_path,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": delta_text
                    },
                    "finish_reason": None
                }]
            }
            
            # Add logprobs if requested
            if gen_params.get("logprobs") and "logprobs" in chunk:
                response_chunk["choices"][0]["logprobs"] = chunk["logprobs"]
                
            yield response_chunk
            
        # Final chunk with finish reason
        if functions and function_call != "none":
            try:
                function_calls = parse_function_calls(accumulated_text)
                if function_calls:
                    # Last chunk contains function call
                    yield {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_path,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": None,
                                "function_call": function_calls[0].to_dict()
                            },
                            "finish_reason": "function_call"
                        }]
                    }
                    return
            except Exception as e:
                logger.warning(f"Error parsing function calls in stream: {e}")
        
        # Last chunk with finish reason
        yield {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_path,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    
    def _execute_function_call(self, function_call: FunctionCall) -> str:
        """Execute a function call and return the result as a string."""
        if not self.function_registry.has_function(function_call.name):
            return f"Error: Function {function_call.name} not found"
        
        try:
            result = self.function_registry.execute(
                function_call.name, 
                function_call.arguments
            )
            
            # Convert result to string (usually JSON)
            if not isinstance(result, str):
                result = json.dumps(result)
                
            return result
        except Exception as e:
            return f"Error executing function {function_call.name}: {str(e)}"
    
    def _ensure_json_output(self, text: str) -> str:
        """Ensure the output is valid JSON, retry if needed."""
        try:
            # Try to parse as JSON
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # Extract JSON-like content if the model added extra text
            import re
            json_match = re.search(r'(\{|\[).*?(\}|\])', text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json.loads(json_str)  # Validate it's proper JSON
                    return json_str
                except json.JSONDecodeError:
                    pass
            
            # If extraction failed, retry with a correction prompt
            correction_prompt = f"""The previous response was not valid JSON. 
Please respond with only a valid JSON object or array.

Previous invalid response:
{text}

Valid JSON response:"""
            
            result = self.backend.generate(
                correction_prompt,
                temperature=0.2,  # Lower temperature for more precise output
                max_new_tokens=self.max_new_tokens,
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            try:
                json.loads(result["text"])
                return result["text"]
            except json.JSONDecodeError:
                # If still failing, try one more extraction
                json_match = re.search(r'(\{|\[).*?(\}|\])', result["text"], re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        pass
                
                # Return a simple valid JSON as fallback
                return '{"error": "Failed to generate valid JSON", "attempted_response": ' + json.dumps(text) + '}'
    
    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        format: Optional[str] = None,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate a completion for the provided prompt.
        
        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature (overrides instance setting)
            max_tokens: Maximum number of tokens to generate (overrides instance setting)
            stream: Whether to stream the response
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top tokens to return logprobs for
            format: Output format ("json" for structured JSON output)
            
        Returns:
            A completion object or iterator of completion chunks if streaming
        """
        # Add text mode instruction for JSON if needed
        if format == "json":
            prompt = f"You must respond with valid JSON only, no other text.\n\n{prompt}"
            
        # Prepare generation parameters
        gen_params = {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_new_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stream": stream,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs or 5 if logprobs else None,
        }
        
        if stream:
            return self._stream_completion_response(prompt, gen_params, format)
        else:
            return self._generate_completion_response(prompt, gen_params, format)
    
    def _generate_completion_response(
        self, 
        prompt: str, 
        gen_params: Dict[str, Any],
        format: Optional[str]
    ) -> Dict[str, Any]:
        """Generate a complete text completion response."""
        # Generate response from the backend
        result = self.backend.generate(prompt, **gen_params)
        text = result["text"]
        
        # Handle JSON formatting
        if format == "json":
            text = self._ensure_json_output(text)
            
        # Prepare response object
        response = {
            "id": f"cmpl-{id(self)}-{id(prompt)}",
            "object": "text_completion",
            "created": self.backend.get_timestamp(),
            "model": self.model_path,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        
        # Add logprobs if requested
        if gen_params.get("logprobs"):
            response["choices"][0]["logprobs"] = result.get("logprobs", {})
            
        # Add usage information
        response["usage"] = {
            "prompt_tokens": self.backend.count_tokens(prompt),
            "completion_tokens": self.backend.count_tokens(text),
            "total_tokens": self.backend.count_tokens(prompt) + self.backend.count_tokens(text)
        }
            
        return response
        
    def _stream_completion_response(
        self, 
        prompt: str, 
        gen_params: Dict[str, Any],
        format: Optional[str]
    ) -> Iterator[Dict[str, Any]]:
        """Stream a text completion response token by token."""
        response_id = f"cmpl-{id(self)}-{id(prompt)}"
        created = self.backend.get_timestamp()
        
        # Start generation stream
        stream = self.backend.generate_stream(prompt, **gen_params)
        
        for chunk in stream:
            delta_text = chunk["text"]
            
            # Prepare chunk response
            response_chunk = {
                "id": response_id,
                "object": "text_completion.chunk",
                "created": created,
                "model": self.model_path,
                "choices": [{
                    "text": delta_text,
                    "index": 0,
                    "finish_reason": None
                }]
            }
            
            # Add logprobs if requested
            if gen_params.get("logprobs") and "logprobs" in chunk:
                response_chunk["choices"][0]["logprobs"] = chunk["logprobs"]
                
            yield response_chunk
            
        # Last chunk with finish reason
        yield {
            "id": response_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": self.model_path,
            "choices": [{
                "text": "",
                "index": 0,
                "finish_reason": "stop"
            }]
        } 