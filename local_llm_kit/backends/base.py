"""
Base class for model backends.
"""
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional


class BaseBackend(ABC):
    """
    Abstract base class for model backends.
    
    All backend implementations must inherit from this class and
    implement the required methods.
    """
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text completion for a prompt.
        
        Args:
            prompt: Input text to complete
            temperature: Sampling temperature (higher = more random)
            max_new_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            stream: Whether to stream the response (should be False for this method)
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top tokens to return logprobs for
            **kwargs: Additional backend-specific parameters
            
        Returns:
            Dictionary with at least a "text" key containing the generated text
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self, 
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream text completion for a prompt.
        
        Args:
            prompt: Input text to complete
            temperature: Sampling temperature (higher = more random)
            max_new_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            logprobs: Whether to return log probabilities
            top_logprobs: Number of top tokens to return logprobs for
            **kwargs: Additional backend-specific parameters
            
        Yields:
            Dictionaries with at least a "text" key containing a chunk of generated text
        """
        pass
    
    @abstractmethod
    def get_context_window(self) -> int:
        """
        Get the context window size of the model.
        
        Returns:
            Maximum context window size in tokens
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        pass
    
    def get_timestamp(self) -> int:
        """
        Get current timestamp.
        
        Returns:
            Current timestamp in seconds
        """
        return int(time.time()) 