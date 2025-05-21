API Reference
============

This page provides detailed documentation for the ``local_llm_kit`` API.

LLMClient
---------

.. py:class:: LLMClient(model: str, **kwargs)

   The main client class for interacting with local language models.

   :param model: The name or path of the model to use
   :param model_path: Optional path to model weights
   :param context_length: Maximum context length (default: 2048)
   :param temperature: Sampling temperature (default: 0.7)
   :param top_p: Top-p sampling parameter (default: 0.9)
   :param backend: Model backend to use ('transformers' or 'llama.cpp')

Chat Completions
--------------

.. py:method:: LLMClient.chat.completions.create(**kwargs)

   Create a chat completion.

   :param model: Model to use for completion
   :param messages: List of message dictionaries
   :param temperature: Sampling temperature
   :param top_p: Top-p sampling parameter
   :param max_tokens: Maximum tokens to generate
   :param stream: Whether to stream the response
   :param functions: List of function definitions
   :param function_call: Function call behavior
   :param response_format: Specify response format (e.g., JSON)
   :return: CompletionResponse object

Memory Management
---------------

.. py:method:: LLMClient.enable_memory(max_tokens: int = 1000)

   Enable conversation memory management.

   :param max_tokens: Maximum tokens to store in memory

.. py:method:: LLMClient.add_to_memory(messages: List[Dict])

   Add messages to conversation memory.

   :param messages: List of message dictionaries

.. py:method:: LLMClient.clear_memory()

   Clear all stored conversation memory.

Response Objects
--------------

CompletionResponse
~~~~~~~~~~~~~~~~

.. py:class:: CompletionResponse

   Represents a completion response.

   :param id: Response ID
   :param object: Object type
   :param created: Creation timestamp
   :param model: Model used
   :param choices: List of completion choices
   :param usage: Token usage statistics

Choice
~~~~~~

.. py:class:: Choice

   Represents a completion choice.

   :param index: Choice index
   :param message: Message content
   :param finish_reason: Reason for completion

Message
~~~~~~~

.. py:class:: Message

   Represents a chat message.

   :param role: Message role (user/assistant/system)
   :param content: Message content
   :param function_call: Optional function call

Usage
~~~~~

.. py:class:: Usage

   Token usage statistics.

   :param prompt_tokens: Number of tokens in prompt
   :param completion_tokens: Number of tokens in completion
   :param total_tokens: Total tokens used

Exceptions
---------

.. py:exception:: ModelNotFoundError

   Raised when specified model is not found.

.. py:exception:: InvalidRequestError

   Raised when request parameters are invalid.

.. py:exception:: TokenLimitError

   Raised when token limit is exceeded.

Configuration
-----------

The following environment variables can be used to configure the client:

- ``LOCAL_LLM_KIT_MODEL_PATH``: Default path to model weights
- ``LOCAL_LLM_KIT_BACKEND``: Default backend to use
- ``LOCAL_LLM_KIT_CONTEXT_LENGTH``: Default context length
- ``LOCAL_LLM_KIT_CACHE_DIR``: Directory for caching model weights 