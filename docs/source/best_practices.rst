Best Practices
=============

This guide outlines best practices for using ``local_llm_kit`` effectively and efficiently.

Model Selection
-------------

Choosing the Right Model
~~~~~~~~~~~~~~~~~~~~~~

1. Consider your requirements:
   * Task complexity
   * Response quality needs
   * Performance constraints
   * Resource availability

2. Match model size to available resources:
   * 7B models for most use cases
   * 13B+ models for higher quality
   * Quantized models for resource constraints

3. Consider specialization:
   * Chat models for dialogue
   * Code models for programming
   * Multi-lingual models for language tasks

Performance Optimization
---------------------

Memory Management
~~~~~~~~~~~~~~~

1. Token Management:
   .. code-block:: python

      client = LLMClient(model="llama2")
      client.enable_memory(max_tokens=1000)
      
      # Regularly clear memory
      client.clear_memory()
      
      # Monitor token usage
      response = client.chat.completions.create(...)
      print(f"Used tokens: {response.usage.total_tokens}")

2. Batch Processing:
   .. code-block:: python

      # Process multiple prompts efficiently
      responses = []
      for prompt in prompts:
          response = client.chat.completions.create(
              model="llama2",
              messages=[{"role": "user", "content": prompt}],
              max_tokens=50  # Limit response length
          )
          responses.append(response)

GPU Utilization
~~~~~~~~~~~~~

1. Optimal Settings:
   .. code-block:: python

      client = LLMClient(
          model="llama2",
          device="cuda",
          use_flash_attention=True,
          max_batch_size=32,
          dtype="float16"
      )

2. Memory Monitoring:
   * Use GPU monitoring tools
   * Adjust batch size based on memory
   * Consider gradient checkpointing

Error Handling
------------

Robust Implementation
~~~~~~~~~~~~~~~~~~

1. Basic Error Handling:
   .. code-block:: python

      try:
          response = client.chat.completions.create(
              model="llama2",
              messages=[{"role": "user", "content": "Hello"}]
          )
      except Exception as e:
          logger.error(f"Chat completion failed: {e}")
          # Implement fallback behavior

2. Specific Error Types:
   .. code-block:: python

      from local_llm_kit.exceptions import (
          ModelNotFoundError,
          TokenLimitError,
          InvalidRequestError
      )

      try:
          # Your code here
      except ModelNotFoundError:
          # Handle missing model
      except TokenLimitError:
          # Handle token limit exceeded
      except InvalidRequestError:
          # Handle invalid parameters

Retry Logic
~~~~~~~~~~

.. code-block:: python

   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(stop=stop_after_attempt(3), wait=wait_exponential())
   def get_completion(prompt):
       return client.chat.completions.create(
           model="llama2",
           messages=[{"role": "user", "content": prompt}]
       )

Prompt Engineering
---------------

Effective Prompts
~~~~~~~~~~~~~~~

1. Clear Instructions:
   .. code-block:: python

      messages = [
          {
              "role": "system",
              "content": "You are a helpful assistant. Provide clear, concise answers."
          },
          {
              "role": "user",
              "content": "What is machine learning? Explain in simple terms."
          }
      ]

2. Context Management:
   .. code-block:: python

      # Add relevant context
      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": "Remember this: I'm John."},
          {"role": "assistant", "content": "Hello John!"},
          {"role": "user", "content": "What's my name?"}
      ]

Temperature Control
~~~~~~~~~~~~~~~~

1. For Deterministic Responses:
   .. code-block:: python

      response = client.chat.completions.create(
          model="llama2",
          messages=messages,
          temperature=0.0,  # More deterministic
          top_p=1.0
      )

2. For Creative Responses:
   .. code-block:: python

      response = client.chat.completions.create(
          model="llama2",
          messages=messages,
          temperature=0.8,  # More creative
          top_p=0.9
      )

Security Considerations
--------------------

Input Validation
~~~~~~~~~~~~~~

1. Sanitize Inputs:
   .. code-block:: python

      def sanitize_input(text):
          # Implement input sanitization
          return cleaned_text

      user_input = sanitize_input(raw_input)
      response = client.chat.completions.create(
          model="llama2",
          messages=[{"role": "user", "content": user_input}]
      )

2. Content Filtering:
   .. code-block:: python

      def is_safe_content(text):
          # Implement content safety checks
          return is_safe

      if not is_safe_content(user_input):
          raise SecurityError("Unsafe content detected")

Model Security
~~~~~~~~~~~~

1. Model Access Control:
   .. code-block:: python

      # Use environment variables for sensitive paths
      model_path = os.getenv("LOCAL_LLM_KIT_MODEL_PATH")
      client = LLMClient(
          model="llama2",
          model_path=model_path
      )

2. Rate Limiting:
   .. code-block:: python

      from ratelimit import limits, sleep_and_retry

      @sleep_and_retry
      @limits(calls=10, period=60)  # 10 calls per minute
      def rate_limited_completion(prompt):
          return client.chat.completions.create(
              model="llama2",
              messages=[{"role": "user", "content": prompt}]
          )

Monitoring and Logging
-------------------

Logging Setup
~~~~~~~~~~~

1. Basic Logging:
   .. code-block:: python

      import logging

      logging.basicConfig(level=logging.INFO)
      logger = logging.getLogger("local_llm_kit")

      logger.info("Initializing client...")
      client = LLMClient(model="llama2")

2. Detailed Logging:
   .. code-block:: python

      handler = logging.FileHandler("llm.log")
      handler.setFormatter(logging.Formatter(
          '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      ))
      logger.addHandler(handler)

Performance Monitoring
~~~~~~~~~~~~~~~~~~

1. Response Times:
   .. code-block:: python

      import time

      start_time = time.time()
      response = client.chat.completions.create(...)
      duration = time.time() - start_time
      
      logger.info(f"Response time: {duration:.2f}s")

2. Resource Usage:
   .. code-block:: python

      import psutil

      def log_resource_usage():
          process = psutil.Process()
          logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
          logger.info(f"CPU usage: {process.cpu_percent()}%")

Testing and Validation
-------------------

Unit Testing
~~~~~~~~~~

1. Basic Tests:
   .. code-block:: python

      import unittest

      class TestLLMClient(unittest.TestCase):
          def setUp(self):
              self.client = LLMClient(model="llama2")
              
          def test_completion(self):
              response = self.client.chat.completions.create(
                  model="llama2",
                  messages=[{"role": "user", "content": "Hello"}]
              )
              self.assertIsNotNone(response)

2. Mock Testing:
   .. code-block:: python

      from unittest.mock import patch

      @patch("local_llm_kit.LLMClient")
      def test_with_mock(mock_client):
          mock_client.return_value.chat.completions.create.return_value = mock_response
          # Test implementation

Integration Testing
~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_end_to_end():
       client = LLMClient(model="llama2")
       
       # Test chat completion
       response1 = client.chat.completions.create(...)
       
       # Test function calling
       response2 = client.chat.completions.create(
           functions=[function_spec],
           function_call="auto",
           ...
       )
       
       # Test streaming
       response3 = client.chat.completions.create(
           stream=True,
           ...
       )

Deployment Best Practices
----------------------

Environment Setup
~~~~~~~~~~~~~~

1. Dependencies:
   .. code-block:: bash

      pip install local-llm-kit[all]  # Install all optional dependencies
      
2. Environment Variables:
   .. code-block:: bash

      export LOCAL_LLM_KIT_MODEL_PATH="/path/to/models"
      export LOCAL_LLM_KIT_CACHE_DIR="/path/to/cache"

Production Configuration
~~~~~~~~~~~~~~~~~~~~~

1. Load Balancing:
   .. code-block:: python

      clients = [
          LLMClient(model="llama2", device=f"cuda:{i}")
          for i in range(torch.cuda.device_count())
      ]

2. Health Checks:
   .. code-block:: python

      def health_check():
          try:
              response = client.chat.completions.create(
                  model="llama2",
                  messages=[{"role": "user", "content": "test"}]
              )
              return True
          except Exception:
              return False

Remember to regularly review and update these practices based on your specific use case and requirements. 