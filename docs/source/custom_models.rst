Custom Models
============

This guide shows how to extend ``local_llm_kit`` to support custom models.

Adding Support for Custom Models
----------------------------

``local_llm_kit`` is designed to be extensible, allowing you to add support for custom models and backends.

Custom Tokenizer
--------------

You can implement a custom tokenizer by creating a class that implements the necessary encoding and decoding methods:

.. code-block:: python

   from local_llm_kit import LLMClient
   from local_llm_kit.backends.base import BaseTokenizer
   
   class MyCustomTokenizer(BaseTokenizer):
       def __init__(self, model_path=None):
           super().__init__()
           # Initialize your tokenizer here
           # This could use a pretrained tokenizer or your own implementation
           self.model_path = model_path
           
       def encode(self, text):
           # Convert text to token IDs
           # Return a list of token IDs
           pass
           
       def decode(self, token_ids):
           # Convert token IDs back to text
           # Return a string
           pass
           
       def get_vocab_size(self):
           # Return the vocabulary size of your tokenizer
           pass

   # Use your custom tokenizer
   client = LLMClient(
       model="custom-model",
       tokenizer=MyCustomTokenizer(model_path="/path/to/tokenizer"),
       model_path="/path/to/model/weights"
   )

Custom Backend
-----------

For more advanced customization, you can implement a custom backend:

.. code-block:: python

   from local_llm_kit.backends.base import BaseBackend
   
   class MyCustomBackend(BaseBackend):
       def __init__(self, model_path, **kwargs):
           super().__init__()
           # Initialize your model here
           self.model_path = model_path
           # Load your model or set up your inference engine
           
       def generate(self, prompt, max_tokens=100, temperature=0.7, **kwargs):
           # Implement the generation logic for your model
           # Return a string containing the generated text
           pass
           
       def get_prompt_tokens(self, prompt):
           # Return the number of tokens in the prompt
           pass
           
       def get_completion_tokens(self, completion):
           # Return the number of tokens in the completion
           pass

   # Register your custom backend
   from local_llm_kit.llm import LLM
   
   LLM.register_backend("my-custom-backend", MyCustomBackend)
   
   # Use your custom backend
   client = LLMClient(
       model="custom-model",
       backend="my-custom-backend",
       model_path="/path/to/model/weights"
   )

Custom Prompt Formatting
---------------------

You can also define custom prompt templates for your models:

.. code-block:: python

   from local_llm_kit.prompt_formatting import register_prompt_formatter
   
   def my_custom_formatter(messages, add_generation_prompt=True):
       """
       Format chat messages for a custom model architecture.
       """
       formatted_prompt = ""
       
       for message in messages:
           role = message["role"]
           content = message["content"]
           
           if role == "system":
               formatted_prompt += f"<|system|>\n{content}\n"
           elif role == "user":
               formatted_prompt += f"<|user|>\n{content}\n"
           elif role == "assistant":
               formatted_prompt += f"<|assistant|>\n{content}\n"
           elif role == "function":
               formatted_prompt += f"<|function|>\n{content}\n"
       
       if add_generation_prompt:
           formatted_prompt += "<|assistant|>\n"
           
       return formatted_prompt
   
   # Register your custom formatter
   register_prompt_formatter("my-custom-model", my_custom_formatter)
   
   # Use your custom formatter
   client = LLMClient(
       model="my-custom-model",
       # Other parameters...
   )

Example: Integrating with vLLM
---------------------------

Here's an example of integrating with the vLLM inference engine:

.. code-block:: python

   from local_llm_kit.backends.base import BaseBackend
   
   class VLLMBackend(BaseBackend):
       def __init__(self, model_path, **kwargs):
           super().__init__()
           
           # Import vLLM here to avoid making it a hard dependency
           from vllm import LLM
           
           # Initialize vLLM engine
           self.engine = LLM(
               model=model_path,
               tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
               gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
               # Other vLLM parameters...
           )
           
       def generate(self, prompt, max_tokens=100, temperature=0.7, **kwargs):
           from vllm import SamplingParams
           
           # Set up sampling parameters
           sampling_params = SamplingParams(
               temperature=temperature,
               max_tokens=max_tokens,
               top_p=kwargs.get("top_p", 1.0),
               # Other sampling parameters...
           )
           
           # Generate text with vLLM
           outputs = self.engine.generate(prompt, sampling_params)
           
           # Extract generated text
           generated_text = outputs[0].outputs[0].text
           
           return generated_text
   
   # Register vLLM backend
   from local_llm_kit.llm import LLM
   
   LLM.register_backend("vllm", VLLMBackend)
   
   # Use vLLM backend
   client = LLMClient(
       model="llama2",
       backend="vllm",
       model_path="meta-llama/Llama-2-70b-chat-hf",
       tensor_parallel_size=4  # For multi-GPU inference
   ) 