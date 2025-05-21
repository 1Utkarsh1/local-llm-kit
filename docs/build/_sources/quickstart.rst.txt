Quickstart Guide
==============

This guide will help you get started with ``local_llm_kit`` quickly.

Installation
-----------

Install the package using pip:

.. code-block:: bash

   pip install local-llm-kit

Basic Usage
----------

1. Initialize the Client
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from local_llm_kit import LLMClient
   
   # Initialize with default settings
   client = LLMClient(model="llama2")

2. Chat Completions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple chat completion
   response = client.chat.completions.create(
       model="llama2",
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "What is machine learning?"}
       ]
   )
   
   print(response.choices[0].message.content)

3. Streaming Responses
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Stream the response
   for chunk in client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": "Write a short story"}],
       stream=True
   ):
       print(chunk.choices[0].delta.content, end="")

4. Function Calling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   functions = [
       {
           "name": "get_weather",
           "description": "Get the weather in a location",
           "parameters": {
               "type": "object",
               "properties": {
                   "location": {"type": "string"},
                   "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
               },
               "required": ["location"]
           }
       }
   ]

   response = client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": "What's the weather like in London?"}],
       functions=functions,
       function_call="auto"
   )

5. JSON Mode
~~~~~~~~~~~

.. code-block:: python

   response = client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": "List three colors with their hex codes"}],
       response_format={"type": "json_object"}
   )

Advanced Usage
-------------

Model Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   client = LLMClient(
       model="llama2",
       model_path="/path/to/model",
       context_length=2048,
       temperature=0.7,
       top_p=0.9
   )

Memory Management
~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable memory management
   client.enable_memory(max_tokens=1000)
   
   # Add conversation context
   client.add_to_memory([
       {"role": "user", "content": "My name is Alice"},
       {"role": "assistant", "content": "Hello Alice!"}
   ])

Error Handling
~~~~~~~~~~~~

.. code-block:: python

   try:
       response = client.chat.completions.create(
           model="nonexistent_model",
           messages=[{"role": "user", "content": "Hello"}]
       )
   except Exception as e:
       print(f"An error occurred: {e}")

Next Steps
---------

- Check out the :doc:`api_reference` for detailed API documentation
- See more :doc:`examples` for advanced usage patterns
- Learn about supported :doc:`models` and their configurations
- Consider :doc:`contributing` to the project 