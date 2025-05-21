Examples
========

This section contains various examples of using ``local_llm_kit``.

Basic Chat Example
----------------

.. code-block:: python

   from local_llm_kit import LLMClient
   
   # Initialize client
   client = LLMClient(model="llama2")
   
   # Create a simple chat message
   messages = [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "What is the capital of France?"}
   ]
   
   # Get completion
   response = client.chat.completions.create(
       model="llama2",
       messages=messages
   )
   
   # Print response
   print(response.choices[0].message.content)

Function Calling Example
---------------------

.. code-block:: python

   import json
   from local_llm_kit import LLMClient
   
   # Define a function to get weather
   def get_weather(location, unit="celsius"):
       # Mock implementation
       return f"The weather in {location} is sunny and 25°{unit[0].upper()}"
   
   # Define function schema
   weather_function = {
       "name": "get_weather",
       "description": "Get the current weather in a location",
       "parameters": {
           "type": "object",
           "properties": {
               "location": {
                   "type": "string",
                   "description": "The city to get weather for"
               },
               "unit": {
                   "type": "string",
                   "enum": ["celsius", "fahrenheit"],
                   "description": "Temperature unit"
               }
           },
           "required": ["location"]
       }
   }
   
   # Initialize client
   client = LLMClient(model="llama2")
   
   # Create function call
   response = client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
       functions=[weather_function],
       function_call="auto"
   )
   
   # Process function call
   message = response.choices[0].message
   if message.function_call:
       func_args = json.loads(message.function_call.arguments)
       weather_info = get_weather(
           location=func_args["location"],
           unit=func_args.get("unit", "celsius")
       )
       print(f"Function result: {weather_info}")
   else:
       print(f"Assistant response: {message.content}")

Streaming Example
--------------

.. code-block:: python

   from local_llm_kit import LLMClient
   
   # Initialize client
   client = LLMClient(model="llama2")
   
   # Stream response
   for chunk in client.chat.completions.create(
       model="llama2",
       messages=[{"role": "user", "content": "Tell me a short story"}],
       stream=True
   ):
       print(chunk.choices[0].delta.content or "", end="", flush=True)
   
   print("\nStreaming complete!") 