Tutorials
=========

This section contains detailed tutorials to help you make the most of ``local_llm_kit``.

Basic Chatbot Tutorial
--------------------

In this tutorial, we'll create a simple chatbot that can maintain context and respond to user queries.

.. code-block:: python

   from local_llm_kit import LLMClient
   
   # Initialize the client with memory management
   client = LLMClient(model="llama2")
   client.enable_memory(max_tokens=2000)
   
   def chatbot():
       # Set initial system message
       system_msg = {
           "role": "system",
           "content": "You are a helpful assistant who provides concise, accurate responses."
       }
       client.add_to_memory([system_msg])
       
       print("Chatbot: Hello! How can I help you today? (Type 'quit' to exit)")
       
       while True:
           user_input = input("You: ").strip()
           if user_input.lower() == 'quit':
               break
           
           # Add user message to memory
           user_msg = {"role": "user", "content": user_input}
           client.add_to_memory([user_msg])
           
           # Get response
           response = client.chat.completions.create(
               model="llama2",
               messages=client.get_memory(),
               temperature=0.7
           )
           
           # Print and store assistant's response
           assistant_msg = response.choices[0].message
           print(f"Chatbot: {assistant_msg.content}")
           client.add_to_memory([{
               "role": "assistant",
               "content": assistant_msg.content
           }])

   if __name__ == "__main__":
       chatbot()

Weather Assistant Tutorial
-----------------------

Let's create a weather assistant using function calling capabilities.

.. code-block:: python

   import json
   from local_llm_kit import LLMClient
   
   def get_weather(location: str, unit: str = "celsius") -> str:
       # This is a mock function - replace with actual weather API call
       weather_data = {
           "London": {"temp": 20, "condition": "cloudy"},
           "New York": {"temp": 25, "condition": "sunny"},
           "Tokyo": {"temp": 28, "condition": "rainy"}
       }
       
       if location not in weather_data:
           return f"Weather data not available for {location}"
           
       data = weather_data[location]
       temp = data["temp"]
       if unit == "fahrenheit":
           temp = (temp * 9/5) + 32
           
       return f"The temperature in {location} is {temp}°{'F' if unit == 'fahrenheit' else 'C'} and it's {data['condition']}"

   # Define the function schema
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

   def weather_assistant():
       client = LLMClient(model="llama2")
       
       while True:
           query = input("Ask about weather (or 'quit' to exit): ").strip()
           if query.lower() == 'quit':
               break
               
           response = client.chat.completions.create(
               model="llama2",
               messages=[{"role": "user", "content": query}],
               functions=[weather_function],
               function_call="auto"
           )
           
           message = response.choices[0].message
           
           if message.function_call:
               # Parse the function call
               func_args = json.loads(message.function_call.arguments)
               
               # Call the function
               weather_info = get_weather(
                   location=func_args["location"],
                   unit=func_args.get("unit", "celsius")
               )
               
               # Get final response
               final_response = client.chat.completions.create(
                   model="llama2",
                   messages=[
                       {"role": "user", "content": query},
                       {
                           "role": "function",
                           "name": "get_weather",
                           "content": weather_info
                       }
                   ]
               )
               print(f"Assistant: {final_response.choices[0].message.content}")
           else:
               print(f"Assistant: {message.content}")

JSON Output Tutorial
-----------------

Learn how to use JSON mode for structured outputs.

.. code-block:: python

   from local_llm_kit import LLMClient
   import json
   
   def structured_data_extractor():
       client = LLMClient(model="llama2")
       
       # Example: Extract person information
       text = """
       John Smith is a 35-year-old software engineer from San Francisco.
       He has been working at Tech Corp for 5 years and specializes in Python programming.
       """
       
       response = client.chat.completions.create(
           model="llama2",
           messages=[{
               "role": "user",
               "content": f"Extract structured information about the person from this text: {text}"
           }],
           response_format={"type": "json_object"}
       )
       
       # Parse and pretty print the JSON response
       person_info = json.loads(response.choices[0].message.content)
       print(json.dumps(person_info, indent=2))

Streaming Response Tutorial
------------------------

Implement a real-time response system with streaming.

.. code-block:: python

   from local_llm_kit import LLMClient
   import sys
   
   def streaming_story_generator():
       client = LLMClient(model="llama2")
       
       prompt = input("Enter a story prompt: ").strip()
       
       print("\nGenerating story...\n")
       
       for chunk in client.chat.completions.create(
           model="llama2",
           messages=[{
               "role": "user",
               "content": f"Write a short story about: {prompt}"
           }],
           stream=True
       ):
           content = chunk.choices[0].delta.content
           if content:
               sys.stdout.write(content)
               sys.stdout.flush()
       
       print("\n\nStory generation complete!")

Running the Tutorials
------------------

To run these tutorials:

1. Install the package:

   .. code-block:: bash

      pip install local-llm-kit

2. Copy the tutorial code into a Python file (e.g., ``tutorial.py``)

3. Run the file:

   .. code-block:: bash

      python tutorial.py

Tips and Best Practices
--------------------

1. Memory Management
   - Clear memory when starting new conversations
   - Monitor token usage to avoid hitting limits
   - Use appropriate context window sizes

2. Function Calling
   - Define clear, specific function descriptions
   - Handle edge cases in function implementations
   - Validate function arguments

3. Streaming
   - Use appropriate buffering for output
   - Handle connection interruptions
   - Consider implementing progress indicators

4. JSON Mode
   - Define expected schema in prompts
   - Handle parsing errors gracefully
   - Validate JSON structure

Next Steps
---------

- Explore the :doc:`api_reference` for more advanced features
- Check out the :doc:`examples` for more use cases
- Join our community and share your implementations 