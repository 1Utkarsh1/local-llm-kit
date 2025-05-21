"""
Command-line interface for local_llm_kit.
"""
import argparse
import json
import sys
import os
import readline
from typing import List, Dict, Any, Optional, Union

from .llm import LLM
from .chat import chat, complete


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Local LLM Kit - Use local LLMs with OpenAI-like API")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with a model")
    chat_parser.add_argument("--model", "-m", required=True, help="Path to the model or model name")
    chat_parser.add_argument("--backend", "-b", choices=["transformers", "llamacpp"], help="Backend to use")
    chat_parser.add_argument("--system", "-s", help="Optional system message")
    chat_parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    chat_parser.add_argument("--stream", action="store_true", help="Stream the response")
    chat_parser.add_argument("--json", action="store_true", help="Request JSON responses")
    chat_parser.add_argument("--functions", "-f", help="Path to JSON file with function definitions")
    chat_parser.add_argument("--function-call", help="Control when functions are called ('auto', 'none', or '{'name': 'function_name'}')")
    chat_parser.add_argument("--gpu-layers", type=int, default=-1, help="Number of GPU layers for llama.cpp (-1 for all)")
    chat_parser.add_argument("--device", help="Device to use for transformers ('cpu', 'cuda', 'mps')")
    
    # Completion command
    completion_parser = subparsers.add_parser("complete", help="Complete a prompt")
    completion_parser.add_argument("--model", "-m", required=True, help="Path to the model or model name")
    completion_parser.add_argument("--backend", "-b", choices=["transformers", "llamacpp"], help="Backend to use")
    completion_parser.add_argument("--prompt", "-p", help="Prompt to complete (if not provided, reads from stdin)")
    completion_parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    completion_parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    completion_parser.add_argument("--stream", action="store_true", help="Stream the response")
    completion_parser.add_argument("--json", action="store_true", help="Request JSON responses")
    completion_parser.add_argument("--gpu-layers", type=int, default=-1, help="Number of GPU layers for llama.cpp (-1 for all)")
    completion_parser.add_argument("--device", help="Device to use for transformers ('cpu', 'cuda', 'mps')")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Prepare backend kwargs based on backend type
    backend_kwargs = {}
    if args.backend == "llamacpp":
        backend_kwargs["n_gpu_layers"] = args.gpu_layers
    elif args.backend == "transformers" and args.device:
        backend_kwargs["device"] = args.device
    
    if args.command == "chat":
        handle_chat_command(args, backend_kwargs)
    elif args.command == "complete":
        handle_completion_command(args, backend_kwargs)


def handle_chat_command(args, backend_kwargs):
    """Handle the chat command."""
    # Load functions if provided
    functions = None
    if args.functions:
        with open(args.functions, 'r') as f:
            functions = json.load(f)
    
    # Parse function_call
    function_call = "auto"
    if args.function_call:
        if args.function_call in ["auto", "none"]:
            function_call = args.function_call
        else:
            try:
                function_call = json.loads(args.function_call)
            except json.JSONDecodeError:
                print(f"Error: Invalid function_call format: {args.function_call}")
                sys.exit(1)
    
    # Create LLM instance
    llm = LLM(
        model_path=args.model,
        backend=args.backend,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        backend_kwargs=backend_kwargs
    )
    
    # Initialize message history
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    
    # Interactive chat loop
    print(f"Chat with {args.model} (type 'exit' or 'quit' to end, 'clear' to reset history)")
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Handle special commands
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "clear":
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                print("Chat history cleared.")
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            format_param = "json" if args.json else None
            
            if args.stream:
                # Handle streaming
                print("\nAssistant: ", end="", flush=True)
                response_text = ""
                
                for chunk in llm.chat(
                    messages=messages,
                    functions=functions,
                    function_call=function_call,
                    stream=True,
                    format=format_param
                ):
                    # Extract and print chunk text
                    if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            print(delta["content"], end="", flush=True)
                            response_text += delta["content"]
                
                print()  # Add newline after streaming finishes
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": response_text})
                
            else:
                # Handle non-streaming response
                response = llm.chat(
                    messages=messages,
                    functions=functions,
                    function_call=function_call,
                    format=format_param
                )
                
                if "choices" in response and response["choices"]:
                    choice = response["choices"][0]
                    response_message = choice["message"]
                    
                    # Add assistant response to history
                    messages.append(response_message)
                    
                    # Print response
                    if "function_call" in response_message:
                        print(f"\nAssistant (function call): {json.dumps(response_message['function_call'], indent=2)}")
                    else:
                        print(f"\nAssistant: {response_message['content']}")
    
    except KeyboardInterrupt:
        print("\nExiting chat...")
    except Exception as e:
        print(f"\nError: {e}")


def handle_completion_command(args, backend_kwargs):
    """Handle the completion command."""
    # Get prompt from args or stdin
    if args.prompt:
        prompt = args.prompt
    else:
        print("Enter your prompt (Ctrl+D to finish):")
        prompt_lines = []
        try:
            for line in sys.stdin:
                prompt_lines.append(line)
        except KeyboardInterrupt:
            print("\nPrompt input cancelled.")
            sys.exit(1)
        prompt = "".join(prompt_lines)
    
    # Exit if prompt is empty
    if not prompt.strip():
        print("Error: Empty prompt")
        sys.exit(1)
    
    format_param = "json" if args.json else None
    
    if args.stream:
        # Handle streaming
        print("\nCompletion: ", end="", flush=True)
        
        for chunk in complete(
            prompt=prompt,
            model_path=args.model,
            backend=args.backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=True,
            format=format_param,
            **backend_kwargs
        ):
            # Extract and print chunk text
            if "choices" in chunk and chunk["choices"] and "text" in chunk["choices"][0]:
                text = chunk["choices"][0]["text"]
                if text:
                    print(text, end="", flush=True)
        
        print()  # Add newline after streaming finishes
        
    else:
        # Handle non-streaming response
        response = complete(
            prompt=prompt,
            model_path=args.model,
            backend=args.backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            format=format_param,
            **backend_kwargs
        )
        
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "text" in choice:
                print(f"\nCompletion: {choice['text']}")


if __name__ == "__main__":
    main() 