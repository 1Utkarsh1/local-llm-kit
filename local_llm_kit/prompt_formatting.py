"""
Prompt formatting for different model types.
"""
import json
import re
from typing import List, Dict, Any, Optional, Union, Type


class BasePromptFormatter:
    """Base class for prompt formatters."""
    
    def format_messages(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        json_mode: bool = False
    ) -> str:
        """
        Format messages according to model's expected format.
        
        Args:
            messages: List of message dicts with "role" and "content"
            functions: List of function specifications
            function_call: Controls when functions are called
            json_mode: Whether to format for JSON output
            
        Returns:
            Formatted prompt string
        """
        raise NotImplementedError("Subclasses must implement format_messages")


class ChatMLPromptFormatter(BasePromptFormatter):
    """
    Formatter for ChatML format used by models like GPT-3.5/4.
    """
    
    def format_messages(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        json_mode: bool = False
    ) -> str:
        """Format messages in ChatML format."""
        formatted_messages = []
        
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                formatted_messages.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_messages.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                if message.get("function_call"):
                    func_call = message["function_call"]
                    func_call_str = json.dumps(func_call)
                    formatted_messages.append(f"<|im_start|>assistant\n{func_call_str}<|im_end|>")
                else:
                    formatted_messages.append(f"<|im_start|>assistant\n{content or ''}<|im_end|>")
            elif role == "function":
                formatted_messages.append(f"<|im_start|>function\nname: {message.get('name')}\ncontent: {content}<|im_end|>")
        
        # Add function definitions if provided
        if functions:
            func_text = "Functions available to call:\n"
            func_text += json.dumps(functions, indent=2)
            formatted_messages.append(f"<|im_start|>system\n{func_text}<|im_end|>")
            
            # Add function call preference
            if function_call != "auto":
                if function_call == "none":
                    func_pref = "You should not call functions."
                elif isinstance(function_call, dict) and "name" in function_call:
                    func_pref = f"You should specifically call the function named {function_call['name']}."
                else:
                    func_pref = "You can call functions if needed."
                    
                formatted_messages.append(f"<|im_start|>system\n{func_pref}<|im_end|>")
        
        # Add JSON mode instruction if requested
        if json_mode:
            formatted_messages.append("<|im_start|>system\nYou must respond with a valid JSON object or array, without any additional text.\n<|im_end|>")
            
        # Add assistant prompt
        formatted_messages.append("<|im_start|>assistant")
        
        return "\n".join(formatted_messages)


class Llama2ChatPromptFormatter(BasePromptFormatter):
    """
    Formatter for Llama2 Chat models using [INST] format.
    """
    
    def format_messages(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        json_mode: bool = False
    ) -> str:
        """Format messages in Llama2 Chat format."""
        system_content = ""
        
        # Extract system message
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
                break
                
        conversation = []
        current_turn = {"user": "", "assistant": ""}
        
        # Process conversation turns
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                # Already handled
                continue
            elif role == "user":
                # Complete the previous turn if needed
                if current_turn["user"] and current_turn["assistant"]:
                    conversation.append(current_turn)
                    current_turn = {"user": "", "assistant": ""}
                
                current_turn["user"] = content
            elif role == "assistant":
                if message.get("function_call"):
                    # Format function call as JSON
                    func_call = message["function_call"]
                    func_call_str = json.dumps(func_call)
                    current_turn["assistant"] = func_call_str
                else:
                    current_turn["assistant"] = content or ""
            elif role == "function":
                # Add function result as part of user's next message
                func_result = f"[Function Result: {message.get('name')}]\n{content}"
                if current_turn["user"]:
                    current_turn["user"] += f"\n\n{func_result}"
                else:
                    current_turn["user"] = func_result
        
        # Add the last turn if not empty
        if current_turn["user"]:
            conversation.append(current_turn)
            
        # Format the prompt
        formatted_parts = []
        
        # Format system message
        if system_content:
            if functions:
                # Add function specifications to system message
                func_text = "\n\nFunctions available to call:\n"
                func_text += json.dumps(functions, indent=2)
                
                # Add function call preference
                if function_call != "auto":
                    if function_call == "none":
                        func_pref = "\n\nYou should not call functions."
                    elif isinstance(function_call, dict) and "name" in function_call:
                        func_pref = f"\n\nYou should specifically call the function named {function_call['name']}."
                    else:
                        func_pref = "\n\nYou can call functions if needed."
                    func_text += func_pref
                
                system_content += func_text
                
            # Add JSON mode instruction if requested
            if json_mode:
                system_content += "\n\nYou must respond with a valid JSON object or array, without any additional text."
        
        # Format conversation
        for i, turn in enumerate(conversation):
            user_text = turn["user"]
            assistant_text = turn["assistant"]
            
            if i == 0 and system_content:
                # First turn includes system message
                formatted_parts.append(f"<s>[INST] {system_content}\n\n{user_text} [/INST] {assistant_text}")
            else:
                # Subsequent turns
                formatted_parts.append(f"[INST] {user_text} [/INST] {assistant_text}")
        
        # If there was no conversation but we have a system message, create a prompt for it
        if not conversation and system_content:
            formatted_parts.append(f"<s>[INST] {system_content} [/INST]")
        
        # Handle the case where the last message is from the user (awaiting assistant response)
        elif conversation and not conversation[-1]["assistant"]:
            last_user_text = conversation[-1]["user"]
            if system_content and len(conversation) == 1:
                formatted_parts.append(f"<s>[INST] {system_content}\n\n{last_user_text} [/INST]")
            else:
                formatted_parts.append(f"[INST] {last_user_text} [/INST]")
                
        return " ".join(formatted_parts)


class MistralInstructPromptFormatter(BasePromptFormatter):
    """
    Formatter for Mistral Instruct models using <s>[INST] format.
    """
    
    def format_messages(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        json_mode: bool = False
    ) -> str:
        """Format messages in Mistral Instruct format."""
        system_content = ""
        
        # Extract system message
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
                break
                
        # Process conversation turns
        conversation_parts = []
        
        for i, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                # Will be added to the first user message
                continue
            elif role == "user":
                # Add system message to the first user message
                if i == 0 or (i == 1 and messages[0]["role"] == "system"):
                    if system_content:
                        if functions:
                            # Add function specifications to system message
                            func_text = "\n\nFunctions available to call:\n"
                            func_text += json.dumps(functions, indent=2)
                            
                            # Add function call preference
                            if function_call != "auto":
                                if function_call == "none":
                                    func_pref = "\n\nYou should not call functions."
                                elif isinstance(function_call, dict) and "name" in function_call:
                                    func_pref = f"\n\nYou should specifically call the function named {function_call['name']}."
                                else:
                                    func_pref = "\n\nYou can call functions if needed."
                                func_text += func_pref
                            
                            system_content += func_text
                        
                        # Add JSON mode instruction if requested
                        if json_mode:
                            system_content += "\n\nYou must respond with a valid JSON object or array, without any additional text."
                            
                        user_content = f"{system_content}\n\n{content}"
                    else:
                        user_content = content
                        
                    conversation_parts.append(f"<s>[INST] {user_content} [/INST]")
                else:
                    conversation_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                if message.get("function_call"):
                    # Format function call as JSON
                    func_call = message["function_call"]
                    func_call_str = json.dumps(func_call)
                    conversation_parts.append(func_call_str)
                else:
                    conversation_parts.append(content or "")
            elif role == "function":
                # Add function result to next user message
                func_result = f"[Function Result: {message.get('name')}]\n{content}"
                if i+1 < len(messages) and messages[i+1]["role"] == "user":
                    # Will be added to the next user message
                    pass
                else:
                    # Add as a separate message
                    conversation_parts.append(f"[INST] {func_result} [/INST]")
        
        return " ".join(conversation_parts)


class VicunaPromptFormatter(BasePromptFormatter):
    """
    Formatter for Vicuna models.
    """
    
    def format_messages(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        json_mode: bool = False
    ) -> str:
        """Format messages in Vicuna format."""
        formatted_parts = []
        system_content = ""
        
        # Extract system message
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
                break
        
        # Add system message
        if system_content:
            if functions:
                # Add function specifications to system message
                func_text = "\n\nFunctions available to call:\n"
                func_text += json.dumps(functions, indent=2)
                
                # Add function call preference
                if function_call != "auto":
                    if function_call == "none":
                        func_pref = "\n\nYou should not call functions."
                    elif isinstance(function_call, dict) and "name" in function_call:
                        func_pref = f"\n\nYou should specifically call the function named {function_call['name']}."
                    else:
                        func_pref = "\n\nYou can call functions if needed."
                    func_text += func_pref
                
                system_content += func_text
            
            # Add JSON mode instruction if requested
            if json_mode:
                system_content += "\n\nYou must respond with a valid JSON object or array, without any additional text."
                
            formatted_parts.append(f"SYSTEM: {system_content}")
        
        # Process conversation turns
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                # Already handled
                continue
            elif role == "user":
                formatted_parts.append(f"HUMAN: {content}")
            elif role == "assistant":
                if message.get("function_call"):
                    # Format function call as JSON
                    func_call = message["function_call"]
                    func_call_str = json.dumps(func_call)
                    formatted_parts.append(f"ASSISTANT: {func_call_str}")
                else:
                    formatted_parts.append(f"ASSISTANT: {content or ''}")
            elif role == "function":
                # Add function result
                func_result = f"FUNCTION {message.get('name')}: {content}"
                formatted_parts.append(func_result)
        
        # Add final assistant prompt if the last message wasn't from the assistant
        if messages and messages[-1]["role"] != "assistant":
            formatted_parts.append("ASSISTANT:")
            
        return "\n".join(formatted_parts)


class PlainInstructPromptFormatter(BasePromptFormatter):
    """
    Formatter for basic instruction-following models without specific chat templates.
    """
    
    def format_messages(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        json_mode: bool = False
    ) -> str:
        """Format messages in a simple instruction format."""
        formatted_parts = []
        system_content = ""
        
        # Extract system message
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
                break
        
        # Add system/instructions content
        if system_content:
            if functions:
                # Add function specifications to system message
                func_text = "\n\nFunctions available to call:\n"
                func_text += json.dumps(functions, indent=2)
                
                # Add function call preference
                if function_call != "auto":
                    if function_call == "none":
                        func_pref = "\n\nYou should not call functions."
                    elif isinstance(function_call, dict) and "name" in function_call:
                        func_pref = f"\n\nYou should specifically call the function named {function_call['name']}."
                    else:
                        func_pref = "\n\nYou can call functions if needed."
                    func_text += func_pref
                
                system_content += func_text
            
            # Add JSON mode instruction if requested
            if json_mode:
                system_content += "\n\nYou must respond with a valid JSON object or array, without any additional text."
                
            formatted_parts.append(f"Instructions: {system_content}\n")
        
        # Process conversation turns
        conversation = []
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                # Already handled
                continue
            elif role == "user":
                conversation.append(f"User: {content}")
            elif role == "assistant":
                if message.get("function_call"):
                    # Format function call as JSON
                    func_call = message["function_call"]
                    func_call_str = json.dumps(func_call)
                    conversation.append(f"Assistant: {func_call_str}")
                else:
                    conversation.append(f"Assistant: {content or ''}")
            elif role == "function":
                # Add function result
                conversation.append(f"Function {message.get('name')}: {content}")
        
        # Add conversation history
        if conversation:
            formatted_parts.append("Conversation:\n" + "\n".join(conversation))
            
        # Add final assistant prompt if the last message wasn't from the assistant
        if messages and messages[-1]["role"] != "assistant":
            formatted_parts.append("\nAssistant:")
            
        return "\n".join(formatted_parts)


def get_prompt_formatter(model_name: str) -> BasePromptFormatter:
    """
    Get the appropriate prompt formatter for a model.
    
    Args:
        model_name: Name or path of the model
        
    Returns:
        Prompt formatter instance
    """
    model_name_lower = model_name.lower()
    
    # Check for Llama 2 Chat models
    if any(x in model_name_lower for x in ["llama-2", "llama2", "llama_2"]) and "chat" in model_name_lower:
        return Llama2ChatPromptFormatter()
    
    # Check for Mistral Instruct models
    elif any(x in model_name_lower for x in ["mistral", "mixtral"]) and any(x in model_name_lower for x in ["instruct", "chat"]):
        return MistralInstructPromptFormatter()
    
    # Check for Vicuna models
    elif "vicuna" in model_name_lower:
        return VicunaPromptFormatter()
    
    # Check for ChatML format (Zephyr, etc.)
    elif any(x in model_name_lower for x in ["chatml", "chat-ml", "zephyr", "openchat"]):
        return ChatMLPromptFormatter()
    
    # Default to plain instruction format
    else:
        return PlainInstructPromptFormatter() 