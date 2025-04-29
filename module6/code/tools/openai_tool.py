"""
OpenAI Tool Implementation
-------------------------
This file contains the OpenAI tool for chat and text generation.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Union
import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .base_tool import BaseTool, ToolResponse

# Load environment variables
load_dotenv()

class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: str = Field(..., description="The role of the message sender (system, user, assistant)")
    content: str = Field(..., description="The content of the message")


class OpenAITool(BaseTool):
    """Tool for interacting with OpenAI API for chat and text generation."""
    
    def __init__(
        self,
        name: str = "openai",
        description: str = "Generate text or chat responses using OpenAI models",
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the OpenAI tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            api_key: OpenAI API key (if None, will try to get from environment)
            model: The model to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0-1)
            max_retries: Maximum number of retries on rate limit or error
            retry_delay: Initial delay between retries in seconds
        """
        super().__init__(name, description)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an argument or as OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = "https://api.openai.com/v1"
        
        # Add model info to metadata
        self.metadata["model"] = model
        self.metadata["max_tokens"] = max_tokens
        self.metadata["temperature"] = temperature
    
    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the OpenAI tool with the provided parameters.
        
        Args:
            **kwargs: Tool-specific parameters including:
                - prompt: Text prompt for completion (for non-chat)
                - messages: List of message dicts with role and content (for chat)
                - stream: Whether to stream the response
                - model: Override the default model
                - temperature: Override the default temperature
                - max_tokens: Override the default max_tokens
        
        Returns:
            ToolResponse: The result of the tool execution
        """
        try:
            # Extract parameters
            prompt = kwargs.get("prompt")
            messages = kwargs.get("messages")
            stream = kwargs.get("stream", False)
            model = kwargs.get("model", self.model)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            # Validate input
            if not prompt and not messages:
                return ToolResponse(
                    success=False,
                    error="Either 'prompt' or 'messages' must be provided"
                )
            
            # Convert prompt to messages format if provided
            if prompt and not messages:
                messages = [{"role": "user", "content": prompt}]
            
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # Make API call with retries
            response_data = self._make_api_call(
                f"{self.base_url}/chat/completions",
                headers,
                payload
            )
            
            # Process response
            if stream:
                # Return the response directly for streaming
                return ToolResponse(
                    success=True,
                    result=response_data,
                    metadata={"model": model, "streaming": True}
                )
            else:
                # Extract the response content
                try:
                    content = response_data["choices"][0]["message"]["content"]
                    usage = response_data.get("usage", {})
                    
                    return ToolResponse(
                        success=True,
                        result=content,
                        metadata={
                            "model": model,
                            "usage": usage,
                            "finish_reason": response_data["choices"][0].get("finish_reason")
                        }
                    )
                except (KeyError, IndexError) as e:
                    return ToolResponse(
                        success=False,
                        error=f"Failed to parse OpenAI response: {str(e)}",
                        result=response_data
                    )
                
        except Exception as e:
            return ToolResponse(
                success=False,
                error=f"OpenAI tool execution failed: {str(e)}"
            )
    
    def _make_api_call(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make an API call with exponential backoff retry logic.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                
                # Handle rate limiting
                if response.status_code == 429:
                    # Get retry-after header or use exponential backoff
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_time = int(retry_after)
                    else:
                        sleep_time = self.retry_delay * (2 ** attempt)
                    
                    print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                
                # Handle other errors
                error_msg = f"API error: {response.status_code} - {response.text}"
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    print(f"{error_msg}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise Exception(error_msg)
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    print(f"Request error: {str(e)}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                else:
                    raise Exception(f"Request failed after {self.max_retries} attempts: {str(e)}")
        
        raise Exception(f"API call failed after {self.max_retries} attempts")
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON Schema for this tool.
        
        Returns:
            Dict[str, Any]: The JSON Schema for this tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt for completion (for non-chat)"
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"],
                                    "description": "The role of the message sender"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content of the message"
                                }
                            },
                            "required": ["role", "content"]
                        },
                        "description": "List of message objects with role and content (for chat)"
                    },
                    "stream": {
                        "type": "boolean",
                        "description": "Whether to stream the response",
                        "default": False
                    },
                    "model": {
                        "type": "string",
                        "description": "Override the default model",
                        "default": self.model
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for generation (0-1)",
                        "default": self.temperature
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum number of tokens to generate",
                        "default": self.max_tokens
                    }
                },
                "oneOf": [
                    {"required": ["prompt"]},
                    {"required": ["messages"]}
                ]
            }
        }
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Convenience method for chat completion.
        
        Args:
            messages: List of message dicts with role and content
            **kwargs: Additional parameters to pass to execute()
            
        Returns:
            str: The generated response text
        """
        response = self.execute(messages=messages, **kwargs)
        if response.success:
            return response.result
        else:
            raise Exception(f"Chat failed: {response.error}")
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Convenience method for text completion.
        
        Args:
            prompt: Text prompt for completion
            **kwargs: Additional parameters to pass to execute()
            
        Returns:
            str: The generated completion text
        """
        response = self.execute(prompt=prompt, **kwargs)
        if response.success:
            return response.result
        else:
            raise Exception(f"Completion failed: {response.error}")


# Example usage
if __name__ == "__main__":
    # Create the tool
    openai_tool = OpenAITool()
    
    # Test chat completion
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = openai_tool.chat(messages)
        print("Chat Response:")
        print(response)
        print("-" * 50)
    except Exception as e:
        print(f"Chat error: {e}")
    
    # Test text completion
    try:
        prompt = "Write a short poem about artificial intelligence."
        response = openai_tool.complete(prompt)
        print("Completion Response:")
        print(response)
        print("-" * 50)
    except Exception as e:
        print(f"Completion error: {e}")
