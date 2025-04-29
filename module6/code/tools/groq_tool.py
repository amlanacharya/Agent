"""
Groq Tool Implementation
-----------------------
This file contains the Groq tool for chat and text generation.
"""

import os
import json
import time
import random
from typing import Any, Dict, List, Optional, Union, Callable
import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .base_tool import BaseTool, ToolResponse

# Load environment variables
load_dotenv()

def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retry_on_status_codes: List[int] = [429, 500, 502, 503, 504]
):
    """
    Decorator that retries a function with exponential backoff when specific exceptions occur.

    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases
        retry_on_status_codes: HTTP status codes to retry on

    Returns:
        Wrapped function with retry logic
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for retry in range(max_retries + 1):
                try:
                    response = func(*args, **kwargs)

                    # Check if this is a requests Response object with status code
                    if hasattr(response, 'status_code') and response.status_code in retry_on_status_codes:
                        if retry >= max_retries:
                            return response  # Return the response even with error status on last retry

                        # Get retry-after header if available
                        retry_after = response.headers.get('Retry-After')
                        if retry_after and retry_after.isdigit():
                            delay = min(float(retry_after), max_delay)

                        print(f"Rate limit hit (status {response.status_code}). Retrying in {delay:.2f} seconds...")
                    else:
                        return response  # Successful response

                except Exception as e:
                    last_exception = e
                    if retry >= max_retries:
                        raise

                    print(f"Error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry+1}/{max_retries})")

                # Add jitter to avoid thundering herd problem
                jitter = random.uniform(0, 0.1 * delay)
                time.sleep(delay + jitter)

                # Increase delay for next retry
                delay = min(delay * backoff_factor, max_delay)

            # If we get here, all retries failed
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: str = Field(..., description="The role of the message sender (system, user, assistant)")
    content: str = Field(..., description="The content of the message")


class GroqTool(BaseTool):
    """Tool for interacting with Groq API for chat and text generation."""
    
    def __init__(
        self,
        name: str = "groq",
        description: str = "Generate text or chat responses using Groq models",
        api_key: Optional[str] = None,
        model: str = "llama3-8b-8192",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the Groq tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            api_key: Groq API key (if None, will try to get from environment)
            model: The model to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0-1)
            max_retries: Maximum number of retries on rate limit or error
            retry_delay: Initial delay between retries in seconds
        """
        super().__init__(name, description)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set it as an argument or as GROQ_API_KEY environment variable.")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Add model info to metadata
        self.metadata["model"] = model
        self.metadata["max_tokens"] = max_tokens
        self.metadata["temperature"] = temperature
    
    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the Groq tool with the provided parameters.
        
        Args:
            **kwargs: Tool-specific parameters including:
                - prompt: Text prompt for completion (for non-chat)
                - messages: List of message dicts with role and content (for chat)
                - stream: Whether to stream the response
                - model: Override the default model
                - temperature: Override the default temperature
                - max_tokens: Override the default max_tokens
                - json_mode: Whether to request JSON output
                - json_schema: Schema for JSON output (if json_mode is True)
        
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
            json_mode = kwargs.get("json_mode", False)
            json_schema = kwargs.get("json_schema")
            
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
            
            # Add JSON mode if requested
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
                if json_schema:
                    # Add schema to system message
                    schema_message = {
                        "role": "system", 
                        "content": f"You must respond with a JSON object that conforms to the following schema:\n{json.dumps(json_schema, indent=2)}"
                    }
                    
                    # Add schema message at the beginning if no system message exists
                    if not any(msg.get("role") == "system" for msg in messages):
                        messages.insert(0, schema_message)
                    # Otherwise, append schema to existing system message
                    else:
                        for msg in messages:
                            if msg.get("role") == "system":
                                msg["content"] += f"\n\n{schema_message['content']}"
                                break
                    
                    payload["messages"] = messages
            
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
                    
                    # Parse JSON if in json_mode
                    if json_mode and content:
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            return ToolResponse(
                                success=False,
                                error="Failed to parse JSON response",
                                result=content
                            )
                    
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
                        error=f"Failed to parse Groq response: {str(e)}",
                        result=response_data
                    )
                
        except Exception as e:
            return ToolResponse(
                success=False,
                error=f"Groq tool execution failed: {str(e)}"
            )
    
    @retry_with_exponential_backoff(max_retries=5, initial_delay=1.0)
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
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            # This will be caught by the retry decorator if it's a retryable status code
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        return response.json()
    
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
                    },
                    "json_mode": {
                        "type": "boolean",
                        "description": "Whether to request JSON output",
                        "default": False
                    },
                    "json_schema": {
                        "type": "object",
                        "description": "Schema for JSON output (if json_mode is True)"
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
    
    def generate_json(self, prompt_or_messages: Union[str, List[Dict[str, str]]], schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Convenience method for generating JSON output.
        
        Args:
            prompt_or_messages: Text prompt or list of messages
            schema: JSON schema for the output (optional)
            **kwargs: Additional parameters to pass to execute()
            
        Returns:
            Dict[str, Any]: The generated JSON output
        """
        # Set up parameters for JSON generation
        params = {
            "json_mode": True,
            "json_schema": schema,
            "temperature": kwargs.pop("temperature", 0.2),  # Lower temperature for more deterministic JSON
            **kwargs
        }
        
        # Determine if we're using prompt or messages
        if isinstance(prompt_or_messages, str):
            params["prompt"] = prompt_or_messages
        else:
            params["messages"] = prompt_or_messages
        
        # Execute the request
        response = self.execute(**params)
        
        if response.success:
            return response.result
        else:
            raise Exception(f"JSON generation failed: {response.error}")


# Example usage
if __name__ == "__main__":
    # Create the tool
    groq_tool = GroqTool()
    
    # Test chat completion
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = groq_tool.chat(messages)
        print("Chat Response:")
        print(response)
        print("-" * 50)
    except Exception as e:
        print(f"Chat error: {e}")
    
    # Test text completion
    try:
        prompt = "Write a short poem about artificial intelligence."
        response = groq_tool.complete(prompt)
        print("Completion Response:")
        print(response)
        print("-" * 50)
    except Exception as e:
        print(f"Completion error: {e}")
    
    # Test JSON generation
    try:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age", "occupation", "skills"]
        }
        
        prompt = "Extract information about John Doe, a 35-year-old software engineer who knows Python, JavaScript, and SQL."
        
        response = groq_tool.generate_json(prompt, schema)
        print("JSON Response:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"JSON generation error: {e}")
