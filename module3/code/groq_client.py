"""
Groq API Integration Utilities for Module 3
----------------------------------------
This file contains utilities for integrating with the Groq API for text generation
to support structured output parsing and validation.
"""

import os
import json
import time
import random
import requests
from typing import List, Dict, Any, Optional, Union, Callable
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retry_on_status_codes: List[int] = [429, 500, 502, 503, 504]
) -> Callable:
    """
    Decorator that retries a function with exponential backoff when specific exceptions occur.

    Args:
        func: The function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases
        retry_on_status_codes: HTTP status codes to retry on

    Returns:
        Wrapped function with retry logic
    """
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


class GroqClient:
    """
    Client for interacting with the Groq API for text generation.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 5):
        """
        Initialize the Groq client

        Args:
            api_key (str, optional): Groq API key. If not provided, will look for GROQ_API_KEY in environment
            max_retries (int): Maximum number of retries for API calls
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("WARNING: Groq API key not found. Please provide it as an argument or set the GROQ_API_KEY environment variable.")
            print("Current environment variables:", list(os.environ.keys()))
            raise ValueError(
                "Groq API key not found. Please provide it as an argument or set the GROQ_API_KEY environment variable."
            )

        # Updated base URL based on Groq documentation
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Retry settings
        self.max_retries = max_retries

        # Default model settings
        self.default_model = "llama3-8b-8192"  # Default model for text generation

    def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using the Groq API

        Args:
            prompt (str): The prompt to generate text from
            model (str, optional): The model to use. Defaults to self.default_model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0.0 to 1.0)
            top_p (float): Controls diversity via nucleus sampling
            stream (bool): Whether to stream the response

        Returns:
            dict: The API response
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        @retry_with_exponential_backoff(max_retries=self.max_retries)
        def _make_api_call():
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                # Check if this is a rate limit error (429)
                if response.status_code == 429:
                    # This will be caught by the retry decorator
                    print(f"Rate limit exceeded. Response: {response.text}")
                    return response
                # For other errors, raise exception
                raise Exception(f"Error generating text: {response.text}")

            return response

        response = _make_api_call()

        # If we got a response object (not JSON), check status and convert
        if hasattr(response, 'json'):
            if response.status_code != 200:
                raise Exception(f"Error generating text after retries: {response.text}")
            return response.json()

        return response

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using the Groq API

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            model (str, optional): The model to use. Defaults to self.default_model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0.0 to 1.0)
            top_p (float): Controls diversity via nucleus sampling
            stream (bool): Whether to stream the response

        Returns:
            dict: The API response
        """
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }

        @retry_with_exponential_backoff(max_retries=self.max_retries)
        def _make_api_call():
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                # Check if this is a rate limit error (429)
                if response.status_code == 429:
                    # This will be caught by the retry decorator
                    print(f"Rate limit exceeded. Response: {response.text}")
                    return response
                # For other errors, raise exception
                raise Exception(f"Error in chat completion: {response.text}")

            return response

        response = _make_api_call()

        # If we got a response object (not JSON), check status and convert
        if hasattr(response, 'json'):
            if response.status_code != 200:
                raise Exception(f"Error in chat completion after retries: {response.text}")
            return response.json()

        return response

    def extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """
        Extract the generated text from a Groq API response

        Args:
            response (dict): The API response

        Returns:
            str: The generated text
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""

    def generate_structured_output(
        self,
        prompt: str,
        format_instructions: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,  # Lower temperature for more deterministic outputs
    ) -> str:
        """
        Generate structured output using the Groq API

        Args:
            prompt (str): The prompt to generate text from
            format_instructions (str): Instructions for formatting the output
            model (str, optional): The model to use. Defaults to self.default_model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0.0 to 1.0)

        Returns:
            str: The generated structured output
        """
        full_prompt = f"{prompt}\n\n{format_instructions}"

        response = self.generate_text(
            prompt=full_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return self.extract_text_from_response(response)

    def generate_json_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,  # Lower temperature for more deterministic outputs
    ) -> Dict[str, Any]:
        """
        Generate JSON output using the Groq API

        Args:
            prompt (str): The prompt to generate text from
            schema (dict): JSON schema for the expected output
            model (str, optional): The model to use. Defaults to self.default_model
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Controls randomness (0.0 to 1.0)

        Returns:
            dict: The generated JSON output
        """
        schema_str = json.dumps(schema, indent=2)

        format_instructions = f"""
        Please provide your response in valid JSON format according to the following schema:

        {schema_str}

        Ensure that your response is properly formatted as JSON and follows the schema exactly.
        """

        output_text = self.generate_structured_output(
            prompt=prompt,
            format_instructions=format_instructions,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Try to extract JSON from the output
        try:
            # Look for JSON-like structure
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = output_text[start_idx:end_idx+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON errors
            try:
                # Replace single quotes with double quotes
                fixed_json = output_text.replace("'", '"')
                return json.loads(fixed_json)
            except:
                pass

        # If all parsing attempts fail
        raise ValueError(f"Could not parse LLM output as JSON: {output_text}")


# Example usage
if __name__ == "__main__":
    # Create a client
    try:
        client = GroqClient()

        # Test text generation
        response = client.generate_text("Explain what structured output parsing is in simple terms.")
        print("Text Generation Response:")
        print(client.extract_text_from_response(response))
        print("-" * 50)

        # Test structured output generation
        person_schema = {
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

        json_output = client.generate_json_output(prompt, person_schema)
        print("JSON Output:")
        print(json.dumps(json_output, indent=2))

    except Exception as e:
        print(f"Error: {e}")
