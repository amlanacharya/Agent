"""
Groq API Integration Utilities
----------------------------
This file contains utilities for integrating with the Groq API for both
text generation and embeddings.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
import requests
from dotenv import load_dotenv

# Try to import sentence_transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

# Load environment variables from .env file
load_dotenv()

class GroqClient:
    """
    Client for interacting with the Groq API for text generation and embeddings.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq client

        Args:
            api_key (str, optional): Groq API key. If not provided, will look for GROQ_API_KEY in environment
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

        # Default model settings
        self.default_model = "llama3-8b-8192"  # Default model for text generation
        # Groq doesn't currently have a dedicated embedding model, so we'll use the same model for embeddings
        # or implement a fallback method
        self.default_embedding_model = "llama3-8b-8192"  # Default model for embeddings

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

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"Error generating text: {response.text}")

        return response.json()

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Get embeddings for text using SentenceTransformers or fallback method

        Args:
            texts (str or list): Text or list of texts to get embeddings for
            model (str, optional): The embedding model to use (for SentenceTransformers)

        Returns:
            list: List of embedding vectors
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        # Try to use SentenceTransformers if available
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                # Use a default model if none specified
                model_name = model or 'all-MiniLM-L6-v2'  # Small, fast model with good performance

                # Load the model (will download if not cached)
                sentence_model = SentenceTransformer(model_name)

                # Generate embeddings
                embeddings = sentence_model.encode(texts, convert_to_numpy=True)

                # Convert numpy arrays to lists for JSON serialization
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                print(f"Warning: SentenceTransformers error: {e}")
                # Fall back to hash-based method
                return [self._fallback_embedding(text) for text in texts]

        # If SentenceTransformers is not available, try Groq API
        try:
            # Try to use OpenAI-compatible embeddings endpoint
            api_model = model or self.default_embedding_model

            payload = {
                "model": api_model,
                "input": texts
            }

            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return [item["embedding"] for item in result["data"]]
            else:
                print(f"Warning: Embedding API failed: {response.text}")
                # Fall back to our hash-based method
                return [self._fallback_embedding(text) for text in texts]

        except Exception as e:
            print(f"Warning: Embedding API error: {e}")
            # Fall back to a simple hash-based embedding
            return [self._fallback_embedding(text) for text in texts]

    def _fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Create a fallback embedding based on hash values

        Args:
            text (str): Text to create embedding for
            dim (int): Dimension of the embedding vector

        Returns:
            list: Embedding vector
        """
        # Create a simple hash-based embedding (not for production use)
        import hashlib

        # Initialize a vector of zeros
        vector = [0.0] * dim

        # Use words to influence different dimensions
        words = text.lower().split()
        for i, word in enumerate(words):
            # Hash the word
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)

            # Use the hash to set values in the vector
            for j in range(min(10, len(word))):
                idx = (hash_value + j) % dim
                vector[idx] = 0.1 * ((hash_value % 20) - 10) + vector[idx]

        # Normalize the vector
        magnitude = sum(x**2 for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector

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

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"Error in chat completion: {response.text}")

        return response.json()

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


# Example usage
if __name__ == "__main__":
    # Create a client
    client = GroqClient()

    # Test text generation
    try:
        response = client.generate_text("Explain what a vector database is in simple terms.")
        print("Text Generation Response:")
        print(client.extract_text_from_response(response))
        print("-" * 50)
    except Exception as e:
        print(f"Text generation error: {e}")

    # Test embeddings
    try:
        texts = ["What is a vector database?", "How do embeddings work?"]
        embeddings = client.get_embeddings(texts)
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {len(embeddings[0])}")
        print("-" * 50)
    except Exception as e:
        print(f"Embedding error: {e}")

    # Test chat completion
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that explains technical concepts simply."},
            {"role": "user", "content": "What is the difference between short-term and long-term memory in AI agents?"}
        ]
        response = client.chat_completion(messages)
        print("Chat Completion Response:")
        print(client.extract_text_from_response(response))
    except Exception as e:
        print(f"Chat completion error: {e}")
