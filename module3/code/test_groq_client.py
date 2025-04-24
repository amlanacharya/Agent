"""
Tests for Groq Client
---------------------
This module contains tests for the groq_client module.
"""

import unittest
import os
import json
from unittest.mock import patch, MagicMock

# Try to import GroqClient
try:
    # Try module path first
    from module3.code.groq_client import GroqClient
    HAS_GROQ_CLIENT = True
except ImportError:
    try:
        # Try local import when running from module3/code directory
        from groq_client import GroqClient
        HAS_GROQ_CLIENT = True
    except ImportError:
        HAS_GROQ_CLIENT = False


@unittest.skipIf(not HAS_GROQ_CLIENT, "GroqClient not available")
class TestGroqClient(unittest.TestCase):
    """Test cases for groq_client module."""

    def setUp(self):
        """Set up test environment."""
        # Check if GROQ_API_KEY is available
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.skipTest("GROQ_API_KEY environment variable not set")

    def test_client_initialization(self):
        """Test GroqClient initialization."""
        # Test with API key from environment
        client = GroqClient()
        self.assertIsNotNone(client.api_key)
        self.assertEqual(client.base_url, "https://api.groq.com/openai/v1")

        # Test with explicit API key
        client = GroqClient(api_key="test_key")
        self.assertEqual(client.api_key, "test_key")

    @patch('requests.post')
    def test_generate_text(self, mock_post):
        """Test generate_text method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Create client and call generate_text
        client = GroqClient(api_key="test_key")
        response = client.generate_text("Test prompt")

        # Check response
        self.assertEqual(
            client.extract_text_from_response(response),
            "This is a test response"
        )

        # Check that post was called with correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test_key")
        self.assertEqual(json.loads(kwargs["data"])["messages"][0]["content"], "Test prompt")

    @patch('requests.post')
    def test_chat_completion(self, mock_post):
        """Test chat_completion method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test response"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Create client and call chat_completion
        client = GroqClient(api_key="test_key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        response = client.chat_completion(messages)

        # Check response
        self.assertEqual(
            client.extract_text_from_response(response),
            "This is a test response"
        )

        # Check that post was called with correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test_key")
        self.assertEqual(json.loads(kwargs["data"])["messages"], messages)

    @patch('requests.post')
    def test_generate_structured_output(self, mock_post):
        """Test generate_structured_output method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"name": "John", "age": 30}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Create client and call generate_structured_output
        client = GroqClient(api_key="test_key")
        output = client.generate_structured_output(
            prompt="Extract information about John",
            format_instructions="Provide output in JSON format"
        )

        # Check output
        self.assertEqual(output, '{"name": "John", "age": 30}')

        # Check that post was called with correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("Extract information about John", json.loads(kwargs["data"])["messages"][0]["content"])
        self.assertIn("Provide output in JSON format", json.loads(kwargs["data"])["messages"][0]["content"])

    @patch('requests.post')
    def test_generate_json_output(self, mock_post):
        """Test generate_json_output method."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"name": "John", "age": 30}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Create client and call generate_json_output
        client = GroqClient(api_key="test_key")
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        output = client.generate_json_output(
            prompt="Extract information about John",
            schema=schema
        )

        # Check output
        self.assertEqual(output, {"name": "John", "age": 30})

        # Check that post was called with correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("Extract information about John", json.loads(kwargs["data"])["messages"][0]["content"])
        self.assertIn("schema", json.loads(kwargs["data"])["messages"][0]["content"])

    @patch('requests.post')
    def test_error_handling(self, mock_post):
        """Test error handling."""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Error message"
        mock_post.return_value = mock_response

        # Create client and call generate_text
        client = GroqClient(api_key="test_key")

        # Check that exception is raised
        with self.assertRaises(Exception):
            client.generate_text("Test prompt")


# Integration tests that actually call the API
# These are skipped by default to avoid API costs
@unittest.skipIf(not HAS_GROQ_CLIENT or not os.getenv("RUN_INTEGRATION_TESTS"), "Integration tests disabled")
class TestGroqClientIntegration(unittest.TestCase):
    """Integration tests for groq_client module."""

    def setUp(self):
        """Set up test environment."""
        # Check if GROQ_API_KEY is available
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.skipTest("GROQ_API_KEY environment variable not set")

        # Create client
        self.client = GroqClient()

    def test_generate_text(self):
        """Test generate_text with actual API call."""
        response = self.client.generate_text("What is 2+2?")
        text = self.client.extract_text_from_response(response)

        # Check that we got a response
        self.assertTrue(len(text) > 0)

    def test_generate_json_output(self):
        """Test generate_json_output with actual API call."""
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "integer"}
            }
        }

        output = self.client.generate_json_output(
            prompt="What is 2+2?",
            schema=schema
        )

        # Check that we got a valid response
        self.assertIsInstance(output, dict)
        self.assertIn("result", output)


if __name__ == "__main__":
    unittest.main()
