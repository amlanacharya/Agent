"""
Test script for the OpenAI tool.
"""

import os
import sys
import unittest
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenAI tool
from module6.code.tools.openai_tool import OpenAITool, ToolResponse

# Load environment variables
load_dotenv()

class TestOpenAITool(unittest.TestCase):
    """Test cases for the OpenAI tool."""
    
    def setUp(self):
        """Set up the test environment."""
        # Check if OPENAI_API_KEY is available
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.skipTest("OPENAI_API_KEY environment variable not set")
        
        # Create the tool with a cheaper model
        self.openai_tool = OpenAITool(
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.3
        )
    
    def test_initialization(self):
        """Test that the tool initializes correctly."""
        self.assertEqual(self.openai_tool.name, "openai")
        self.assertEqual(self.openai_tool.model, "gpt-3.5-turbo")
        self.assertEqual(self.openai_tool.max_tokens, 100)
        self.assertEqual(self.openai_tool.temperature, 0.3)
    
    def test_get_schema(self):
        """Test that the tool returns a valid schema."""
        schema = self.openai_tool.get_schema()
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["name"], "openai")
        self.assertIn("parameters", schema)
        self.assertIn("properties", schema["parameters"])
        self.assertIn("prompt", schema["parameters"]["properties"])
        self.assertIn("messages", schema["parameters"]["properties"])
    
    def test_chat_completion(self):
        """Test chat completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        # Test using execute method
        response = self.openai_tool.execute(messages=messages)
        self.assertTrue(response.success)
        self.assertIsNotNone(response.result)
        self.assertIn("model", response.metadata)
        self.assertIn("usage", response.metadata)
        
        # Test using convenience method
        result = self.openai_tool.chat(messages)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_text_completion(self):
        """Test text completion."""
        prompt = "What is the capital of France?"
        
        # Test using execute method
        response = self.openai_tool.execute(prompt=prompt)
        self.assertTrue(response.success)
        self.assertIsNotNone(response.result)
        
        # Test using convenience method
        result = self.openai_tool.complete(prompt)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_invalid_input(self):
        """Test that the tool handles invalid input correctly."""
        # Test with no prompt or messages
        response = self.openai_tool.execute()
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("must be provided", response.error)


if __name__ == "__main__":
    unittest.main()
