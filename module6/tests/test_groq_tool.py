"""
Test script for the Groq tool.
"""

import os
import sys
import unittest
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Groq tool
from module6.code.tools.groq_tool import GroqTool, ToolResponse

# Load environment variables
load_dotenv()

class TestGroqTool(unittest.TestCase):
    """Test cases for the Groq tool."""
    
    def setUp(self):
        """Set up the test environment."""
        # Check if GROQ_API_KEY is available
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            self.skipTest("GROQ_API_KEY environment variable not set")
        
        # Create the tool
        self.groq_tool = GroqTool(
            model="llama3-8b-8192",
            max_tokens=100,
            temperature=0.3
        )
    
    def test_initialization(self):
        """Test that the tool initializes correctly."""
        self.assertEqual(self.groq_tool.name, "groq")
        self.assertEqual(self.groq_tool.model, "llama3-8b-8192")
        self.assertEqual(self.groq_tool.max_tokens, 100)
        self.assertEqual(self.groq_tool.temperature, 0.3)
    
    def test_get_schema(self):
        """Test that the tool returns a valid schema."""
        schema = self.groq_tool.get_schema()
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["name"], "groq")
        self.assertIn("parameters", schema)
        self.assertIn("properties", schema["parameters"])
        self.assertIn("prompt", schema["parameters"]["properties"])
        self.assertIn("messages", schema["parameters"]["properties"])
        self.assertIn("json_mode", schema["parameters"]["properties"])
    
    def test_chat_completion(self):
        """Test chat completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        # Test using execute method
        response = self.groq_tool.execute(messages=messages)
        self.assertTrue(response.success)
        self.assertIsNotNone(response.result)
        self.assertIn("model", response.metadata)
        self.assertIn("usage", response.metadata)
        
        # Test using convenience method
        result = self.groq_tool.chat(messages)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_text_completion(self):
        """Test text completion."""
        prompt = "What is the capital of France?"
        
        # Test using execute method
        response = self.groq_tool.execute(prompt=prompt)
        self.assertTrue(response.success)
        self.assertIsNotNone(response.result)
        
        # Test using convenience method
        result = self.groq_tool.complete(prompt)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_json_generation(self):
        """Test JSON generation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"}
            },
            "required": ["name", "age", "occupation"]
        }
        
        prompt = "Extract information about John Doe, a 35-year-old software engineer."
        
        # Test using generate_json method
        result = self.groq_tool.generate_json(prompt, schema)
        self.assertIsInstance(result, dict)
        self.assertIn("name", result)
        self.assertIn("age", result)
        self.assertIn("occupation", result)
    
    def test_invalid_input(self):
        """Test that the tool handles invalid input correctly."""
        # Test with no prompt or messages
        response = self.groq_tool.execute()
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("must be provided", response.error)


if __name__ == "__main__":
    unittest.main()
