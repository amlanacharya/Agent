"""
Test script for the Search tool.
"""

import os
import sys
import unittest
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Search tool
from module6.code.tools.search_tool import SearchTool, SearchResult, ToolResponse

# Load environment variables
load_dotenv()

class TestSearchTool(unittest.TestCase):
    """Test cases for the Search tool."""

    def setUp(self):
        """Set up the test environment."""
        # Check if SERPER_API_KEY is available
        self.serper_api_key = os.getenv("SERPER_API_KEY")

        # Create the tool
        self.search_tool = SearchTool(
            max_results=3,
            use_fallback=True
        )

    def test_initialization(self):
        """Test that the tool initializes correctly."""
        self.assertEqual(self.search_tool.name, "search")
        self.assertEqual(self.search_tool.max_results, 3)
        self.assertEqual(self.search_tool.use_fallback, True)
        self.assertEqual(self.search_tool.metadata["max_results"], 3)
        self.assertEqual(self.search_tool.metadata["has_serper_key"], bool(self.serper_api_key))

    def test_get_schema(self):
        """Test that the tool returns a valid schema."""
        schema = self.search_tool.get_schema()
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["name"], "search")
        self.assertIn("parameters", schema)
        self.assertIn("properties", schema["parameters"])
        self.assertIn("query", schema["parameters"]["properties"])
        self.assertIn("num_results", schema["parameters"]["properties"])

    def test_search_with_serper(self):
        """Test search with Serper API."""
        # Skip if no Serper API key
        if not self.serper_api_key:
            self.skipTest("SERPER_API_KEY environment variable not set")

        try:
            # Test using execute method
            response = self.search_tool.execute(query="Python programming language")

            # If the API key is invalid or expired, skip the test
            if not response.success and "Unauthorized" in str(response.error):
                self.skipTest("Serper API key is invalid or expired")

            self.assertTrue(response.success)
            self.assertIsNotNone(response.result)
            self.assertIsInstance(response.result, list)
            self.assertGreater(len(response.result), 0)
            self.assertIn("query", response.metadata)

            # Test using convenience method
            results = self.search_tool.search("Python programming language")
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)

            # Check result structure
            result = results[0]
            self.assertIsInstance(result, SearchResult)
            self.assertTrue(hasattr(result, "title"))
            self.assertTrue(hasattr(result, "link"))
            self.assertTrue(hasattr(result, "source"))
        except Exception as e:
            # If there's an API error, skip the test
            if "Serper API error" in str(e) or "Unauthorized" in str(e):
                self.skipTest(f"Serper API error: {str(e)}")
            else:
                raise

    def test_search_with_fallback(self):
        """Test search with fallback to DuckDuckGo."""
        # Test forcing fallback
        response = self.search_tool.execute(
            query="Python programming language",
            force_fallback=True
        )

        # If the test fails, it might be because DuckDuckGo is blocking requests
        # In that case, we'll skip the test
        if not response.success and "DuckDuckGo" in response.error:
            self.skipTest("DuckDuckGo is blocking requests")

        self.assertTrue(response.success)
        self.assertIsNotNone(response.result)
        self.assertIsInstance(response.result, list)
        self.assertIn("used_fallback", response.metadata)
        self.assertTrue(response.metadata["used_fallback"])

    def test_invalid_input(self):
        """Test that the tool handles invalid input correctly."""
        # Test with no query
        response = self.search_tool.execute()
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("query must be provided", response.error)


if __name__ == "__main__":
    unittest.main()
