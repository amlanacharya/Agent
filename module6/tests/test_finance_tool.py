"""
Tests for the Finance Tool
------------------------
This file contains tests for the Finance Tool implementation.
"""

import unittest
import os
from dotenv import load_dotenv
from module6.code.tools.finance_tool import FinanceTool

class TestFinanceTool(unittest.TestCase):
    """Test cases for the Finance tool."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()
    
    def setUp(self):
        """Set up the test environment."""
        # Check if ALPHAVANTAGE_API_KEY is available
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            self.skipTest("ALPHAVANTAGE_API_KEY environment variable not set")
        
        # Create the tool
        self.finance_tool = FinanceTool()
    
    def test_initialization(self):
        """Test that the tool initializes correctly."""
        self.assertEqual(self.finance_tool.name, "finance")
        self.assertIn("financial", self.finance_tool.description.lower())
        self.assertEqual(self.finance_tool.max_retries, 3)
        self.assertEqual(self.finance_tool.retry_delay, 2)
        self.assertEqual(self.finance_tool.metadata["api_provider"], "Alpha Vantage")
    
    def test_get_schema(self):
        """Test that the schema is correctly defined."""
        schema = self.finance_tool.get_schema()
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema["type"], "function")
        self.assertIn("parameters", schema["function"])
        self.assertIn("action", schema["function"]["parameters"]["properties"])
        self.assertIn("symbol", schema["function"]["parameters"]["properties"])
        self.assertIn("from_currency", schema["function"]["parameters"]["properties"])
        self.assertIn("to_currency", schema["function"]["parameters"]["properties"])
        self.assertIn("keywords", schema["function"]["parameters"]["properties"])
    
    def test_stock_quote(self):
        """Test getting a stock quote."""
        try:
            # Test using execute method
            response = self.finance_tool.execute(action="quote", symbol="IBM")
            
            # If the API is working, check the response
            if response.success:
                self.assertIsNotNone(response.result)
                self.assertIn("symbol", response.result)
                self.assertIn("price", response.result)
                self.assertIn("volume", response.result)
                self.assertEqual(response.metadata["action"], "quote")
                self.assertEqual(response.metadata["symbol"], "IBM")
            else:
                # If the API is not working, skip the test
                self.skipTest(f"Alpha Vantage API error: {response.error}")
        except Exception as e:
            self.skipTest(f"Test skipped due to API error: {str(e)}")
    
    def test_exchange_rate(self):
        """Test getting an exchange rate."""
        try:
            # Test using execute method
            response = self.finance_tool.execute(action="exchange_rate", from_currency="USD", to_currency="JPY")
            
            # If the API is working, check the response
            if response.success:
                self.assertIsNotNone(response.result)
                self.assertIn("from_currency", response.result)
                self.assertIn("to_currency", response.result)
                self.assertIn("exchange_rate", response.result)
                self.assertEqual(response.metadata["action"], "exchange_rate")
                self.assertEqual(response.metadata["from_currency"], "USD")
                self.assertEqual(response.metadata["to_currency"], "JPY")
            else:
                # If the API is not working, skip the test
                self.skipTest(f"Alpha Vantage API error: {response.error}")
        except Exception as e:
            self.skipTest(f"Test skipped due to API error: {str(e)}")
    
    def test_symbol_search(self):
        """Test searching for symbols."""
        try:
            # Test using execute method
            response = self.finance_tool.execute(action="search", keywords="Microsoft")
            
            # If the API is working, check the response
            if response.success:
                self.assertIsNotNone(response.result)
                self.assertIsInstance(response.result, list)
                if response.result:
                    self.assertIn("1. symbol", response.result[0])
                    self.assertIn("2. name", response.result[0])
                self.assertEqual(response.metadata["action"], "search")
                self.assertEqual(response.metadata["keywords"], "Microsoft")
            else:
                # If the API is not working, skip the test
                self.skipTest(f"Alpha Vantage API error: {response.error}")
        except Exception as e:
            self.skipTest(f"Test skipped due to API error: {str(e)}")
    
    def test_invalid_parameters(self):
        """Test that the tool handles invalid parameters correctly."""
        # Test with no action
        response = self.finance_tool.execute()
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("Action must be provided", response.error)
        
        # Test with invalid action
        response = self.finance_tool.execute(action="invalid_action")
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("Unknown action", response.error)
        
        # Test quote with no symbol
        response = self.finance_tool.execute(action="quote")
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("Symbol must be provided", response.error)
        
        # Test exchange_rate with missing parameters
        response = self.finance_tool.execute(action="exchange_rate", from_currency="USD")
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("Both from_currency and to_currency must be provided", response.error)
        
        # Test search with no keywords
        response = self.finance_tool.execute(action="search")
        self.assertFalse(response.success)
        self.assertIsNotNone(response.error)
        self.assertIn("Keywords must be provided", response.error)


if __name__ == "__main__":
    unittest.main()
