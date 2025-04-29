"""
Alpha Vantage Tool Implementation
-------------------------------
This file contains a tool for fetching financial data using the Alpha Vantage API.
"""

from typing import Dict, Any, Optional, List
import os
import time
import json
import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .base_tool import BaseTool, ToolResponse

# Load environment variables
load_dotenv()

class AlphaVantageTool(BaseTool):
    """Tool for fetching financial data using the Alpha Vantage API."""

    def __init__(
        self,
        name: str = "alpha_vantage",
        description: str = "Get financial data including stock quotes and currency exchange rates",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the Alpha Vantage tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            api_key: Alpha Vantage API key (if None, will try to get from environment)
            max_retries: Maximum number of retries on rate limit or error
            retry_delay: Initial delay between retries in seconds
        """
        super().__init__(name, description)
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set it as an argument or as ALPHAVANTAGE_API_KEY environment variable.")

        # Clean the API key (remove any comments and quotes)
        if "#" in self.api_key:
            self.api_key = self.api_key.split("#")[0].strip()

        # Remove quotes if present
        self.api_key = self.api_key.strip('"\'').strip()

        # Print the API key for debugging
        print(f"Using Alpha Vantage API key: {self.api_key}")

        self.base_url = "https://www.alphavantage.co/query"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Add API info to metadata
        self.metadata["api_provider"] = "Alpha Vantage"

    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the Alpha Vantage tool with the provided parameters.

        Args:
            action: The action to perform (quote, exchange_rate, search)
            symbol: Stock symbol for quote action
            from_currency: From currency code for exchange_rate action
            to_currency: To currency code for exchange_rate action
            keywords: Keywords for search action

        Returns:
            ToolResponse: The result of the tool execution
        """
        try:
            # Check parameters
            action = kwargs.get("action", "").lower()

            if not action:
                return ToolResponse(
                    success=False,
                    error="Action must be provided (quote, exchange_rate, search)"
                )

            # Handle different actions
            if action == "quote":
                symbol = kwargs.get("symbol")
                if not symbol:
                    return ToolResponse(
                        success=False,
                        error="Symbol must be provided for quote action"
                    )

                result = self._get_quote_endpoint(symbol)
                return ToolResponse(
                    success=True,
                    result=result,
                    metadata={
                        "action": action,
                        "symbol": symbol
                    }
                )

            elif action == "exchange_rate":
                from_currency = kwargs.get("from_currency")
                to_currency = kwargs.get("to_currency")
                if not from_currency or not to_currency:
                    return ToolResponse(
                        success=False,
                        error="Both from_currency and to_currency must be provided for exchange_rate action"
                    )

                result = self._get_exchange_rate(from_currency, to_currency)
                return ToolResponse(
                    success=True,
                    result=result,
                    metadata={
                        "action": action,
                        "from_currency": from_currency,
                        "to_currency": to_currency
                    }
                )

            elif action == "search":
                keywords = kwargs.get("keywords")
                if not keywords:
                    return ToolResponse(
                        success=False,
                        error="Keywords must be provided for search action"
                    )

                result = self.search_symbols(keywords)
                return ToolResponse(
                    success=True,
                    result=result,
                    metadata={
                        "action": action,
                        "keywords": keywords
                    }
                )

            else:
                return ToolResponse(
                    success=False,
                    error=f"Unknown action: {action}. Supported actions are: quote, exchange_rate, search"
                )

        except ValueError as e:
            # Handle invalid parameters
            return ToolResponse(
                success=False,
                error=str(e)
            )

        except Exception as e:
            # Handle other errors
            return ToolResponse(
                success=False,
                error=f"Alpha Vantage tool execution failed: {str(e)}"
            )

    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Make API request with retry logic and rate limit handling.

        Args:
            params: Request parameters

        Returns:
            Dict[str, Any]: API response

        Raises:
            Exception: If all retries fail
        """
        # Add API key to params
        params["apikey"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                # Check for error messages in the response
                if "Error Message" in data:
                    raise ValueError(f"API error: {data['Error Message']}")

                if "Information" in data:
                    raise ValueError(f"API limit reached: {data['Information']}")

                if "Note" in data and "API call frequency" in data["Note"]:
                    print(f"Warning: {data['Note']}")

                return data

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit error
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # exponential backoff
                        print(f"Rate limit hit. Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue

                # If we've exhausted retries or it's not a rate limit error
                error_msg = f"API error: {e.response.status_code} - {e.response.text}"
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"{error_msg}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(error_msg)

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Request failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                raise Exception(f"Request failed after {self.max_retries} attempts: {str(e)}")

    def _get_quote_endpoint(self, symbol: str) -> Dict[str, Any]:
        """
        Get a stock quote for the given symbol.

        Args:
            symbol: Stock symbol (e.g., 'IBM', 'AAPL')

        Returns:
            Dict[str, Any]: Stock quote information

        Raises:
            ValueError: If the symbol is invalid or not found
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol
        }

        data = self._make_request(params)

        if "Global Quote" not in data or not data["Global Quote"]:
            raise ValueError(f"No quote data found for symbol: {symbol}")

        return data["Global Quote"]

    def _get_exchange_rate(self, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """
        Get the exchange rate between two currencies.

        Args:
            from_currency: From currency code (e.g., 'USD')
            to_currency: To currency code (e.g., 'JPY')

        Returns:
            Dict[str, Any]: Exchange rate information

        Raises:
            ValueError: If the currency codes are invalid or not found
        """
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency
        }

        data = self._make_request(params)

        if "Realtime Currency Exchange Rate" not in data:
            raise ValueError(f"No exchange rate data found for {from_currency} to {to_currency}")

        return data["Realtime Currency Exchange Rate"]

    def search_symbols(self, keywords: str) -> List[Dict[str, str]]:
        """
        Search for symbols matching the given keywords.

        Args:
            keywords: Search keywords

        Returns:
            List[Dict[str, str]]: List of matching symbols and their information

        Raises:
            ValueError: If no matches are found
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords
        }

        data = self._make_request(params)

        if "bestMatches" not in data or not data["bestMatches"]:
            raise ValueError(f"No matches found for keywords: {keywords}")

        return data["bestMatches"]

    def _get_time_series_daily(self, symbol: str) -> Dict[str, Any]:
        """
        Get daily time series data for the given symbol.

        Args:
            symbol: Stock symbol (e.g., 'IBM', 'AAPL')

        Returns:
            Dict[str, Any]: Time series data

        Raises:
            ValueError: If the symbol is invalid or not found
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol
        }

        data = self._make_request(params)

        if "Time Series (Daily)" not in data:
            raise ValueError(f"No time series data found for symbol: {symbol}")

        return data

    def _get_top_gainers_losers(self) -> Dict[str, Any]:
        """
        Get the top gainers, losers, and most active stocks.

        Returns:
            Dict[str, Any]: Market status information

        Raises:
            ValueError: If market status data is not available
        """
        params = {
            "function": "TOP_GAINERS_LOSERS"
        }

        data = self._make_request(params)

        if not data or ("top_gainers" not in data and "top_losers" not in data and "most_actively_traded" not in data):
            raise ValueError("Market status data not available")

        return data

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON Schema for this tool.

        Returns:
            Dict[str, Any]: The JSON Schema for this tool
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["quote", "exchange_rate", "search"],
                            "description": "The action to perform"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol for quote action (e.g., 'IBM', 'AAPL')"
                        },
                        "from_currency": {
                            "type": "string",
                            "description": "From currency code for exchange_rate action (e.g., 'USD')"
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "To currency code for exchange_rate action (e.g., 'JPY')"
                        },
                        "keywords": {
                            "type": "string",
                            "description": "Keywords for search action"
                        }
                    },
                    "required": ["action"],
                    "allOf": [
                        {
                            "if": {
                                "properties": {"action": {"enum": ["quote"]}},
                                "required": ["action"]
                            },
                            "then": {"required": ["symbol"]}
                        },
                        {
                            "if": {
                                "properties": {"action": {"enum": ["exchange_rate"]}},
                                "required": ["action"]
                            },
                            "then": {"required": ["from_currency", "to_currency"]}
                        },
                        {
                            "if": {
                                "properties": {"action": {"enum": ["search"]}},
                                "required": ["action"]
                            },
                            "then": {"required": ["keywords"]}
                        }
                    ]
                }
            }
        }


# Example usage
if __name__ == "__main__":
    # Create the tool
    alpha_vantage_tool = AlphaVantageTool()

    # Test stock quote
    try:
        response = alpha_vantage_tool.execute(action="quote", symbol="IBM")
        print("Stock Quote for IBM:")
        print(json.dumps(response.result, indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")

    # Test exchange rate
    try:
        response = alpha_vantage_tool.execute(action="exchange_rate", from_currency="USD", to_currency="JPY")
        print("Exchange Rate from USD to JPY:")
        print(json.dumps(response.result, indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")

    # Test symbol search
    try:
        response = alpha_vantage_tool.execute(action="search", keywords="Microsoft")
        print("Search Results for 'Microsoft':")
        print(json.dumps(response.result, indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
