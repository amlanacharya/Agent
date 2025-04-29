"""
Finance Tool Implementation
-------------------------
This file contains a tool for fetching financial data using the Alpha Vantage API.
"""

from typing import Dict, Any, Optional, Union, List
import os
import time
import json
import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .base_tool import BaseTool, ToolResponse

# Check if LangChain is available
try:
    from langchain.tools import BaseTool as LangChainBaseTool
    from langchain.tools import Tool as LangChainTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Load environment variables
load_dotenv()

class StockQuote(BaseModel):
    """Model for stock quotes."""
    symbol: str = Field(..., description="Stock symbol")
    price: float = Field(..., description="Current price")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    volume: int = Field(..., description="Trading volume")
    latest_trading_day: str = Field(..., description="Latest trading day")
    previous_close: float = Field(..., description="Previous close price")
    change: float = Field(..., description="Price change")
    change_percent: str = Field(..., description="Price change percentage")

class ExchangeRate(BaseModel):
    """Model for currency exchange rates."""
    from_currency: str = Field(..., description="From currency code")
    from_currency_name: str = Field(..., description="From currency name")
    to_currency: str = Field(..., description="To currency code")
    to_currency_name: str = Field(..., description="To currency name")
    exchange_rate: float = Field(..., description="Exchange rate")
    last_refreshed: str = Field(..., description="Last refreshed time")
    time_zone: str = Field(..., description="Time zone")
    bid_price: Optional[float] = Field(None, description="Bid price")
    ask_price: Optional[float] = Field(None, description="Ask price")


class FinanceTool(BaseTool):
    """Tool for fetching financial data using the Alpha Vantage API."""

    def __init__(
        self,
        name: str = "finance",
        description: str = "Get financial data including stock quotes and currency exchange rates",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the Finance tool.

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

        self.base_url = "https://www.alphavantage.co/query"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Add API info to metadata
        self.metadata["api_provider"] = "Alpha Vantage"

    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the finance tool with the provided parameters.

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
                result = self.get_stock_quote(symbol)
                return ToolResponse(
                    success=True,
                    result=result.model_dump() if isinstance(result, BaseModel) else result,
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
                result = self.get_exchange_rate(from_currency, to_currency)
                return ToolResponse(
                    success=True,
                    result=result.model_dump() if isinstance(result, BaseModel) else result,
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

            elif action == "market_status":
                result = self.get_market_status()
                return ToolResponse(
                    success=True,
                    result=result,
                    metadata={
                        "action": action
                    }
                )

            else:
                return ToolResponse(
                    success=False,
                    error=f"Unknown action: {action}. Supported actions are: quote, exchange_rate, search, market_status"
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
                error=f"Finance tool execution failed: {str(e)}"
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

    def get_stock_quote(self, symbol: str) -> StockQuote:
        """
        Get a stock quote for the given symbol.

        Args:
            symbol: Stock symbol (e.g., 'IBM', 'AAPL')

        Returns:
            StockQuote: Stock quote information

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

        quote = data["Global Quote"]

        return StockQuote(
            symbol=quote["01. symbol"],
            price=float(quote["05. price"]),
            open=float(quote["02. open"]),
            high=float(quote["03. high"]),
            low=float(quote["04. low"]),
            volume=int(quote["06. volume"]),
            latest_trading_day=quote["07. latest trading day"],
            previous_close=float(quote["08. previous close"]),
            change=float(quote["09. change"]),
            change_percent=quote["10. change percent"]
        )

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> ExchangeRate:
        """
        Get the exchange rate between two currencies.

        Args:
            from_currency: From currency code (e.g., 'USD')
            to_currency: To currency code (e.g., 'JPY')

        Returns:
            ExchangeRate: Exchange rate information

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

        rate_data = data["Realtime Currency Exchange Rate"]

        return ExchangeRate(
            from_currency=rate_data["1. From_Currency Code"],
            from_currency_name=rate_data["2. From_Currency Name"],
            to_currency=rate_data["3. To_Currency Code"],
            to_currency_name=rate_data["4. To_Currency Name"],
            exchange_rate=float(rate_data["5. Exchange Rate"]),
            last_refreshed=rate_data["6. Last Refreshed"],
            time_zone=rate_data["7. Time Zone"],
            bid_price=float(rate_data["8. Bid Price"]) if "8. Bid Price" in rate_data else None,
            ask_price=float(rate_data["9. Ask Price"]) if "9. Ask Price" in rate_data else None
        )

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

    def get_market_status(self) -> Dict[str, Any]:
        """
        Get the current market status (top gainers, losers, most active).

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
                            "enum": ["quote", "exchange_rate", "search", "market_status"],
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

    def to_langchain_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Any:
        """
        Convert this tool to a LangChain tool.

        Args:
            name: Optional name for the LangChain tool
            description: Optional description for the LangChain tool

        Returns:
            A LangChain tool

        Raises:
            ImportError: If LangChain is not installed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install it with 'pip install langchain'"
            )

        # Create a custom LangChain tool
        try:
            from langchain.tools import Tool as LangChainTool

            def run_finance_tool(query: str) -> str:
                """Run the finance tool with the given query."""
                try:
                    # Parse the query to extract parameters
                    # First, check if the query is just a stock symbol
                    query_stripped = query.strip(",.?!()[]{}'\"-")
                    if query_stripped.isupper() and 1 <= len(query_stripped) <= 5:
                        # The query is likely just a stock symbol
                        response = self.execute(action="quote", symbol=query_stripped)
                        if response.success:
                            result = response.result
                            return f"Stock quote for {query_stripped}: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                    # Otherwise, check for stock-related keywords
                    if "stock" in query.lower() or "quote" in query.lower() or "price" in query.lower():
                        # Extract symbol - look for common stock symbols
                        words = query.split()
                        symbols = []

                        # Look for uppercase words that might be stock symbols
                        for word in words:
                            # Clean the word of punctuation
                            clean_word = word.strip(",.?!()[]{}'\"-")
                            if clean_word.isupper() and 1 <= len(clean_word) <= 5:
                                symbols.append(clean_word)

                        # If no uppercase symbols found, look for words after stock/price/quote
                        if not symbols:
                            for i, word in enumerate(words):
                                if word.lower() in ["stock", "quote", "price", "for"]:
                                    if i + 1 < len(words):
                                        symbol = words[i + 1].strip(",.?!()[]{}'\"-").upper()
                                        symbols.append(symbol)

                        # Try each potential symbol
                        for symbol in symbols:
                            response = self.execute(action="quote", symbol=symbol)
                            if response.success:
                                result = response.result
                                return f"Stock quote for {symbol}: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                        # If we got here, try the most common interpretation
                        if "IBM" in query.upper():
                            response = self.execute(action="quote", symbol="IBM")
                            if response.success:
                                result = response.result
                                return f"Stock quote for IBM: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                        # Try Microsoft if mentioned
                        if "MICROSOFT" in query.upper() or "MSFT" in query.upper():
                            response = self.execute(action="quote", symbol="MSFT")
                            if response.success:
                                result = response.result
                                return f"Stock quote for MSFT: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                        return "Error: Could not extract a valid stock symbol from query. Please specify a stock symbol like 'IBM' or 'AAPL'."

                    elif "exchange" in query.lower() or "currency" in query.lower() or "rate" in query.lower():
                        # Extract currencies
                        words = query.split()
                        from_currency = None
                        to_currency = None

                        # Look for currency codes (3 uppercase letters)
                        currency_codes = []
                        for word in words:
                            # Clean the word of punctuation
                            clean_word = word.strip(",.?!()[]{}'\"-/")
                            if clean_word.isupper() and len(clean_word) == 3:
                                currency_codes.append(clean_word)

                        # If we found exactly two currency codes, use them
                        if len(currency_codes) >= 2:
                            from_currency = currency_codes[0]
                            to_currency = currency_codes[1]

                        # If we didn't find currency codes, look for words after from/to
                        if not from_currency or not to_currency:
                            for i, word in enumerate(words):
                                if word.lower() in ["from", "convert"]:
                                    if i + 1 < len(words):
                                        from_currency = words[i + 1].strip(",.?!()[]{}'\"-").upper()
                                if word.lower() in ["to", "into"]:
                                    if i + 1 < len(words):
                                        to_currency = words[i + 1].strip(",.?!()[]{}'\"-").upper()

                        # If we still don't have currencies, try the most common interpretation
                        if "USD" in query.upper() and "JPY" in query.upper():
                            from_currency = "USD"
                            to_currency = "JPY"

                        if from_currency and to_currency:
                            response = self.execute(action="exchange_rate", from_currency=from_currency, to_currency=to_currency)
                            if response.success:
                                result = response.result
                                return f"Exchange rate from {result['from_currency']} to {result['to_currency']}: {result['exchange_rate']}"
                            else:
                                return f"Error: {response.error}"

                        return "Error: Could not extract valid currency codes from query. Please specify currency codes like 'USD' and 'JPY'."

                    elif "search" in query.lower() or "find" in query.lower():
                        # Extract keywords
                        keywords = query.replace("search", "").replace("find", "").strip()
                        response = self.execute(action="search", keywords=keywords)
                        if response.success:
                            results = response.result[:3]  # Limit to top 3 results
                            output = "Search results:\n"
                            for i, result in enumerate(results, 1):
                                output += f"{i}. {result['2. name']} ({result['1. symbol']}): {result['4. region']}\n"
                            return output
                        else:
                            return f"Error: {response.error}"

                    elif "market" in query.lower() or "gainers" in query.lower() or "losers" in query.lower():
                        response = self.execute(action="market_status")
                        if response.success:
                            result = response.result
                            output = "Market Status:\n"

                            if "top_gainers" in result and result["top_gainers"]:
                                output += "Top Gainers:\n"
                                for i, gainer in enumerate(result["top_gainers"][:3], 1):
                                    output += f"{i}. {gainer['ticker']}: ${gainer['price']} ({gainer['change_percentage']})\n"

                            if "top_losers" in result and result["top_losers"]:
                                output += "\nTop Losers:\n"
                                for i, loser in enumerate(result["top_losers"][:3], 1):
                                    output += f"{i}. {loser['ticker']}: ${loser['price']} ({loser['change_percentage']})\n"

                            return output
                        else:
                            return f"Error: {response.error}"

                    else:
                        # Check if the query might be a stock symbol
                        words = query.split()
                        for word in words:
                            clean_word = word.strip(",.?!()[]{}'\"-")
                            if clean_word.isupper() and 1 <= len(clean_word) <= 5:
                                response = self.execute(action="quote", symbol=clean_word)
                                if response.success:
                                    result = response.result
                                    return f"Stock quote for {clean_word}: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                        # Check for common company names
                        if "MICROSOFT" in query.upper() or "MSFT" in query.upper():
                            response = self.execute(action="quote", symbol="MSFT")
                            if response.success:
                                result = response.result
                                return f"Stock quote for MSFT: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                        if "APPLE" in query.upper() or "AAPL" in query.upper():
                            response = self.execute(action="quote", symbol="AAPL")
                            if response.success:
                                result = response.result
                                return f"Stock quote for AAPL: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                        if "GOOGLE" in query.upper() or "ALPHABET" in query.upper() or "GOOGL" in query.upper():
                            response = self.execute(action="quote", symbol="GOOGL")
                            if response.success:
                                result = response.result
                                return f"Stock quote for GOOGL: Price: ${result['price']}, Change: {result['change']} ({result['change_percent']})"

                        return "I'm not sure what financial information you're looking for. Try asking about stock quotes, exchange rates, or market status."

                except Exception as e:
                    return f"Error processing finance query: {str(e)}"

            # Create a LangChain tool
            return LangChainTool(
                name=name or "finance",
                description=description or "Get financial data including stock quotes, currency exchange rates, and market status. Ask about stock prices, exchange rates, or market gainers/losers.",
                func=run_finance_tool
            )
        except ImportError as e:
            raise ImportError(f"Error loading LangChain tools: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create the tool
    finance_tool = FinanceTool()

    # Test stock quote
    try:
        response = finance_tool.execute(action="quote", symbol="IBM")
        print("Stock Quote for IBM:")
        print(json.dumps(response.result, indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")

    # Test exchange rate
    try:
        response = finance_tool.execute(action="exchange_rate", from_currency="USD", to_currency="JPY")
        print("Exchange Rate from USD to JPY:")
        print(json.dumps(response.result, indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")

    # Test symbol search
    try:
        response = finance_tool.execute(action="search", keywords="Microsoft")
        print("Search Results for 'Microsoft':")
        print(json.dumps(response.result, indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
