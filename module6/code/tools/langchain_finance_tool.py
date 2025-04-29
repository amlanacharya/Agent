"""
LangChain Finance Tool Integration
--------------------------------
This file provides integration with LangChain's Alpha Vantage API wrapper.
"""

from typing import Dict, Any, Optional, List, Type
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .base_tool import BaseTool, ToolResponse

# Load environment variables
load_dotenv()

# Check if LangChain is available
try:
    from langchain.tools import BaseTool as LangChainBaseTool
    from langchain.tools import Tool as LangChainTool
    from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class FinanceInput(BaseModel):
    """Input for the finance tool."""
    query: str = Field(..., description="The finance query, e.g., 'What is the stock price of IBM?'")


class LangChainFinanceTool(BaseTool):
    """Tool for fetching financial data using LangChain's Alpha Vantage API wrapper."""

    def __init__(
        self,
        name: str = "finance",
        description: str = "Get financial data including stock quotes and currency exchange rates",
        api_key: Optional[str] = None
    ):
        """
        Initialize the Finance tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            api_key: Alpha Vantage API key (if None, will try to get from environment)
        """
        super().__init__(name, description)

        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install it with 'pip install langchain langchain_community'"
            )

        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set it as an argument or as ALPHAVANTAGE_API_KEY environment variable.")

        # Clean the API key (remove any comments and quotes)
        if "#" in self.api_key:
            self.api_key = self.api_key.split("#")[0].strip()

        # Remove quotes if present
        self.api_key = self.api_key.strip('"\'').strip()

        # Initialize the Alpha Vantage API wrapper
        self.alpha_vantage = AlphaVantageAPIWrapper(api_key=self.api_key)

        # Add API info to metadata
        self.metadata["api_provider"] = "Alpha Vantage"
        self.metadata["using_langchain"] = True

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

                try:
                    result = self.alpha_vantage._get_quote_endpoint(symbol)
                    if "Global Quote" in result:
                        quote_data = result["Global Quote"]
                        return ToolResponse(
                            success=True,
                            result=quote_data,
                            metadata={
                                "action": action,
                                "symbol": symbol
                            }
                        )
                    else:
                        return ToolResponse(
                            success=False,
                            error=f"No quote data found for symbol: {symbol}"
                        )
                except Exception as e:
                    return ToolResponse(
                        success=False,
                        error=f"Error getting quote: {str(e)}"
                    )

            elif action == "exchange_rate":
                from_currency = kwargs.get("from_currency")
                to_currency = kwargs.get("to_currency")
                if not from_currency or not to_currency:
                    return ToolResponse(
                        success=False,
                        error="Both from_currency and to_currency must be provided for exchange_rate action"
                    )

                try:
                    result = self.alpha_vantage._get_exchange_rate(from_currency, to_currency)
                    if "Realtime Currency Exchange Rate" in result:
                        rate_data = result["Realtime Currency Exchange Rate"]
                        return ToolResponse(
                            success=True,
                            result=rate_data,
                            metadata={
                                "action": action,
                                "from_currency": from_currency,
                                "to_currency": to_currency
                            }
                        )
                    else:
                        return ToolResponse(
                            success=False,
                            error=f"No exchange rate data found for {from_currency} to {to_currency}"
                        )
                except Exception as e:
                    return ToolResponse(
                        success=False,
                        error=f"Error getting exchange rate: {str(e)}"
                    )

            elif action == "search":
                keywords = kwargs.get("keywords")
                if not keywords:
                    return ToolResponse(
                        success=False,
                        error="Keywords must be provided for search action"
                    )

                try:
                    result = self.alpha_vantage.search_symbols(keywords)
                    return ToolResponse(
                        success=True,
                        result=result,
                        metadata={
                            "action": action,
                            "keywords": keywords
                        }
                    )
                except Exception as e:
                    return ToolResponse(
                        success=False,
                        error=f"Error searching symbols: {str(e)}"
                    )

            elif action == "time_series_daily":
                symbol = kwargs.get("symbol")
                if not symbol:
                    return ToolResponse(
                        success=False,
                        error="Symbol must be provided for time_series_daily action"
                    )

                try:
                    result = self.alpha_vantage._get_time_series_daily(symbol)
                    return ToolResponse(
                        success=True,
                        result=result,
                        metadata={
                            "action": action,
                            "symbol": symbol
                        }
                    )
                except Exception as e:
                    return ToolResponse(
                        success=False,
                        error=f"Error getting time series data: {str(e)}"
                    )

            elif action == "market_status":
                try:
                    result = self.alpha_vantage._get_top_gainers_losers()
                    return ToolResponse(
                        success=True,
                        result=result,
                        metadata={
                            "action": action
                        }
                    )
                except Exception as e:
                    return ToolResponse(
                        success=False,
                        error=f"Error getting market status: {str(e)}"
                    )

            else:
                return ToolResponse(
                    success=False,
                    error=f"Unknown action: {action}. Supported actions are: quote, exchange_rate, search, time_series_daily, market_status"
                )

        except Exception as e:
            # Handle other errors
            return ToolResponse(
                success=False,
                error=f"Finance tool execution failed: {str(e)}"
            )

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
                            "enum": ["quote", "exchange_rate", "search", "time_series_daily", "market_status"],
                            "description": "The action to perform"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol for quote or time_series_daily action (e.g., 'IBM', 'AAPL')"
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
                                "properties": {"action": {"enum": ["quote", "time_series_daily"]}},
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

    def to_langchain_tool(self) -> Any:
        """
        Convert this tool to a LangChain tool.

        Returns:
            A LangChain tool
        """
        def run_finance_tool(query: str) -> str:
            """Run the finance tool with the given query."""
            try:
                # Parse the query to extract parameters
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
                        try:
                            quote_data = self.alpha_vantage._get_quote_endpoint(symbol)
                            if "Global Quote" in quote_data:
                                data = quote_data["Global Quote"]
                                return f"Stock quote for {symbol}: Price: ${data['05. price']}, Change: {data['09. change']} ({data['10. change percent']})"
                        except:
                            continue

                    # If we got here, try the most common interpretation
                    if "IBM" in query.upper():
                        try:
                            quote_data = self.alpha_vantage._get_quote_endpoint("IBM")
                            if "Global Quote" in quote_data:
                                data = quote_data["Global Quote"]
                                return f"Stock quote for IBM: Price: ${data['05. price']}, Change: {data['09. change']} ({data['10. change percent']})"
                        except:
                            pass

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
                        try:
                            rate_data = self.alpha_vantage._get_exchange_rate(from_currency, to_currency)
                            if "Realtime Currency Exchange Rate" in rate_data:
                                data = rate_data["Realtime Currency Exchange Rate"]
                                return f"Exchange rate from {data['1. From_Currency Code']} to {data['3. To_Currency Code']}: {data['5. Exchange Rate']}"
                        except Exception as e:
                            return f"Error getting exchange rate: {str(e)}"

                    return "Error: Could not extract valid currency codes from query. Please specify currency codes like 'USD' and 'JPY'."

                elif "search" in query.lower() or "find" in query.lower():
                    # Extract keywords
                    keywords = query.replace("search", "").replace("find", "").strip()
                    try:
                        results = self.alpha_vantage.search_symbols(keywords)
                        if results:
                            # Limit to top 3 results
                            results = results[:3]
                            output = "Search results:\n"
                            for i, result in enumerate(results, 1):
                                output += f"{i}. {result['2. name']} ({result['1. symbol']}): {result['4. region']}\n"
                            return output
                        else:
                            return f"No results found for '{keywords}'"
                    except Exception as e:
                        return f"Error searching symbols: {str(e)}"

                elif "market" in query.lower() or "gainers" in query.lower() or "losers" in query.lower():
                    try:
                        result = self.alpha_vantage._get_top_gainers_losers()
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
                    except Exception as e:
                        return f"Error getting market status: {str(e)}"

                else:
                    return "I'm not sure what financial information you're looking for. Try asking about stock quotes, exchange rates, or market status."

            except Exception as e:
                return f"Error processing finance query: {str(e)}"

        # Create a LangChain tool
        return LangChainTool(
            name=self.name,
            description=self.description,
            func=run_finance_tool
        )


def get_langchain_finance_tool(api_key: Optional[str] = None) -> LangChainTool:
    """
    Get a LangChain-compatible finance tool.

    Args:
        api_key: Alpha Vantage API key (if None, will try to get from environment)

    Returns:
        LangChainTool: A LangChain-compatible finance tool
    """
    finance_tool = LangChainFinanceTool(api_key=api_key)
    return finance_tool.to_langchain_tool()
