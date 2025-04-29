"""
LangChain Finance Wrapper Example
-------------------------------
This script demonstrates how to use the LangChain Alpha Vantage wrapper.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50 + "\n")

def print_json(data):
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2))

def main():
    """Run the example."""
    print_section("LangChain Finance Wrapper Example")

    # Check if required packages are installed
    try:
        from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
    except ImportError:
        print("❌ LangChain or Alpha Vantage packages are not installed.")
        print("Please install them with:")
        print("pip install langchain langchain_community")
        return

    # Check if API key is set
    alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not alphavantage_api_key:
        print("❌ ALPHAVANTAGE_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        return

    # Clean the API key (remove any comments and quotes)
    if "#" in alphavantage_api_key:
        alphavantage_api_key = alphavantage_api_key.split("#")[0].strip()

    # Remove quotes if present
    alphavantage_api_key = alphavantage_api_key.strip('"\'').strip()

    print("✅ API key found")

    # Create the Alpha Vantage wrapper
    alpha_vantage = AlphaVantageAPIWrapper(api_key=alphavantage_api_key)
    print("✅ Created Alpha Vantage wrapper")

    # Example 1: Get a stock quote
    print_section("Example 1: Get Stock Quote")
    try:
        result = alpha_vantage._get_quote_endpoint("IBM")
        print("Stock Quote for IBM:")
        print_json(result)
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 2: Get an exchange rate
    print_section("Example 2: Get Exchange Rate")
    try:
        result = alpha_vantage._get_exchange_rate("USD", "JPY")
        print("Exchange Rate from USD to JPY:")
        print_json(result)
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 3: Search for symbols
    print_section("Example 3: Search for Symbols")
    try:
        result = alpha_vantage.search_symbols("Microsoft")
        print("Search Results for 'Microsoft':")
        # Limit to top 3 results for readability
        results = result[:3] if len(result) > 3 else result
        print_json(results)
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 4: Get time series data
    print_section("Example 4: Get Time Series Data")
    try:
        result = alpha_vantage._get_time_series_daily("IBM")
        print("Time Series Data for IBM:")
        # Just show the first day for brevity
        if "Time Series (Daily)" in result:
            first_date = list(result["Time Series (Daily)"].keys())[0]
            first_day_data = result["Time Series (Daily)"][first_date]
            print(f"Data for {first_date}:")
            print_json(first_day_data)
        else:
            print_json(result)
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 5: Get market status
    print_section("Example 5: Get Market Status")
    try:
        result = alpha_vantage._get_top_gainers_losers()
        print("Market Status:")
        # Just show the number of gainers/losers for brevity
        if "top_gainers" in result:
            print(f"Top Gainers: {len(result['top_gainers'])} stocks")
        if "top_losers" in result:
            print(f"Top Losers: {len(result['top_losers'])} stocks")
        if "most_actively_traded" in result:
            print(f"Most Active: {len(result['most_actively_traded'])} stocks")
    except Exception as e:
        print(f"❌ Error: {e}")

    print_section("Example Complete")
    print("You can now use the Alpha Vantage wrapper in your own applications!")

if __name__ == "__main__":
    main()
