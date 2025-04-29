"""
Direct Finance API Example
-----------------------
This script demonstrates how to use the Alpha Vantage API directly.
"""

import os
import sys
import json
import requests
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

def get_api_key():
    """Get and clean the API key."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable is not set")
    
    # Clean the API key (remove any comments and quotes)
    if "#" in api_key:
        api_key = api_key.split("#")[0].strip()
    
    # Remove quotes if present
    api_key = api_key.strip('"\'').strip()
    
    return api_key

def make_request(function, **params):
    """Make a request to the Alpha Vantage API."""
    base_url = "https://www.alphavantage.co/query"
    
    # Add function and API key to params
    params["function"] = function
    params["apikey"] = get_api_key()
    
    # Make the request
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    return response.json()

def get_stock_quote(symbol):
    """Get a stock quote for the given symbol."""
    return make_request("GLOBAL_QUOTE", symbol=symbol)

def get_exchange_rate(from_currency, to_currency):
    """Get the exchange rate between two currencies."""
    return make_request("CURRENCY_EXCHANGE_RATE", from_currency=from_currency, to_currency=to_currency)

def search_symbols(keywords):
    """Search for symbols matching the given keywords."""
    return make_request("SYMBOL_SEARCH", keywords=keywords)

def get_time_series_daily(symbol):
    """Get daily time series data for the given symbol."""
    return make_request("TIME_SERIES_DAILY", symbol=symbol)

def get_top_gainers_losers():
    """Get the top gainers, losers, and most active stocks."""
    return make_request("TOP_GAINERS_LOSERS")

def main():
    """Run the example."""
    print_section("Direct Finance API Example")
    
    try:
        api_key = get_api_key()
        print(f"✅ API key found: {api_key[:4]}...{api_key[-4:]}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Example 1: Get a stock quote
    print_section("Example 1: Get Stock Quote")
    try:
        result = get_stock_quote("IBM")
        print("Stock Quote for IBM:")
        print_json(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Get an exchange rate
    print_section("Example 2: Get Exchange Rate")
    try:
        result = get_exchange_rate("USD", "JPY")
        print("Exchange Rate from USD to JPY:")
        print_json(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Search for symbols
    print_section("Example 3: Search for Symbols")
    try:
        result = search_symbols("Microsoft")
        print("Search Results for 'Microsoft':")
        # Limit to top 3 results for readability
        if "bestMatches" in result:
            results = result["bestMatches"][:3] if len(result["bestMatches"]) > 3 else result["bestMatches"]
            print_json(results)
        else:
            print_json(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Get time series data
    print_section("Example 4: Get Time Series Data")
    try:
        result = get_time_series_daily("IBM")
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
        result = get_top_gainers_losers()
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
    print("You can now use the Alpha Vantage API in your own applications!")

if __name__ == "__main__":
    main()
