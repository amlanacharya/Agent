"""
Alpha Vantage Tool Example
------------------------
This script demonstrates how to use the Alpha Vantage Tool.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from module6.code.tools.alpha_vantage_tool import AlphaVantageTool

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
    print_section("Alpha Vantage Tool Example")

    # Create the Alpha Vantage tool with explicit API key
    try:
        # Use the new API key directly
        api_key = "7W1J5NTH80LKBVZQ"
        print(f"Using API key: {api_key}")

        alpha_vantage_tool = AlphaVantageTool(api_key=api_key)
        print("✅ Alpha Vantage Tool created successfully")
    except ValueError as e:
        print(f"❌ Failed to create Alpha Vantage Tool: {e}")
        print("Make sure you have set the ALPHAVANTAGE_API_KEY environment variable")
        return

    # Example 1: Get a stock quote
    print_section("Example 1: Get Stock Quote")
    try:
        response = alpha_vantage_tool.execute(action="quote", symbol="IBM")
        if response.success:
            print("Stock Quote for IBM:")
            print_json(response.result)
        else:
            print(f"❌ Error: {response.error}")
    except Exception as e:
        print(f"❌ Exception: {e}")

    # Example 2: Get an exchange rate
    print_section("Example 2: Get Exchange Rate")
    try:
        response = alpha_vantage_tool.execute(action="exchange_rate", from_currency="USD", to_currency="JPY")
        if response.success:
            print("Exchange Rate from USD to JPY:")
            print_json(response.result)
        else:
            print(f"❌ Error: {response.error}")
    except Exception as e:
        print(f"❌ Exception: {e}")

    # Example 3: Search for symbols
    print_section("Example 3: Search for Symbols")
    try:
        response = alpha_vantage_tool.execute(action="search", keywords="Microsoft")
        if response.success:
            print("Search Results for 'Microsoft':")
            # Limit to top 3 results for readability
            results = response.result[:3] if len(response.result) > 3 else response.result
            print_json(results)
        else:
            print(f"❌ Error: {response.error}")
    except Exception as e:
        print(f"❌ Exception: {e}")

    print_section("Example Complete")
    print("You can now use the Alpha Vantage Tool in your own applications!")

if __name__ == "__main__":
    main()
