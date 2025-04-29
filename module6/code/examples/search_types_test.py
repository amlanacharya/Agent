"""
Example script demonstrating how to use the Search tool with different search types.
This script tests the LangChain integration and different search types.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path so we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)

# Import the Search tool
from module6.code.tools.search_tool import SearchTool

# Load environment variables
load_dotenv()

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"{text.center(80)}")
    print("="*80)

def print_section(text):
    """Print a formatted section header."""
    print("\n" + "-"*80)
    print(f"{text}")
    print("-"*80)

def print_results(results):
    """Print search results in a formatted way."""
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.title}")
        print(f"   URL: {result.link}")
        if result.snippet:
            print(f"   Snippet: {result.snippet}")
        print(f"   Source: {result.source}")

def main():
    """Main function to test the Search tool with different search types."""
    print_header("Search Tool Test - Different Search Types")

    # Create the tool with the API key
    search_tool = SearchTool(
        serper_api_key="26be841e26a32aea8c1f43bbc7e497d9fe6393ed",
        max_results=3,
        use_fallback=True
    )

    print(f"Initialized Search tool with max_results: {search_tool.max_results}")
    print(f"Has Serper API key: {search_tool.metadata['has_serper_key']}")
    print(f"Using LangChain: {search_tool.metadata['using_langchain']}")
    print(f"Using LangChain Serper: {search_tool.metadata['using_langchain_serper']}")
    print(f"Using fallback: {search_tool.use_fallback}")

    # Test 1: Regular web search (search type: "search")
    print_section("Test 1: Regular Web Search (search_type='search')")
    query = "What is Python programming language?"
    print(f"Query: {query}")
    print(f"Search type: search")

    try:
        results = search_tool.search(
            query=query,
            search_type="search"
        )
        print_results(results)
    except Exception as e:
        print(f"Search failed: {e}")

    # Test 2: News search (search type: "news")
    print_section("Test 2: News Search (search_type='news')")
    query = "Latest news about artificial intelligence"
    print(f"Query: {query}")
    print(f"Search type: news")

    try:
        results = search_tool.search(
            query=query,
            search_type="news"
        )
        print_results(results)
    except Exception as e:
        print(f"Search failed: {e}")

    # Test 3: Places search (search type: "places")
    print_section("Test 3: Places Search (search_type='places')")
    query = "Coffee shops in New York"
    print(f"Query: {query}")
    print(f"Search type: places")

    try:
        results = search_tool.search(
            query=query,
            search_type="places"
        )
        print_results(results)
    except Exception as e:
        print(f"Search failed: {e}")

    # Test 4: Images search (search type: "images")
    print_section("Test 4: Images Search (search_type='images')")
    query = "Cute puppies"
    print(f"Query: {query}")
    print(f"Search type: images")

    try:
        results = search_tool.search(
            query=query,
            search_type="images"
        )
        print_results(results)
    except Exception as e:
        print(f"Search failed: {e}")

    # Test 5: LangChain Tool Integration
    if search_tool.metadata['using_langchain']:
        print_section("Test 5: LangChain Tool Integration")
        print("Creating a LangChain Tool from our SearchTool")

        try:
            # Create a LangChain Tool from our SearchTool
            search_langchain_tool = search_tool.to_langchain_tool(
                name="web_search",
                description="Search the web for information on a given query"
            )

            # Test the tool with a simple query
            print("\nTesting LangChain Tool with query: 'What is the capital of France?'")
            result = search_langchain_tool.func("What is the capital of France?")
            print("\nLangChain Tool result:")
            print(result)
        except Exception as e:
            print(f"Error testing LangChain tool: {e}")
    else:
        print_section("Test 5: LangChain Tool Integration (Skipped)")
        print("LangChain is not available. Install with 'pip install langchain-community langchain'")

    print_header("End of Tests")

if __name__ == "__main__":
    main()
