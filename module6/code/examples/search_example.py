"""
Example script demonstrating how to use the Search tool.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the Search tool
from module6.code.tools.search_tool import SearchTool

# Try to import LangChain components for the LangChain integration example
try:
    from langchain_openai import OpenAI
    from langchain.agents import initialize_agent, AgentType
    LANGCHAIN_AGENT_AVAILABLE = True
except ImportError:
    LANGCHAIN_AGENT_AVAILABLE = False
    print("LangChain agent components not available. Install with 'pip install langchain-openai langchain'")

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

def main():
    """Main function to demonstrate the Search tool."""
    print_header("Search Tool Demo")

    # Create the tool with the new API key
    search_tool = SearchTool(
        serper_api_key="26be841e26a32aea8c1f43bbc7e497d9fe6393ed",
        max_results=5,
        use_fallback=True
    )

    print(f"Initialized Search tool with max_results: {search_tool.max_results}")
    print(f"Has Serper API key: {search_tool.metadata['has_serper_key']}")
    print(f"Using fallback: {search_tool.use_fallback}")

    # Example 1: Basic search
    print_section("Example 1: Basic Search")
    query = "What is Python programming language?"
    print(f"Query: {query}")

    results = search_tool.search(query)
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result.title}")
        print(f"   URL: {result.link}")
        if result.snippet:
            print(f"   Snippet: {result.snippet}")
        print(f"   Source: {result.source}")

    # Example 2: Search with different parameters
    print_section("Example 2: Search with Different Parameters")
    query = "Latest news about artificial intelligence"
    print(f"Query: {query}")
    print("Parameters: search_type=news, num_results=3")

    try:
        results = search_tool.search(
            query=query,
            search_type="news",
            num_results=3
        )
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.title}")
            print(f"   URL: {result.link}")
            if result.snippet:
                print(f"   Snippet: {result.snippet}")
            print(f"   Source: {result.source}")
    except Exception as e:
        print(f"Search failed: {e}")

    # Example 3: Force using fallback
    print_section("Example 3: Force Using Fallback (DuckDuckGo)")
    query = "Best programming languages to learn in 2023"
    print(f"Query: {query}")
    print("Parameters: force_fallback=True")

    try:
        results = search_tool.search(
            query=query,
            force_fallback=True
        )
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.title}")
            print(f"   URL: {result.link}")
            if result.snippet:
                print(f"   Snippet: {result.snippet}")
            print(f"   Source: {result.source}")
    except Exception as e:
        print(f"Search failed: {e}")

    # Example 4: Using the execute method directly
    print_section("Example 4: Using the execute Method Directly")
    query = "How to learn machine learning"
    print(f"Query: {query}")

    response = search_tool.execute(query=query, num_results=2)
    print("\nSuccess:", response.success)
    if response.success:
        print(f"Found {len(response.result)} results:")
        for i, result in enumerate(response.result):
            print(f"\n{i+1}. {result.title}")
            print(f"   URL: {result.link}")
    else:
        print("Error:", response.error)
    print("Metadata:", response.metadata)

    # Example 5: LangChain Integration
    if LANGCHAIN_AGENT_AVAILABLE:
        print_section("Example 5: LangChain Integration")
        print("Creating a LangChain Tool from our SearchTool and using it with an agent")

        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("OpenAI API key not found. Skipping LangChain agent example.")
        else:
            try:
                # Create a LangChain Tool from our SearchTool
                search_langchain_tool = search_tool.to_langchain_tool(
                    name="web_search",
                    description="Search the web for information on a given query"
                )

                # Create a simple LLM
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

                # Create a list of tools
                tools = [search_langchain_tool]

                # Initialize the agent
                agent = initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True
                )

                # Run the agent
                print("\nRunning LangChain agent with our search tool...")
                result = agent.run("What is the capital of France and what is its population?")
                print("\nAgent result:", result)
            except Exception as e:
                print(f"Error running LangChain agent: {e}")
    else:
        print_section("Example 5: LangChain Integration (Skipped)")
        print("LangChain components not available. Install with 'pip install langchain-openai langchain'")

    print_header("End of Demo")

if __name__ == "__main__":
    main()
