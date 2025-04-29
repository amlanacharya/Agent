"""
LangChain Finance Agent Example
----------------------------
This script demonstrates how to use the LangChain Finance Tool with an agent.
"""

import os
import sys
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

def main():
    """Run the example."""
    print_section("LangChain Finance Agent Example")

    # Check if required packages are installed
    try:
        from langchain.agents import AgentType, initialize_agent
        from langchain_openai import OpenAI
    except ImportError:
        print("❌ LangChain or OpenAI packages are not installed.")
        print("Please install them with:")
        print("pip install langchain langchain_openai langchain_community")
        return

    # Check if API keys are set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        return

    if not alphavantage_api_key:
        print("❌ ALPHAVANTAGE_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        return

    # Clean the API keys (remove any comments and quotes)
    if "#" in alphavantage_api_key:
        alphavantage_api_key = alphavantage_api_key.split("#")[0].strip()

    if "#" in openai_api_key:
        openai_api_key = openai_api_key.split("#")[0].strip()

    # Remove quotes if present
    alphavantage_api_key = alphavantage_api_key.strip('"\'').strip()
    openai_api_key = openai_api_key.strip('"\'').strip()

    print("✅ API keys found")

    # Import our finance tool
    from module6.code.tools.langchain_finance_tool import get_langchain_finance_tool

    print_section("Creating LangChain Agent")

    # Create the LLM
    llm = OpenAI(temperature=0)
    print("✅ Created OpenAI LLM")

    # Create our finance tool
    try:
        finance_tool = get_langchain_finance_tool(api_key=alphavantage_api_key)
        print("✅ Created Finance Tool")

        # Create the agent
        agent = initialize_agent(
            tools=[finance_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        print("✅ Created LangChain Agent")

        print_section("Running Agent")

        # Run the agent with financial queries
        queries = [
            "What's the current stock price of IBM?",
            "What's the exchange rate from USD to JPY?",
            "Can you find information about Microsoft stock?"
        ]

        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            try:
                result = agent.invoke({"input": query})
                print(f"Result: {result['output']}")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Error running agent: {e}")
                print("-" * 50)

    except ImportError as e:
        print(f"❌ Error: {e}")

    print_section("Example Complete")
    print("You can now use the Finance Tool with LangChain in your own applications!")

if __name__ == "__main__":
    main()
