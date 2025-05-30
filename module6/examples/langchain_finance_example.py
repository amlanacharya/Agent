"""
LangChain Finance Tool Example
---------------------------
This script demonstrates how to use the Finance Tool with LangChain.
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
    print_section("LangChain Finance Tool Example")
    
    # Check if required packages are installed
    try:
        from langchain.agents import AgentType, initialize_agent
        from langchain_openai import OpenAI
    except ImportError:
        print("❌ LangChain or OpenAI packages are not installed.")
        print("Please install them with:")
        print("pip install langchain langchain_openai")
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
    
    print("✅ API keys found")
    
    # Import our finance tool
    from module6.code.tools.finance_tool import FinanceTool
    
    print_section("Creating LangChain Agent")
    
    # Create the LLM
    llm = OpenAI(temperature=0)
    print("✅ Created OpenAI LLM")
    
    # Create our finance tool and convert it to a LangChain tool
    try:
        finance_tool = FinanceTool()
        langchain_finance_tool = finance_tool.to_langchain_tool()
        print("✅ Created Finance Tool and converted to LangChain format")
        
        # Create the agent
        agent = initialize_agent(
            tools=[langchain_finance_tool],
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
            "What are the top market gainers today?"
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
