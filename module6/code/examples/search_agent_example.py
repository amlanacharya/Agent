"""
Example script demonstrating how to use the Search tool with a LangChain agent.
This script shows how to create a simple agent that can search the web for information.
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

def main():
    """Main function to demonstrate the Search tool with a LangChain agent."""
    print_header("Search Agent Example")

    # Create the search tool
    search_tool = SearchTool(
        serper_api_key="26be841e26a32aea8c1f43bbc7e497d9fe6393ed",
        max_results=3,
        use_fallback=True
    )

    print(f"Initialized Search tool with max_results: {search_tool.max_results}")
    print(f"Has Serper API key: {search_tool.metadata['has_serper_key']}")
    print(f"Using LangChain: {search_tool.metadata['using_langchain']}")
    print(f"Using LangChain Serper: {search_tool.metadata['using_langchain_serper']}")

    # Check if LangChain is available
    if search_tool.metadata['using_langchain']:
        try:
            # Import LangChain components
            from langchain_openai import OpenAI
            from langchain.agents import initialize_agent, AgentType
            
            # Check if OpenAI API key is available
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("OpenAI API key not found. Skipping LangChain agent example.")
                return
                
            print_section("Creating a LangChain Agent with the Search Tool")
            
            # Convert the search tool to a LangChain tool
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
            
            # Run the agent with different queries
            queries = [
                "What is the capital of France and what is its population?",
                "What are the latest news about artificial intelligence?",
                "What are some good coffee shops in New York?",
                "Show me information about cute puppies"
            ]
            
            for i, query in enumerate(queries):
                print_section(f"Query {i+1}: {query}")
                try:
                    result = agent.run(query)
                    print("\nAgent result:")
                    print(result)
                except Exception as e:
                    print(f"Error running agent: {e}")
                    
        except ImportError as e:
            print(f"Error importing LangChain components: {e}")
            print("Make sure you have installed the required packages:")
            print("pip install langchain-openai langchain")
    else:
        print("LangChain is not available. Install with 'pip install langchain-community langchain'")

    print_header("End of Example")

if __name__ == "__main__":
    main()
