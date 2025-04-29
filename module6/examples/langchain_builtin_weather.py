"""
LangChain Built-in Weather Tool Example
-------------------------------------
This script demonstrates how to use LangChain's built-in OpenWeatherMap tool.
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
    print_section("LangChain Built-in Weather Tool Example")
    
    # Check if required packages are installed
    try:
        from langchain.agents import AgentType, initialize_agent, load_tools
        from langchain_openai import OpenAI
    except ImportError:
        print("❌ LangChain or OpenAI packages are not installed.")
        print("Please install them with:")
        print("pip install langchain langchain_openai")
        return
    
    # Check if API keys are set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        return
    
    if not openweathermap_api_key:
        print("❌ OPENWEATHERMAP_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        return
    
    print("✅ API keys found")
    
    print_section("Creating LangChain Agent")
    
    # Create the LLM
    llm = OpenAI(temperature=0)
    print("✅ Created OpenAI LLM")
    
    # Load the built-in OpenWeatherMap tool
    tools = load_tools(["openweathermap-api"], llm=llm)
    print(f"✅ Loaded {len(tools)} built-in tools")
    
    # Create the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    print("✅ Created LangChain Agent")
    
    print_section("Running Agent")
    
    # Run the agent with a weather query
    try:
        result = agent.run("What's the weather like in London?")
        print("\nResult:", result)
    except Exception as e:
        print(f"❌ Error running agent: {e}")
    
    print_section("Example Complete")

if __name__ == "__main__":
    main()
