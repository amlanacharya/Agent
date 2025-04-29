"""
Multi-Tool Agent Example
----------------------
This script demonstrates how to use multiple tools together in a LangChain agent.
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
    print_section("Multi-Tool Agent Example")
    
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
    serper_api_key = os.getenv("SERPER_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        return
    
    if not openweathermap_api_key:
        print("❌ OPENWEATHERMAP_API_KEY environment variable is not set.")
        print("Please set it in your .env file or environment.")
        return
    
    print("✅ API keys found")
    
    # Import our tools
    from module6.code.tools.weather_tool import WeatherTool
    
    try:
        from module6.code.tools.search_tool import SearchTool
        search_tool_available = True
    except ImportError:
        search_tool_available = False
        print("⚠️ SearchTool not available, will use built-in tools instead")
    
    print_section("Creating LangChain Agent")
    
    # Create the LLM
    llm = OpenAI(temperature=0)
    print("✅ Created OpenAI LLM")
    
    # Create our weather tool and convert it to a LangChain tool
    weather_tool = WeatherTool()
    langchain_weather_tool = weather_tool.to_langchain_tool()
    print("✅ Created Weather Tool and converted to LangChain format")
    
    # Create a list of tools
    tools = [langchain_weather_tool]
    
    # Add search tool if available
    if search_tool_available and serper_api_key:
        search_tool = SearchTool()
        langchain_search_tool = search_tool.to_langchain_tool()
        tools.append(langchain_search_tool)
        print("✅ Added Search Tool")
    
    # Add built-in tools
    builtin_tools = load_tools(["llm-math"], llm=llm)
    tools.extend(builtin_tools)
    print(f"✅ Added {len(builtin_tools)} built-in tools")
    
    # Create the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    print("✅ Created LangChain Agent with multiple tools")
    
    print_section("Running Agent")
    
    # Run the agent with a query that might use multiple tools
    try:
        result = agent.run(
            "What's the weather like in London? Also, what is the square root of 256?"
        )
        print("\nResult:", result)
    except Exception as e:
        print(f"❌ Error running agent: {e}")
    
    print_section("Example Complete")
    print("You can now use multiple tools together in your own applications!")

if __name__ == "__main__":
    main()
