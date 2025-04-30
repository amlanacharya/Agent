"""
Example script demonstrating how to use the Tool Registry.
This script shows how to register tools, categorize them, and retrieve them by category.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path so we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)

# Import the tools and registry
from module6.code.tools import OpenAITool, GroqTool, SearchTool, WeatherTool, AlphaVantageTool
from module6.code.registry import ToolRegistry, ToolCategory

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
    """Main function to demonstrate the Tool Registry."""
    print_header("Tool Registry Example")
    
    # Step 1: Create the Tool Registry
    print_section("Step 1: Creating the Tool Registry")
    registry = ToolRegistry()
    print(f"Created empty Tool Registry: {registry}")
    
    # Step 2: Create some tools
    print_section("Step 2: Creating Tools")
    
    # Check if API keys are set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    serper_api_key = os.getenv("SERPER_API_KEY")
    openweathermap_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    
    tools_to_create = []
    
    # Create OpenAI Tool if API key is available
    if openai_api_key:
        print("✅ OPENAI_API_KEY found, creating OpenAI Tool")
        openai_tool = OpenAITool(
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.7
        )
        tools_to_create.append((openai_tool, [ToolCategory.CONTENT_GENERATION]))
    else:
        print("❌ OPENAI_API_KEY not found, skipping OpenAI Tool")
    
    # Create Groq Tool if API key is available
    if groq_api_key:
        print("✅ GROQ_API_KEY found, creating Groq Tool")
        groq_tool = GroqTool(
            model="llama3-8b-8192",
            max_tokens=150,
            temperature=0.7
        )
        tools_to_create.append((groq_tool, [ToolCategory.CONTENT_GENERATION]))
    else:
        print("❌ GROQ_API_KEY not found, skipping Groq Tool")
    
    # Create Search Tool if API key is available
    if serper_api_key:
        print("✅ SERPER_API_KEY found, creating Search Tool")
        search_tool = SearchTool(
            max_results=3,
            use_fallback=True
        )
        tools_to_create.append((search_tool, [ToolCategory.INFORMATION_RETRIEVAL, ToolCategory.EXTERNAL_API]))
    else:
        print("❌ SERPER_API_KEY not found, skipping Search Tool")
    
    # Create Weather Tool if API key is available
    if openweathermap_api_key:
        print("✅ OPENWEATHERMAP_API_KEY found, creating Weather Tool")
        weather_tool = WeatherTool()
        tools_to_create.append((weather_tool, [ToolCategory.EXTERNAL_API]))
    else:
        print("❌ OPENWEATHERMAP_API_KEY not found, skipping Weather Tool")
    
    # Create AlphaVantage Tool if API key is available
    if alphavantage_api_key:
        print("✅ ALPHAVANTAGE_API_KEY found, creating AlphaVantage Tool")
        alphavantage_tool = AlphaVantageTool()
        tools_to_create.append((alphavantage_tool, [ToolCategory.EXTERNAL_API, "finance"]))
    else:
        print("❌ ALPHAVANTAGE_API_KEY not found, skipping AlphaVantage Tool")
    
    # Step 3: Register the tools
    print_section("Step 3: Registering Tools")
    
    for tool, categories in tools_to_create:
        try:
            registry.register_tool(tool, categories)
            print(f"✅ Registered {tool.name} tool with categories: {[str(c) if isinstance(c, str) else c.name for c in categories]}")
        except Exception as e:
            print(f"❌ Error registering {tool.name} tool: {e}")
    
    # Step 4: Explore the registry
    print_section("Step 4: Exploring the Registry")
    
    print(f"Registered tools: {registry.list_tools()}")
    print(f"Available categories: {registry.get_categories()}")
    
    # Step 5: Get tools by category
    print_section("Step 5: Getting Tools by Category")
    
    # Get content generation tools
    content_tools = registry.get_tools_by_category(ToolCategory.CONTENT_GENERATION)
    print(f"Content Generation Tools: {[tool.name for tool in content_tools]}")
    
    # Get external API tools
    api_tools = registry.get_tools_by_category(ToolCategory.EXTERNAL_API)
    print(f"External API Tools: {[tool.name for tool in api_tools]}")
    
    # Get information retrieval tools
    info_tools = registry.get_tools_by_category(ToolCategory.INFORMATION_RETRIEVAL)
    print(f"Information Retrieval Tools: {[tool.name for tool in info_tools]}")
    
    # Get finance tools (custom category)
    finance_tools = registry.get_tools_by_category("finance")
    print(f"Finance Tools: {[tool.name for tool in finance_tools]}")
    
    # Step 6: Get tool schemas
    print_section("Step 6: Getting Tool Schemas")
    
    schemas = registry.get_all_schemas()
    print(f"Found {len(schemas)} tool schemas")
    
    # Print the first schema as an example
    if schemas:
        print("\nExample Schema:")
        for key, value in schemas[0].items():
            if key != "properties":
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {{...}}")  # Don't print the full properties
    
    # Step 7: Convert to LangChain tools (if available)
    print_section("Step 7: Converting to LangChain Tools")
    
    try:
        langchain_tools = registry.to_langchain_tools()
        print(f"✅ Converted {len(langchain_tools)} tools to LangChain format")
    except ImportError:
        print("❌ LangChain not available, skipping conversion")
    except Exception as e:
        print(f"❌ Error converting to LangChain tools: {e}")
    
    # Step 8: Unregister a tool
    print_section("Step 8: Unregistering a Tool")
    
    if registry.list_tools():
        tool_to_remove = registry.list_tools()[0]
        success = registry.unregister_tool(tool_to_remove)
        if success:
            print(f"✅ Unregistered tool: {tool_to_remove}")
        else:
            print(f"❌ Failed to unregister tool: {tool_to_remove}")
        
        print(f"Remaining tools: {registry.list_tools()}")
    else:
        print("No tools to unregister")
    
    # Step 9: Export registry to JSON
    print_section("Step 9: Exporting Registry to JSON")
    
    registry_json = registry.to_json()
    print("Registry JSON (truncated):")
    print(registry_json[:500] + "..." if len(registry_json) > 500 else registry_json)
    
    print_section("Example Complete")
    print("You can now use the Tool Registry in your own applications!")

if __name__ == "__main__":
    main()
