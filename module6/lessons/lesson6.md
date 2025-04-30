# 🚀 Module 6: Tool Integration & Function Calling - Lesson 6: Creating a Tool Registry 📚

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧰 Understand the concept of a tool registry in agentic systems
- 🔍 Learn how to design a centralized tool management system
- 📋 Implement a tool registry with categorization and discovery features
- 🔄 Convert tools to LangChain format for integration with agents
- 🛠️ Build a system that can dynamically discover and manage tools

---

## 📚 Introduction to Tool Registries

A tool registry is a central system for managing, discovering, and organizing tools in an agentic system. It provides a way to:

1. **Register and unregister tools** at runtime
2. **Categorize tools** by functionality or domain
3. **Discover tools** based on categories or capabilities
4. **Retrieve tool schemas** for function calling
5. **Convert tools** to different formats (e.g., LangChain tools)

In this lesson, we'll build a comprehensive tool registry that can manage all the tools we've created so far.

### Key Concepts

A well-designed tool registry should have:

1. **Tool Registration**: Methods to add and remove tools
2. **Categorization**: Support for organizing tools into categories
3. **Discovery**: Methods to find tools by name or category
4. **Schema Management**: Access to tool schemas for function calling
5. **Format Conversion**: Ability to convert tools to different formats

```python
# Example of a basic tool registry
class ToolRegistry:
    """A registry for managing and discovering tools."""
    
    def __init__(self):
        self.tools = {}  # Dictionary mapping tool names to tool instances
        self.categories = {}  # Dictionary mapping category names to sets of tool names
    
    def register_tool(self, tool, categories=None):
        """Register a tool with the registry."""
        self.tools[tool.name] = tool
        
        if categories:
            for category in categories:
                if category not in self.categories:
                    self.categories[category] = set()
                self.categories[category].add(tool.name)
    
    def get_tool(self, tool_name):
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category):
        """Get all tools in a category."""
        tool_names = self.categories.get(category, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
```

## 🧩 Designing a Tool Registry

Before implementing our tool registry, let's design its architecture and features.

### Tool Registry Architecture

Our `ToolRegistry` class will be the central component for managing tools. It will:

1. Store tools in a dictionary mapping tool names to tool instances
2. Maintain categories in a dictionary mapping category names to sets of tool names
3. Provide methods for registering, unregistering, and retrieving tools
4. Support categorization and discovery of tools
5. Enable conversion to LangChain tools for integration with agents

### Tool Categories

We'll support both predefined categories (using an Enum) and custom string categories:

```python
class ToolCategory(Enum):
    """Enum for tool categories."""
    CONTENT_GENERATION = auto()
    INFORMATION_RETRIEVAL = auto()
    DATA_ANALYSIS = auto()
    EXTERNAL_API = auto()
    UTILITY = auto()
    OTHER = auto()
```

This allows us to have a standardized set of categories while still supporting custom categories for specific domains.

## 🔄 Implementing the Tool Registry

Now, let's implement our tool registry with all the features we've designed.

### Tool Registry Implementation

```python
class ToolRegistry:
    """
    A registry for managing and discovering tools.
    
    The ToolRegistry provides a central place to register, discover, and manage tools.
    It supports categorization, filtering, and conversion to LangChain tools.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        # Dictionary mapping tool names to tool instances
        self.tools: Dict[str, BaseTool] = {}
        
        # Dictionary mapping category names to sets of tool names
        self.categories: Dict[Union[str, ToolCategory], Set[str]] = {}
        
        # Dictionary mapping tool names to their metadata
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default categories
        for category in ToolCategory:
            self.categories[category] = set()
            self.categories[category.name.lower()] = set()
    
    def register_tool(self, tool: BaseTool, categories: Optional[List[Union[str, ToolCategory]]] = None) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: The tool to register
            categories: Optional list of categories to assign to the tool
        
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self.tools:
            raise ValueError(f"A tool with the name '{tool.name}' is already registered")
        
        # Store the tool
        self.tools[tool.name] = tool
        
        # Store tool metadata
        self.tool_metadata[tool.name] = {
            "registered_at": time.time(),
            "description": tool.description,
            **tool.metadata
        }
        
        # Add to categories
        if categories:
            for category in categories:
                self.add_tool_to_category(tool.name, category)
        
        print(f"Registered tool: {tool.name}")
```

### Tool Categorization

Our registry supports adding tools to categories and retrieving tools by category:

```python
def add_tool_to_category(self, tool_name: str, category: Union[str, ToolCategory]) -> bool:
    """
    Add a tool to a category.
    
    Args:
        tool_name: The name of the tool to add
        category: The category to add the tool to
        
    Returns:
        bool: True if the tool was added to the category, False otherwise
        
    Raises:
        ValueError: If the tool is not registered
    """
    if tool_name not in self.tools:
        raise ValueError(f"Tool '{tool_name}' is not registered")
    
    # Convert ToolCategory enum to string if needed
    if isinstance(category, ToolCategory):
        category_name = category.name.lower()
        enum_category = category
    else:
        category_name = category.lower()
        try:
            enum_category = ToolCategory[category.upper()]
        except KeyError:
            enum_category = None
    
    # Create category if it doesn't exist
    if category_name not in self.categories:
        self.categories[category_name] = set()
    
    # Add tool to string category
    self.categories[category_name].add(tool_name)
    
    # Add tool to enum category if it exists
    if enum_category and enum_category not in self.categories:
        self.categories[enum_category] = set()
    
    if enum_category:
        self.categories[enum_category].add(tool_name)
    
    return True

def get_tools_by_category(self, category: Union[str, ToolCategory]) -> List[BaseTool]:
    """
    Get all tools in a category.
    
    Args:
        category: The category to get tools for
        
    Returns:
        List[BaseTool]: List of tools in the category
    """
    # Convert ToolCategory enum to string if needed
    if isinstance(category, ToolCategory):
        category_name = category.name.lower()
    else:
        category_name = category.lower()
    
    # Check if category exists
    if category_name not in self.categories:
        return []
    
    # Get tools in category
    tool_names = self.categories[category_name]
    return [self.tools[name] for name in tool_names if name in self.tools]
```

### LangChain Integration

Our registry can convert tools to LangChain format for integration with LangChain agents:

```python
def to_langchain_tools(self) -> List[Any]:
    """
    Convert all registered tools to LangChain tools.
    
    Returns:
        List[Any]: List of LangChain tools
        
    Raises:
        ImportError: If LangChain is not available
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not available. Install with 'pip install langchain'")
    
    langchain_tools = []
    
    for tool in self.tools.values():
        # Check if the tool has a to_langchain_tool method
        if hasattr(tool, 'to_langchain_tool') and callable(getattr(tool, 'to_langchain_tool')):
            try:
                langchain_tool = tool.to_langchain_tool()
                langchain_tools.append(langchain_tool)
            except Exception as e:
                print(f"Error converting tool '{tool.name}' to LangChain tool: {e}")
        else:
            print(f"Tool '{tool.name}' does not have a to_langchain_tool method")
    
    return langchain_tools
```

## 📊 Using the Tool Registry

Now that we've implemented our tool registry, let's see how to use it in practice.

### Basic Usage

```python
from module6.code.registry import ToolRegistry, ToolCategory
from module6.code.tools import OpenAITool, SearchTool, WeatherTool

# Create the registry
registry = ToolRegistry()

# Create some tools
openai_tool = OpenAITool(model="gpt-3.5-turbo")
search_tool = SearchTool()
weather_tool = WeatherTool()

# Register the tools with categories
registry.register_tool(openai_tool, [ToolCategory.CONTENT_GENERATION])
registry.register_tool(search_tool, [ToolCategory.INFORMATION_RETRIEVAL, ToolCategory.EXTERNAL_API])
registry.register_tool(weather_tool, [ToolCategory.EXTERNAL_API])

# Get all registered tools
all_tools = registry.list_tools()
print(f"Registered tools: {all_tools}")

# Get tools by category
content_tools = registry.get_tools_by_category(ToolCategory.CONTENT_GENERATION)
api_tools = registry.get_tools_by_category(ToolCategory.EXTERNAL_API)

print(f"Content Generation Tools: {[tool.name for tool in content_tools]}")
print(f"External API Tools: {[tool.name for tool in api_tools]}")

# Get tool schemas for function calling
schemas = registry.get_all_schemas()
```

### Creating a Multi-Tool Agent

The tool registry makes it easy to create a multi-tool agent that can use all registered tools:

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI

# Create an LLM
llm = OpenAI(temperature=0)

# Convert tools to LangChain format
langchain_tools = registry.to_langchain_tools()

# Create the agent
agent = initialize_agent(
    tools=langchain_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What's the weather like in London?")
print(result)
```

## 🧪 Testing the Tool Registry

To ensure our tool registry works correctly, we've created comprehensive unit tests:

```python
class TestToolRegistry(unittest.TestCase):
    """Test cases for the Tool Registry."""
    
    def setUp(self):
        """Set up the test environment."""
        self.registry = ToolRegistry()
        self.tool1 = MockTool("tool1", "Test tool 1")
        self.tool2 = MockTool("tool2", "Test tool 2")
    
    def test_register_tool(self):
        """Test registering a tool."""
        self.registry.register_tool(self.tool1)
        self.assertIn("tool1", self.registry.list_tools())
        self.assertEqual(self.registry.get_tool("tool1"), self.tool1)
    
    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        self.registry.add_tool_to_category("tool1", ToolCategory.UTILITY)
        self.registry.add_tool_to_category("tool2", ToolCategory.UTILITY)
        
        utility_tools = self.registry.get_tools_by_category(ToolCategory.UTILITY)
        self.assertEqual(len(utility_tools), 2)
        self.assertIn(self.tool1, utility_tools)
        self.assertIn(self.tool2, utility_tools)
```

## 🚀 Example: Tool Registry in Action

Let's see a complete example of using the tool registry:

```python
def main():
    """Main function to demonstrate the Tool Registry."""
    print("Tool Registry Example")
    
    # Create the Tool Registry
    registry = ToolRegistry()
    
    # Create some tools
    openai_tool = OpenAITool(model="gpt-3.5-turbo")
    search_tool = SearchTool()
    weather_tool = WeatherTool()
    
    # Register the tools with categories
    registry.register_tool(openai_tool, [ToolCategory.CONTENT_GENERATION])
    registry.register_tool(search_tool, [ToolCategory.INFORMATION_RETRIEVAL, ToolCategory.EXTERNAL_API])
    registry.register_tool(weather_tool, [ToolCategory.EXTERNAL_API])
    
    # Get tools by category
    content_tools = registry.get_tools_by_category(ToolCategory.CONTENT_GENERATION)
    api_tools = registry.get_tools_by_category(ToolCategory.EXTERNAL_API)
    
    print(f"Content Generation Tools: {[tool.name for tool in content_tools]}")
    print(f"External API Tools: {[tool.name for tool in api_tools]}")
    
    # Use a tool from the registry
    weather_tool = registry.get_tool("weather")
    if weather_tool:
        result = weather_tool.get_weather_by_location("London")
        print(f"Weather in London: {result.temperature}°C, {result.description}")
    
    # Convert to LangChain tools
    try:
        langchain_tools = registry.to_langchain_tools()
        print(f"Converted {len(langchain_tools)} tools to LangChain format")
    except ImportError:
        print("LangChain not available")
```

---

## 💪 Practice Exercises

1. **Exercise 1: Dynamic Tool Discovery**
   - Implement a method to discover tools from a directory
   - Add support for loading tools from configuration files
   - Create a plugin system for third-party tools

2. **Exercise 2: Tool Verification**
   - Add validation for tool schemas
   - Implement a method to check if a tool meets certain requirements
   - Create a system to verify tool outputs

3. **Exercise 3: Advanced Categorization**
   - Implement hierarchical categories
   - Add support for tagging tools with multiple attributes
   - Create a search function to find tools by capability

---

## 🔍 Key Concepts to Remember

1. **Tool Registry**: A central system for managing and discovering tools
2. **Categorization**: Organizing tools into categories for easier discovery
3. **Tool Discovery**: Finding tools based on categories or capabilities
4. **Schema Management**: Accessing tool schemas for function calling
5. **Format Conversion**: Converting tools to different formats for integration

---

## 🚀 Next Steps

In the next lesson, we'll:
- Implement function calling patterns with our tools
- Create a multi-tool agent that can use all our tools
- Build a system for selecting the right tool for a given task
- Implement error handling and fallback mechanisms
- Create a complete agentic system with tool integration

---

## 📚 Resources

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [JSON Schema Specification](https://json-schema.org/specification.html)
- [Plugin Architecture Patterns](https://martinfowler.com/articles/patterns-of-distributed-systems/plugin.html)

---

## 🎯 Mini-Project Progress: Multi-Tool Agent

In this lesson, we've made progress on our mini-project by:
- Designing and implementing the tool registry
- Creating a system for categorizing and discovering tools
- Implementing conversion to LangChain tools
- Setting up the foundation for our multi-tool agent

In the next lesson, we'll continue by:
- Implementing function calling patterns
- Building a multi-tool agent that can use all our tools
- Creating a system for selecting the right tool for a given task
- Implementing error handling and fallback mechanisms

---

> 💡 **Note on Tool Integration**: The tool registry is a key component of our multi-tool agent system. It provides a central place to manage and discover tools, making it easier to build complex agentic systems.

---

Happy coding! 🚀
