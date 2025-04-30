# 🧰 Module 6: Tools

This directory contains the implementation of various tools that can be used by agents to interact with external systems and APIs.

## 📋 Overview

Each tool follows a standardized interface defined in `base_tool.py`, making it easy to add new tools or swap out existing ones. The tools are designed to be used individually or combined into a multi-tool agent.

## 🛠️ Available Tools

### BaseTool (`base_tool.py`)
The abstract base class that all tools inherit from. It defines the common interface and provides utility methods.

### OpenAI Tool (`openai_tool.py`)
A tool for interacting with the OpenAI API to generate text and chat responses.

### Groq Tool (`groq_tool.py`)
A tool for interacting with the Groq API to generate text and chat responses.

### Search Tool (`search_tool.py`)
A tool for performing web searches using the Serper API with DuckDuckGo as a fallback.

### Weather Tool (`weather_tool.py`)
A tool for retrieving weather information using the OpenWeatherMap API.

### Alpha Vantage Tool (`alpha_vantage_tool.py`)
A tool for retrieving financial data using the Alpha Vantage API.

## 🚀 Usage

Each tool can be used individually:

```python
from module6.code.tools import OpenAITool

# Create the tool
openai_tool = OpenAITool(model="gpt-3.5-turbo")

# Generate a text completion
prompt = "Write a short poem about artificial intelligence."
response = openai_tool.complete(prompt)
print(response)
```

Or combined into a multi-tool agent:

```python
from module6.code.tools import OpenAITool, SearchTool, WeatherTool

# Create the tools
openai_tool = OpenAITool()
search_tool = SearchTool()
weather_tool = WeatherTool()

# Use the tools together
search_results = search_tool.search("Weather in New York")
weather_info = weather_tool.get_weather_by_location("New York")
summary = openai_tool.complete(f"Summarize this information: {search_results} {weather_info}")
print(summary)
```

## 📊 Tool Response Format

All tools return a standardized `ToolResponse` object with the following fields:

- `success`: Whether the tool execution was successful
- `result`: The result of the tool execution
- `error`: Error message if the tool execution failed
- `metadata`: Additional metadata about the execution

## 🔍 Adding New Tools

To add a new tool:

1. Create a new file for your tool (e.g., `my_tool.py`)
2. Import the `BaseTool` and `ToolResponse` classes
3. Implement the `execute` and `get_schema` methods
4. Add convenience methods for common use cases
5. Update the `__init__.py` file to expose your tool

Example:

```python
from .base_tool import BaseTool, ToolResponse

class MyTool(BaseTool):
    """My custom tool."""
    
    def __init__(self, name="my_tool", description="My custom tool"):
        super().__init__(name, description)
    
    def execute(self, **kwargs) -> ToolResponse:
        # Implement your tool logic here
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        # Define your tool's schema here
        pass
```

## 🧪 Testing

Each tool has a corresponding test file in the `tests` directory. To run the tests:

```bash
python -m unittest discover -s module6/tests
```

Or to test a specific tool:

```bash
python -m unittest module6.tests.test_openai_tool
```
