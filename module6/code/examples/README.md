# 📚 Module 6: Examples

This directory contains example scripts demonstrating how to use the tools implemented in the `tools` directory.

## 📋 Available Examples

### OpenAI Tool Example (`openai_example.py`)
Demonstrates how to use the OpenAI tool for text generation and chat.

### Groq Tool Example (`groq_example.py`)
Demonstrates how to use the Groq tool for text generation and chat.

### Search Tool Example (`search_example.py`)
Demonstrates how to use the Search tool to perform web searches.

### Weather Tool Example (`weather_example.py`)
Demonstrates how to use the Weather tool to retrieve weather information.

### Multi-Tool Agent Example (`multi_tool_agent_example.py`)
Demonstrates how to combine multiple tools into a single agent.

## 🚀 Running the Examples

To run an example:

```bash
# Make sure you're in the root directory of the project
python -m module6.code.examples.openai_example
```

## 🔑 API Keys

Most examples require API keys to function properly. You can set these in your `.env` file or as environment variables:

```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
```

## 📝 Creating Your Own Examples

Feel free to create your own examples to explore different ways of using the tools. Here's a template to get you started:

```python
"""
Example script demonstrating [description of what your example does].
"""

import os
from dotenv import load_dotenv
from module6.code.tools import [ToolName]

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate the [ToolName] tool."""
    print("[ToolName] Demo")
    print("-" * 50)
    
    # Create the tool
    tool = [ToolName]()
    
    # Use the tool
    # ...
    
    print("Example Complete")

if __name__ == "__main__":
    main()
```
