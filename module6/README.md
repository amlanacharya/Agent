# Module 6: Tool Integration & Function Calling 🛠️

This module focuses on building and integrating tools with language models to create powerful agentic systems. You'll learn how to build individual tools, combine them into a multi-tool agent, and implement function calling patterns.

## 📋 Overview

In this module, we'll explore:

1. Building individual tools for specific tasks
2. Creating a tool registry system
3. Implementing function calling with LLMs
4. Developing multi-tool agents
5. Handling tool selection and routing

## 🧰 Tools Implemented

This module includes implementations of several tools:

- **OpenAI Tool**: For text generation and chat using OpenAI models
- **Groq Tool**: For text generation and chat using Groq models
- **Search Tool**: For web search using Serper API
- **Weather Tool**: For weather information using OpenWeatherMap API
- **News Tool**: For retrieving news articles using News API
- **Finance Tool**: For financial data using AlphaVantage API

## 📚 Lessons

1. [Lesson 1: Building an OpenAI Tool](lessons/lesson1_openai_tool.md)
2. Lesson 2: Building a Groq Tool
3. Lesson 3: Building a Search Tool
4. Lesson 4: Building a Weather Tool
5. Lesson 5: Building a News Tool
6. Lesson 6: Creating a Tool Registry
7. Lesson 7: Implementing a Multi-Tool Agent

## 🚀 Getting Started

To get started with this module:

1. Make sure you have the required API keys in your `.env` file
2. Install the required dependencies
3. Follow the lessons in order
4. Run the example scripts to see the tools in action

## 🔑 Required API Keys

This module requires several API keys to function properly:

- `OPENAI_API_KEY`: For the OpenAI tool
- `GROQ_API_KEY`: For the Groq tool
- `SERPER_API_KEY`: For the Search tool
- `OPENWEATHERMAP_API_KEY`: For the Weather tool
- `NEWS_API_KEY`: For the News tool
- `ALPHAVANTAGE_API_KEY`: For the Finance tool

You can set these in your `.env` file or as environment variables.

## 📦 Dependencies

This module requires the following dependencies:

```
requests
python-dotenv
pydantic
```

For enhanced functionality, the following optional dependencies are recommended:

```
langchain-community  # For GoogleSerperAPIWrapper and other utilities
langchain-groq       # For Groq integration
langchain-openai     # For OpenAI integration
```

## 🧪 Running Tests

To run the tests for this module:

```bash
# Run all tests
python -m unittest discover -s module6/tests

# Run a specific test
python -m module6.tests.test_openai_tool
```

## 📝 Examples

Check out the example scripts in the `module6/code/examples` directory to see how to use the tools:

- `openai_example.py`: Examples of using the OpenAI tool
- `groq_example.py`: Examples of using the Groq tool
- `search_example.py`: Examples of using the Search tool
- `weather_example.py`: Examples of using the Weather tool
- `multi_tool_example.py`: Examples of using multiple tools together
