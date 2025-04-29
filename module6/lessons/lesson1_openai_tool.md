# 🔧 Lesson 1: Building an OpenAI Tool

In this lesson, we'll learn how to build a tool that interacts with the OpenAI API to generate text and chat responses. This tool will serve as a foundation for our multi-tool agent system.

## 📋 Overview

The OpenAI tool provides a simple interface for interacting with OpenAI's language models. It allows you to:

1. Generate text completions from prompts
2. Have chat-based conversations with the model
3. Control parameters like temperature and token limits
4. Handle rate limiting and retries automatically

## 🛠️ Implementation

Our OpenAI tool is built on top of the `BaseTool` class, which provides a standard interface for all tools in our system. The tool uses the OpenAI API to generate text and chat responses.

### Key Components

1. **ToolResponse**: A Pydantic model for standardizing tool responses
2. **OpenAITool**: The main class that implements the OpenAI tool functionality
3. **Error handling and retries**: Logic to handle API errors and rate limits
4. **Convenience methods**: Simple methods for common use cases

### Code Structure

```python
class OpenAITool(BaseTool):
    """Tool for interacting with OpenAI API for chat and text generation."""
    
    def __init__(
        self,
        name: str = "openai",
        description: str = "Generate text or chat responses using OpenAI models",
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        # Initialization code...
    
    def execute(self, **kwargs) -> ToolResponse:
        # Main execution method...
    
    def get_schema(self) -> Dict[str, Any]:
        # Schema definition...
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convenience method for chat...
    
    def complete(self, prompt: str, **kwargs) -> str:
        # Convenience method for completion...
```

## 🚀 Usage Examples

### Basic Text Completion

```python
from module6.code.tools.openai_tool import OpenAITool

# Create the tool
openai_tool = OpenAITool(model="gpt-3.5-turbo")

# Generate a text completion
prompt = "Write a short poem about artificial intelligence."
response = openai_tool.complete(prompt)
print(response)
```

### Chat Completion

```python
# Create a chat conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Get a response
response = openai_tool.chat(messages)
print(response)
```

### Advanced Usage with Parameters

```python
# Override default parameters
response = openai_tool.complete(
    prompt="Explain quantum computing in simple terms.",
    temperature=0.3,
    max_tokens=200,
    model="gpt-4"  # Use a different model
)
print(response)
```

### Using the Execute Method Directly

```python
# Use the execute method for more control
response = openai_tool.execute(
    prompt="What are some good books to read?",
    temperature=0.5
)

# Check if the execution was successful
if response.success:
    print(response.result)
    print(f"Token usage: {response.metadata.get('usage', {})}")
else:
    print(f"Error: {response.error}")
```

## ⚠️ Error Handling

The OpenAI tool includes built-in error handling for common issues:

1. **Rate limiting**: Automatically retries with exponential backoff
2. **API errors**: Provides clear error messages
3. **Invalid inputs**: Validates inputs before making API calls

Example of handling errors:

```python
try:
    response = openai_tool.complete("My prompt")
    print(response)
except Exception as e:
    print(f"An error occurred: {str(e)}")
```

## 🔑 API Key Management

The OpenAI tool requires an API key to function. You can provide it in several ways:

1. **Environment variable**: Set `OPENAI_API_KEY` in your environment
2. **Constructor parameter**: Pass `api_key` when creating the tool
3. **.env file**: Use a .env file with the dotenv package

Example with explicit API key:

```python
openai_tool = OpenAITool(
    api_key="your-api-key-here",
    model="gpt-3.5-turbo"
)
```

## 💰 Cost Considerations

When using the OpenAI API, be mindful of costs:

1. **Choose cheaper models** for development and testing (e.g., gpt-3.5-turbo instead of gpt-4)
2. **Limit max_tokens** to control response length
3. **Monitor usage** through the metadata returned in responses

## 🧪 Testing

The OpenAI tool includes comprehensive tests to ensure it works correctly:

1. **Unit tests**: Test individual components
2. **Integration tests**: Test the tool with the actual API
3. **Error handling tests**: Verify that errors are handled properly

Run the tests with:

```bash
python -m module6.tests.test_openai_tool
```

## 📝 Next Steps

Now that you've learned how to use the OpenAI tool, you can:

1. Experiment with different models and parameters
2. Integrate the tool into your applications
3. Build more complex tools that use the OpenAI tool as a component
4. Create a multi-tool agent that can use the OpenAI tool alongside other tools

In the next lesson, we'll build a Groq tool that provides similar functionality but uses the Groq API instead of OpenAI.
