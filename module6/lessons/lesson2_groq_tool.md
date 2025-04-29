# 🔧 Lesson 2: Building a Groq Tool

In this lesson, we'll learn how to build a tool that interacts with the Groq API to generate text and chat responses. Groq is known for its high-performance inference services, offering fast response times for language model queries.

## 📋 Overview

The Groq tool provides a simple interface for interacting with Groq's language models. It allows you to:

1. Generate text completions from prompts
2. Have chat-based conversations with the model
3. Generate structured JSON outputs
4. Control parameters like temperature and token limits
5. Handle rate limiting and retries automatically

## 🛠️ Implementation

Our Groq tool is built on top of the `BaseTool` class, which provides a standard interface for all tools in our system. The tool uses the Groq API to generate text and chat responses.

### Key Components

1. **ToolResponse**: A Pydantic model for standardizing tool responses
2. **GroqTool**: The main class that implements the Groq tool functionality
3. **Retry mechanism**: Advanced retry logic with exponential backoff
4. **JSON generation**: Special support for generating structured JSON outputs
5. **Convenience methods**: Simple methods for common use cases

### Code Structure

```python
class GroqTool(BaseTool):
    """Tool for interacting with Groq API for chat and text generation."""
    
    def __init__(
        self,
        name: str = "groq",
        description: str = "Generate text or chat responses using Groq models",
        api_key: Optional[str] = None,
        model: str = "llama3-8b-8192",
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
        
    def generate_json(self, prompt_or_messages: Union[str, List[Dict[str, str]]], schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        # Convenience method for JSON generation...
```

## 🚀 Usage Examples

### Basic Text Completion

```python
from module6.code.tools.groq_tool import GroqTool

# Create the tool
groq_tool = GroqTool(model="llama3-8b-8192")

# Generate a text completion
prompt = "Write a short poem about artificial intelligence."
response = groq_tool.complete(prompt)
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
response = groq_tool.chat(messages)
print(response)
```

### JSON Generation

```python
# Define a schema for the output
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age", "occupation", "skills"]
}

# Generate structured JSON output
prompt = "Extract information about John Doe, a 35-year-old software engineer who knows Python, JavaScript, and SQL."
result = groq_tool.generate_json(prompt, schema)
print(result)
```

### Advanced Usage with Parameters

```python
# Override default parameters
response = groq_tool.complete(
    prompt="Explain quantum computing in simple terms.",
    temperature=0.3,
    max_tokens=200,
    model="mixtral-8x7b-32768"  # Use a different model
)
print(response)
```

### Using the Execute Method Directly

```python
# Use the execute method for more control
response = groq_tool.execute(
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

## ⚠️ Error Handling and Retries

The Groq tool includes sophisticated error handling and retry logic:

1. **Exponential backoff**: Automatically increases wait time between retries
2. **Jitter**: Adds randomness to retry timing to prevent thundering herd problems
3. **Status code handling**: Intelligently handles different HTTP status codes
4. **Retry-After header**: Respects the Retry-After header if provided by the API

Example of the retry decorator:

```python
@retry_with_exponential_backoff(max_retries=5, initial_delay=1.0)
def _make_api_call(self, url, headers, payload):
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    return response.json()
```

## 🔑 API Key Management

The Groq tool requires an API key to function. You can provide it in several ways:

1. **Environment variable**: Set `GROQ_API_KEY` in your environment
2. **Constructor parameter**: Pass `api_key` when creating the tool
3. **.env file**: Use a .env file with the dotenv package

Example with explicit API key:

```python
groq_tool = GroqTool(
    api_key="your-api-key-here",
    model="llama3-8b-8192"
)
```

## 💰 Cost and Performance Considerations

When using the Groq API, consider:

1. **Model selection**: Different models have different performance characteristics and costs
2. **Token limits**: Control costs by limiting max_tokens
3. **Throughput**: Groq is known for high-throughput, low-latency inference
4. **Rate limits**: Be aware of Groq's rate limits for your account tier

## 🧪 Testing

The Groq tool includes comprehensive tests to ensure it works correctly:

1. **Unit tests**: Test individual components
2. **Integration tests**: Test the tool with the actual API
3. **JSON generation tests**: Verify structured output capabilities

Run the tests with:

```bash
python -m module6.tests.test_groq_tool
```

## 📝 Next Steps

Now that you've learned how to use the Groq tool, you can:

1. Experiment with different Groq models
2. Compare performance between Groq and OpenAI
3. Build applications that leverage Groq's high-performance inference
4. Integrate the Groq tool with other tools in a multi-tool agent

In the next lesson, we'll build a Search tool that uses the Serper API to perform web searches.
