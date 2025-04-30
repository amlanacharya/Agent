# 🚀 Module 6: Tool Integration & Function Calling - Lesson 1: Building an OpenAI Tool 🔧

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔍 Understand the concept of tools in agentic systems
- 🧩 Learn how to design a standardized tool interface
- 🔄 Implement a tool for interacting with the OpenAI API
- 📊 Handle API errors and rate limiting gracefully
- 🛠️ Create convenience methods for common use cases

---

## 📚 Introduction to Tools in Agentic Systems

Tools are specialized components that extend an agent's capabilities by allowing it to interact with external systems, APIs, and services. They provide a standardized interface for performing specific tasks, such as generating text, searching the web, or retrieving weather information.

In this module, we'll build a collection of tools that can be used individually or combined into a multi-tool agent. Each tool will follow a consistent interface, making it easy to add new tools or swap out existing ones.

### Key Concepts

A well-designed tool system should have:

1. **Standardized Interface**: All tools should implement a common interface
2. **Error Handling**: Tools should handle errors gracefully and provide useful feedback
3. **Metadata**: Tools should provide information about their capabilities
4. **Schema Definition**: Tools should define their input parameters and output format
5. **Convenience Methods**: Tools should provide simple methods for common use cases

```python
# Example of a standardized tool interface
class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.metadata = {"created_at": time.time()}
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResponse:
        """Execute the tool with the provided parameters."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON Schema for this tool."""
        pass
```

## 🧩 Designing a Tool Interface

Before implementing specific tools, we need to design a standardized interface that all tools will follow. This ensures consistency and makes it easier to integrate tools with agents.

### Base Tool Class

Our `BaseTool` class defines the common interface that all tools must implement:

```python
class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        self.name = name
        self.description = description
        self.metadata = {
            "created_at": time.time()
        }
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the tool with the provided parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResponse: The result of the tool execution
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON Schema for this tool.
        
        Returns:
            Dict[str, Any]: The JSON Schema for this tool
        """
        pass
```

### Tool Response Model

We also need a standardized way to represent tool responses. We'll use a Pydantic model for this:

```python
class ToolResponse(BaseModel):
    """Base model for tool responses."""
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Any = Field(None, description="The result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if the tool execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the execution")
```

## 🔄 Implementing the OpenAI Tool

Now that we have our base classes, let's implement a tool for interacting with the OpenAI API. This tool will allow us to generate text completions and chat responses.

### OpenAI Tool Implementation

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
        """
        Initialize the OpenAI tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            api_key: OpenAI API key (if None, will try to get from environment)
            model: The model to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0-1)
            max_retries: Maximum number of retries on rate limit or error
            retry_delay: Initial delay between retries in seconds
        """
        super().__init__(name, description)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as an argument or as OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Add model info to metadata
        self.metadata["model"] = model
        self.metadata["provider"] = "OpenAI"
```

### Execute Method

The `execute` method is the main entry point for using the tool:

```python
def execute(self, **kwargs) -> ToolResponse:
    """
    Execute the OpenAI tool with the provided parameters.
    
    Args:
        **kwargs: Tool-specific parameters including:
            - messages: List of message dicts for chat completion
            - prompt: Text prompt for completion
            - model: Override the default model
            - max_tokens: Override the default max_tokens
            - temperature: Override the default temperature
            
    Returns:
        ToolResponse: The result of the tool execution
    """
    try:
        # Extract parameters
        messages = kwargs.get("messages")
        prompt = kwargs.get("prompt")
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        # Validate parameters
        if messages is None and prompt is None:
            return ToolResponse(
                success=False,
                error="Either 'messages' or 'prompt' must be provided"
            )
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Handle chat completion
        if messages is not None:
            endpoint = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Make request with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract the response text
                    result = data["choices"][0]["message"]["content"]
                    
                    # Return successful response
                    return ToolResponse(
                        success=True,
                        result=result,
                        metadata={
                            "model": model,
                            "completion_type": "chat",
                            "tokens": data.get("usage", {})
                        }
                    )
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # Handle text completion
        else:
            endpoint = "https://api.openai.com/v1/completions"
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Make request with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract the response text
                    result = data["choices"][0]["text"]
                    
                    # Return successful response
                    return ToolResponse(
                        success=True,
                        result=result,
                        metadata={
                            "model": model,
                            "completion_type": "text",
                            "tokens": data.get("usage", {})
                        }
                    )
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
    
    except Exception as e:
        # Handle any errors
        return ToolResponse(
            success=False,
            error=f"OpenAI tool execution failed: {str(e)}"
        )
```

### Convenience Methods

To make the tool easier to use, we'll add convenience methods for common operations:

```python
def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Convenience method for chat completion.
    
    Args:
        messages: List of message dicts for chat completion
        **kwargs: Additional parameters to pass to execute()
        
    Returns:
        str: The generated response
        
    Raises:
        ValueError: If the execution fails
    """
    response = self.execute(messages=messages, **kwargs)
    if response.success:
        return response.result
    else:
        raise ValueError(response.error)

def complete(self, prompt: str, **kwargs) -> str:
    """
    Convenience method for text completion.
    
    Args:
        prompt: Text prompt for completion
        **kwargs: Additional parameters to pass to execute()
        
    Returns:
        str: The generated response
        
    Raises:
        ValueError: If the execution fails
    """
    response = self.execute(prompt=prompt, **kwargs)
    if response.success:
        return response.result
    else:
        raise ValueError(response.error)
```

### Schema Definition

Finally, we need to implement the `get_schema` method to define the tool's input parameters:

```python
def get_schema(self) -> Dict[str, Any]:
    """
    Get the JSON Schema for this tool.
    
    Returns:
        Dict[str, Any]: The JSON Schema for this tool
    """
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"]
                                },
                                "content": {
                                    "type": "string"
                                }
                            },
                            "required": ["role", "content"]
                        },
                        "description": "List of messages for chat completion"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt for completion"
                    },
                    "model": {
                        "type": "string",
                        "description": f"The model to use (default: {self.model})"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": f"Maximum number of tokens to generate (default: {self.max_tokens})"
                    },
                    "temperature": {
                        "type": "number",
                        "description": f"Temperature for generation (0-1, default: {self.temperature})"
                    }
                },
                "oneOf": [
                    {"required": ["messages"]},
                    {"required": ["prompt"]}
                ]
            }
        }
    }
```

## 📊 Using the OpenAI Tool

Now that we've implemented our OpenAI tool, let's see how to use it in practice.

### Basic Usage

```python
from module6.code.tools.openai_tool import OpenAITool

# Create the tool
openai_tool = OpenAITool(model="gpt-3.5-turbo")

# Generate a text completion
prompt = "Write a short poem about artificial intelligence."
response = openai_tool.complete(prompt)
print(response)

# Generate a chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
response = openai_tool.chat(messages)
print(response)
```

### Advanced Usage

```python
# Using the execute method directly
response = openai_tool.execute(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.2,
    max_tokens=50
)

if response.success:
    print(response.result)
    print(f"Model used: {response.metadata['model']}")
    print(f"Tokens used: {response.metadata['tokens']}")
else:
    print(f"Error: {response.error}")
```

## 🛠️ Putting It All Together

Let's create a complete example that demonstrates how to use the OpenAI tool in a real application:

```python
import os
from dotenv import load_dotenv
from module6.code.tools.openai_tool import OpenAITool, ToolResponse

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate the OpenAI tool."""
    print("OpenAI Tool Demo")
    print("-" * 50)
    
    # Create the tool
    try:
        openai_tool = OpenAITool(
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.7
        )
        print(f"Initialized OpenAI tool with model: {openai_tool.model}")
    except ValueError as e:
        print(f"Error initializing OpenAI tool: {e}")
        print("Make sure you have set the OPENAI_API_KEY environment variable.")
        return
    
    # Example 1: Simple text completion
    print("\nExample 1: Simple Text Completion")
    print("-" * 50)
    prompt = "Write a short poem about artificial intelligence."
    print(f"Prompt: {prompt}")
    
    try:
        response = openai_tool.complete(prompt)
        print("\nResponse:")
        print(response)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 2: Chat completion
    print("\nExample 2: Chat Completion")
    print("-" * 50)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
        {"role": "user", "content": "Tell me about the weather today."}
    ]
    print(f"Messages: {messages}")
    
    try:
        response = openai_tool.chat(messages)
        print("\nResponse:")
        print(response)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example 3: Advanced usage with execute
    print("\nExample 3: Advanced Usage with Execute")
    print("-" * 50)
    try:
        response = openai_tool.execute(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What are the three laws of robotics?"}
            ],
            temperature=0.2,
            max_tokens=200
        )
        
        if response.success:
            print("Response:")
            print(response.result)
            print("\nMetadata:")
            for key, value in response.metadata.items():
                print(f"- {key}: {value}")
        else:
            print(f"Error: {response.error}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

---

## 💪 Practice Exercises

1. **Exercise 1: Error Handling Enhancement**
   - Modify the OpenAI tool to add more specific error handling for different types of API errors
   - Implement a logging system to track API calls and errors
   - Add a method to check the API status before making calls

2. **Exercise 2: Model Parameter Validation**
   - Add validation for the model parameter to ensure it's a valid OpenAI model
   - Implement a method to list available models
   - Add a feature to automatically select the best model based on the task

3. **Exercise 3: JSON Output Format**
   - Add a new method to generate structured JSON outputs
   - Implement schema validation for the JSON output
   - Create a convenience method for extracting specific fields from the JSON response

---

## 🔍 Key Concepts to Remember

1. **Tool Interface**: All tools should implement a common interface with execute and get_schema methods
2. **Tool Response**: Standardize responses with success, result, error, and metadata fields
3. **Error Handling**: Implement robust error handling with retries and exponential backoff
4. **Convenience Methods**: Provide simple methods for common use cases to improve usability
5. **Schema Definition**: Define the tool's input parameters using JSON Schema for better integration

---

## 🚀 Next Steps

In the next lesson, we'll:
- Build a Groq tool for interacting with the Groq API
- Learn how to handle different model parameters
- Implement specialized methods for JSON generation
- Compare the performance of OpenAI and Groq models
- Explore strategies for selecting the right model for different tasks

---

## 📚 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [JSON Schema Specification](https://json-schema.org/specification.html)
- [Python Requests Library](https://docs.python-requests.org/en/latest/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## 🎯 Mini-Project Progress: Multi-Tool Agent

In this lesson, we've made progress on our mini-project by:
- Designing the base tool interface that all tools will implement
- Creating the OpenAI tool for text generation and chat
- Implementing error handling and retry logic
- Defining a standardized response format

In the next lesson, we'll continue by:
- Building the Groq tool as an alternative to OpenAI
- Comparing the performance and capabilities of different models
- Preparing for the integration of multiple tools into a single agent

---

> 💡 **Note on LLM Integration**: This lesson uses real API calls to OpenAI. Make sure you have a valid API key and be mindful of usage costs. For testing purposes, you can use the gpt-3.5-turbo model which is more cost-effective than GPT-4.

---

Happy coding! 🚀
