# 🔍 Lesson 3: Building a Search Tool

In this lesson, we'll learn how to build a tool that performs web searches using the Serper API with DuckDuckGo as a fallback option. This tool will allow our agents to access up-to-date information from the web.

## 📋 Overview

The Search tool provides a simple interface for performing web searches. It allows you to:

1. Search the web for information on any topic
2. Get structured search results with titles, links, and snippets
3. Use different search types (web, news, images)
4. Fall back to an alternative search engine if the primary one fails
5. Control the number and type of results returned

## 🛠️ Implementation

Our Search tool is built on top of the `BaseTool` class, which provides a standard interface for all tools in our system. The tool uses the Serper API as the primary search engine, with DuckDuckGo as a fallback option. We've integrated LangChain's `GoogleSerperAPIWrapper` for more robust Serper API handling.

### Key Components

1. **ToolResponse**: A Pydantic model for standardizing tool responses
2. **SearchResult**: A Pydantic model for structured search results
3. **SearchTool**: The main class that implements the search functionality
4. **LangChain Integration**: Using LangChain's GoogleSerperAPIWrapper when available
5. **Fallback mechanism**: Logic to switch to DuckDuckGo if Serper fails
6. **Result parsing**: Functions to parse and normalize results from different search engines

### Code Structure

```python
# Try to import LangChain's GoogleSerperAPIWrapper
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
    LANGCHAIN_SERPER_AVAILABLE = True
except ImportError:
    LANGCHAIN_SERPER_AVAILABLE = False

class SearchResult(BaseModel):
    """Model for search results."""
    title: str = Field(..., description="The title of the search result")
    link: str = Field(..., description="The URL of the search result")
    snippet: Optional[str] = Field(None, description="A snippet or description of the search result")
    position: Optional[int] = Field(None, description="The position of the result in the search results")
    source: str = Field("unknown", description="The source of the search result (serper, duckduckgo, etc.)")


class SearchTool(BaseTool):
    """Tool for performing web searches using Serper API with DuckDuckGo as a fallback."""

    def __init__(
        self,
        name: str = "search",
        description: str = "Search the web for information on a given query",
        serper_api_key: Optional[str] = None,
        max_results: int = 5,
        max_retries: int = 3,
        retry_delay: int = 1,
        use_fallback: bool = True
    ):
        # Initialization code...
        # Add LangChain info to metadata
        self.metadata["using_langchain"] = LANGCHAIN_SERPER_AVAILABLE

    def execute(self, **kwargs) -> ToolResponse:
        # Main execution method...

    def _search_with_serper(self, query: str, num_results: int, search_type: str, country: str, language: str) -> List[SearchResult]:
        # Use LangChain's GoogleSerperAPIWrapper if available
        if LANGCHAIN_SERPER_AVAILABLE:
            try:
                # Initialize the wrapper with our API key
                serper = GoogleSerperAPIWrapper(
                    serper_api_key=self.serper_api_key,
                    gl=country,
                    hl=language,
                    type=search_type if search_type != "search" else None
                )

                # Get results from LangChain wrapper
                results_json = serper.results(query)

                # Parse the results
                return self._parse_serper_results(results_json, num_results)
            except Exception as e:
                raise Exception(f"LangChain Serper API error: {str(e)}")

        # Fall back to direct API call if LangChain is not available
        # Direct API implementation...

    def _search_with_duckduckgo(self, query: str, num_results: int) -> List[SearchResult]:
        # DuckDuckGo search implementation...

    def get_schema(self) -> Dict[str, Any]:
        # Schema definition...

    def search(self, query: str, **kwargs) -> List[SearchResult]:
        # Convenience method for searching...
```

## 🚀 Usage Examples

### Basic Search

```python
from module6.code.tools.search_tool import SearchTool

# Create the tool
search_tool = SearchTool()

# Perform a search
results = search_tool.search("What is Python programming language?")

# Print the results
for result in results:
    print(f"Title: {result.title}")
    print(f"URL: {result.link}")
    print(f"Snippet: {result.snippet}")
    print(f"Source: {result.source}")
    print()
```

### Search with Different Parameters

```python
# Search for news with specific parameters
results = search_tool.search(
    query="Latest news about artificial intelligence",
    search_type="news",
    num_results=3,
    country="us",
    language="en"
)

# Print the results
for result in results:
    print(f"Title: {result.title}")
    print(f"URL: {result.link}")
    print(f"Snippet: {result.snippet}")
    print()
```

### Force Using Fallback

```python
# Force using DuckDuckGo instead of Serper
results = search_tool.search(
    query="Best programming languages to learn",
    force_fallback=True
)

# Print the results
for result in results:
    print(f"Title: {result.title}")
    print(f"URL: {result.link}")
    print(f"Source: {result.source}")  # Should be "duckduckgo"
    print()
```

### Using the Execute Method Directly

```python
# Use the execute method for more control
response = search_tool.execute(
    query="How to learn machine learning",
    num_results=5
)

# Check if the execution was successful
if response.success:
    print(f"Found {len(response.result)} results")
    for result in response.result:
        print(f"Title: {result.title}")
        print(f"URL: {result.link}")
else:
    print(f"Error: {response.error}")

# Access metadata
print(f"Query: {response.metadata['query']}")
print(f"Used fallback: {response.metadata['used_fallback']}")
```

## ⚠️ Error Handling and Fallbacks

The Search tool includes sophisticated error handling and fallback mechanisms:

1. **Automatic retries**: Retries failed requests with exponential backoff
2. **Fallback to DuckDuckGo**: If Serper fails or no API key is available
3. **Graceful degradation**: Returns partial results if available
4. **Detailed error messages**: Provides clear error information

Example of the fallback mechanism:

```python
# Try Serper API first if we have an API key and not forcing fallback
results = []
serper_error = None

if self.serper_api_key and not force_fallback:
    try:
        results = self._search_with_serper(...)
    except Exception as e:
        serper_error = str(e)
        print(f"Serper API error: {serper_error}")

# Use DuckDuckGo as fallback if Serper failed or we don't have an API key
if (not results or force_fallback) and self.use_fallback:
    try:
        results = self._search_with_duckduckgo(...)
    except Exception as e:
        # If both search engines failed, return error with both messages
        if serper_error:
            return ToolResponse(
                success=False,
                error=f"All search engines failed. Serper: {serper_error}, DuckDuckGo: {str(e)}"
            )
        else:
            return ToolResponse(
                success=False,
                error=f"DuckDuckGo search failed: {str(e)}"
            )
```

## 🔌 LangChain Integration

Our Search tool integrates with LangChain in two important ways:

### 1. Using GoogleSerperAPIWrapper

The tool uses LangChain's `GoogleSerperAPIWrapper` for more robust handling of the Serper API. This integration provides several benefits:

1. **Standardized interface**: Uses LangChain's well-tested API wrapper
2. **Error handling**: Benefits from LangChain's error handling logic
3. **Compatibility**: Works with other LangChain components
4. **Maintenance**: Automatically benefits from updates to the LangChain library

The tool checks if LangChain is available and uses it if possible:

```python
# Try to import LangChain components
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
    from langchain.tools import Tool as LangChainTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
```

If LangChain is not available, the tool falls back to a direct API implementation:

```python
# Use LangChain's GoogleSerperAPIWrapper if available
if LANGCHAIN_AVAILABLE:
    # Use LangChain wrapper
    serper = GoogleSerperAPIWrapper(...)
    results_json = serper.results(query)
else:
    # Use direct API implementation
    # ...
```

### 2. Converting to a LangChain Tool

The tool can be converted to a LangChain Tool for use with LangChain agents:

```python
def to_langchain_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Any:
    """
    Convert this tool to a LangChain Tool for use with LangChain agents.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not available.")

    # Define a function that formats the results as a string
    def search_and_format(query: str) -> str:
        results = self.search(query)
        # Format results as a string...
        return formatted_results

    # Create and return a LangChain Tool
    return LangChainTool(
        name=name or self.name,
        description=description or self.description,
        func=search_and_format
    )
```

This allows you to use the Search tool with LangChain agents:

```python
# Create a LangChain Tool from our SearchTool
search_langchain_tool = search_tool.to_langchain_tool(
    name="web_search",
    description="Search the web for information on a given query"
)

# Create a simple LLM
llm = OpenAI(temperature=0)

# Initialize the agent
agent = initialize_agent(
    [search_langchain_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What is the capital of France?")
```

## 🔑 API Key Management

The Search tool requires a Serper API key for optimal performance. You can provide it in several ways:

1. **Environment variable**: Set `SERPER_API_KEY` in your environment
2. **Constructor parameter**: Pass `serper_api_key` when creating the tool
3. **.env file**: Use a .env file with the dotenv package

If no API key is available, the tool will automatically fall back to using DuckDuckGo (if `use_fallback` is set to `True`).

## 💰 Cost and Rate Limit Considerations

When using the Serper API, be mindful of:

1. **API usage limits**: Serper has usage limits based on your subscription
2. **Cost per search**: Each search request costs credits
3. **Rate limits**: Too many requests in a short time may be rate-limited

To manage these considerations:

1. **Limit the number of results**: Use `num_results` to control how many results are returned
2. **Cache results**: Consider caching search results for frequently asked questions
3. **Use the fallback**: Enable `use_fallback` to use DuckDuckGo when appropriate

## 🧪 Testing

The Search tool includes comprehensive tests to ensure it works correctly:

1. **Unit tests**: Test individual components
2. **Integration tests**: Test the tool with the actual APIs
3. **Fallback tests**: Verify that the fallback mechanism works

Run the tests with:

```bash
python -m module6.tests.test_search_tool
```

## 📝 Next Steps

Now that you've learned how to use the Search tool, you can:

1. Integrate it with other tools like the OpenAI or Groq tools
2. Build a research agent that can search for information and summarize it
3. Extend the tool to support other search engines or specialized search APIs
4. Create a multi-tool agent that can use the Search tool alongside other tools

In the next lesson, we'll build a Weather tool that uses the OpenWeatherMap API to get weather information.
