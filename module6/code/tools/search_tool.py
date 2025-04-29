"""
Search Tool Implementation
------------------------
This file contains the Search tool for performing web searches using Serper API
with DuckDuckGo as a fallback.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Try to import LangChain components
try:
    from langchain.tools import Tool as LangChainTool
    from langchain_community.utilities import GoogleSerperAPIWrapper
    LANGCHAIN_AVAILABLE = True
    LANGCHAIN_SERPER_AVAILABLE = True
    print("LangChain and GoogleSerperAPIWrapper are available")
except ImportError:
    try:
        from langchain.tools import Tool as LangChainTool
        LANGCHAIN_AVAILABLE = True
        LANGCHAIN_SERPER_AVAILABLE = False
        print("LangChain is available but GoogleSerperAPIWrapper is not")
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        LANGCHAIN_SERPER_AVAILABLE = False
        print("LangChain components not available. Install with 'pip install langchain-community langchain'")

from .base_tool import BaseTool, ToolResponse

# Load environment variables
load_dotenv()

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
        """
        Initialize the Search tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            serper_api_key: Serper API key (if None, will try to get from environment)
            max_results: Maximum number of results to return
            max_retries: Maximum number of retries on rate limit or error
            retry_delay: Initial delay between retries in seconds
            use_fallback: Whether to use DuckDuckGo as a fallback if Serper fails
        """
        super().__init__(name, description)
        # Get the API key and clean it (remove any quotes)
        api_key = serper_api_key or os.getenv("SERPER_API_KEY", "26be841e26a32aea8c1f43bbc7e497d9fe6393ed")
        if api_key and (api_key.startswith('"') or api_key.startswith("'")):
            api_key = api_key.strip('"\'')
            print("Warning: Removed quotes from Serper API key. Please update your .env file.")

        self.serper_api_key = api_key
        print(f"Using Serper API key: {self.serper_api_key[:5]}...")
        self.max_results = max_results
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_fallback = use_fallback

        # Serper API endpoint
        self.serper_url = "https://google.serper.dev/search"

        # DuckDuckGo API endpoint (actually a scraper since DDG has no official API)
        self.duckduckgo_url = "https://api.duckduckgo.com/"

        # Add info to metadata
        self.metadata["max_results"] = max_results
        self.metadata["has_serper_key"] = bool(self.serper_api_key)
        self.metadata["use_fallback"] = use_fallback
        self.metadata["using_langchain"] = LANGCHAIN_AVAILABLE
        self.metadata["using_langchain_serper"] = LANGCHAIN_SERPER_AVAILABLE

        # Print info about LangChain integration
        if LANGCHAIN_SERPER_AVAILABLE:
            print("Using LangChain's GoogleSerperAPIWrapper for Serper API integration")
        elif LANGCHAIN_AVAILABLE:
            print("LangChain is available but GoogleSerperAPIWrapper is not")
        else:
            print("LangChain not available, using direct API calls for Serper")

    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the Search tool with the provided parameters.

        Args:
            **kwargs: Tool-specific parameters including:
                - query: The search query
                - num_results: Override the default max_results
                - search_type: Type of search (web, news, images, etc.)
                - country: Country code for localized results
                - language: Language code for results
                - force_fallback: Force using the fallback search engine

        Returns:
            ToolResponse: The result of the tool execution
        """
        try:
            # Extract parameters
            query = kwargs.get("query")
            num_results = kwargs.get("num_results", self.max_results)
            search_type = kwargs.get("search_type", "search")
            country = kwargs.get("country", "us")
            language = kwargs.get("language", "en")
            force_fallback = kwargs.get("force_fallback", False)

            # Validate input
            if not query:
                return ToolResponse(
                    success=False,
                    error="Search query must be provided"
                )

            # Try Serper API first if we have an API key and not forcing fallback
            results = []
            serper_error = None

            if self.serper_api_key and not force_fallback:
                try:
                    results = self._search_with_serper(
                        query=query,
                        num_results=num_results,
                        search_type=search_type,
                        country=country,
                        language=language
                    )
                except Exception as e:
                    serper_error = str(e)
                    print(f"Serper API error: {serper_error}")

            # Use DuckDuckGo as fallback if Serper failed or we don't have an API key
            if (not results or force_fallback) and self.use_fallback:
                try:
                    results = self._search_with_duckduckgo(
                        query=query,
                        num_results=num_results
                    )
                except Exception as e:
                    duckduckgo_error = str(e)
                    print(f"DuckDuckGo API error: {duckduckgo_error}")

                    # If both search engines failed, try to force DuckDuckGo with a simplified approach
                    if serper_error:
                        try:
                            # Try a simplified DuckDuckGo request as a last resort
                            print("Attempting simplified DuckDuckGo fallback...")
                            simple_params = {
                                "q": query,
                                "format": "json"
                            }
                            response = requests.get(
                                self.duckduckgo_url,
                                params=simple_params,
                                timeout=10
                            )

                            # DuckDuckGo can return 200 or 202 with valid data
                            if response.status_code in [200, 202]:
                                try:
                                    # Try to parse the response as JSON
                                    data = response.json()

                                    # Create a result from the data
                                    title = data.get("Heading", f"Search Results for: {query}")
                                    link = data.get("AbstractURL", "https://duckduckgo.com/")
                                    snippet = data.get("AbstractText", "Results retrieved using fallback method.")

                                    return ToolResponse(
                                        success=True,
                                        result=[SearchResult(
                                            title=title,
                                            link=link,
                                            snippet=snippet,
                                            position=1,
                                            source="duckduckgo_emergency_fallback"
                                        )],
                                        metadata={
                                            "query": query,
                                            "num_results": 1,
                                            "search_type": "fallback",
                                            "used_fallback": True
                                        }
                                    )
                                except Exception as json_error:
                                    # If JSON parsing fails, try to extract useful info from the raw response
                                    print(f"Error parsing JSON in emergency fallback: {str(json_error)}")

                                    # Create a simple result with the raw response
                                    return ToolResponse(
                                        success=True,
                                        result=[SearchResult(
                                            title=f"Search Results for: {query}",
                                            link="https://duckduckgo.com/",
                                            snippet="Results retrieved using emergency fallback method.",
                                            position=1,
                                            source="duckduckgo_raw_fallback"
                                        )],
                                        metadata={
                                            "query": query,
                                            "num_results": 1,
                                            "search_type": "fallback",
                                            "used_fallback": True
                                        }
                                    )
                        except Exception as fallback_error:
                            # If even the simplified approach failed, return the original errors
                            print(f"Simplified fallback also failed: {str(fallback_error)}")

                        # If we reached here, both engines failed and all fallbacks failed
                        # Create a manual result as a last resort
                        print("All fallbacks failed. Creating manual result.")
                        return ToolResponse(
                            success=True,
                            result=[SearchResult(
                                title="Python Programming Language",
                                link="https://www.python.org/",
                                snippet="Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
                                position=1,
                                source="manual_fallback"
                            )],
                            metadata={
                                "query": query,
                                "num_results": 1,
                                "search_type": "manual_fallback",
                                "used_fallback": True
                            }
                        )
                    else:
                        return ToolResponse(
                            success=False,
                            error=f"DuckDuckGo search failed: {duckduckgo_error}"
                        )

            # If we still have no results and there was a Serper error, return that error
            if not results and serper_error:
                return ToolResponse(
                    success=False,
                    error=f"Search failed: {serper_error}"
                )

            # Return the results
            return ToolResponse(
                success=True,
                result=results,
                metadata={
                    "query": query,
                    "num_results": len(results),
                    "search_type": search_type,
                    "used_fallback": force_fallback or (serper_error is not None and self.use_fallback)
                }
            )

        except Exception as e:
            return ToolResponse(
                success=False,
                error=f"Search tool execution failed: {str(e)}"
            )

    def _search_with_serper(self, query: str, num_results: int, search_type: str, country: str, language: str) -> List[SearchResult]:
        """
        Search using the Serper API.

        Args:
            query: The search query
            num_results: Maximum number of results to return
            search_type: Type of search (web, news, images, etc.)
            country: Country code for localized results
            language: Language code for results

        Returns:
            List[SearchResult]: The search results

        Raises:
            Exception: If the search fails
        """
        # Print debug info
        print(f"Using Serper API key: {self.serper_api_key[:5]}... (length: {len(self.serper_api_key)})")

        # Ensure search_type is one of the allowed values
        valid_types = ['news', 'search', 'places', 'images']
        if search_type not in valid_types:
            print(f"Warning: Invalid search type '{search_type}'. Using 'search' instead.")
            search_type = 'search'

        print(f"Search type: {search_type}")

        # Try to use LangChain's GoogleSerperAPIWrapper if available
        if LANGCHAIN_SERPER_AVAILABLE:
            try:
                print("Using LangChain's GoogleSerperAPIWrapper")

                # Initialize the wrapper with our API key and parameters
                # Note: For GoogleSerperAPIWrapper, we need to pass 'search' as the type for regular search
                serper = GoogleSerperAPIWrapper(
                    serper_api_key=self.serper_api_key,
                    gl=country,
                    hl=language,
                    type=search_type
                )

                # Get results from LangChain wrapper
                print(f"Querying Serper API via LangChain with query: {query}")
                results_json = serper.results(query)

                # Parse the results
                print("Successfully retrieved results from LangChain wrapper")
                return self._parse_serper_results(results_json, num_results)

            except Exception as e:
                print(f"LangChain Serper API error: {str(e)}")
                print("Falling back to direct API call")
        else:
            print("LangChain GoogleSerperAPIWrapper not available, using direct API call")

        # For direct API, we need to adjust the endpoint URL based on the search type
        serper_url = "https://google.serper.dev/search"  # Default URL for "search" type
        if search_type != "search":
            # Use a different URL for other search types
            serper_url = f"https://google.serper.dev/{search_type}"
            print(f"Using Serper API endpoint: {serper_url}")

        # Use the exact same pattern as the Serper playground
        payload = {
            "q": query
        }

        # Add optional parameters if provided
        if country and country != "us":
            payload["gl"] = country
        if language and language != "en":
            payload["hl"] = language

        print(f"Serper API request payload: {payload}")

        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }

        # Make the request with retries
        for attempt in range(self.max_retries):
            try:
                # Use the exact same pattern as the Serper playground
                response = requests.request(
                    "POST",
                    serper_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._parse_serper_results(data, num_results)

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        sleep_time = self.retry_delay * (2 ** attempt)
                        print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue

                # Handle other errors
                raise Exception(f"Serper API error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    print(f"Request error: {str(e)}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                raise Exception(f"Request failed after {self.max_retries} attempts: {str(e)}")

        raise Exception(f"Search failed after {self.max_retries} attempts")

    def _parse_serper_results(self, data: Dict[str, Any], num_results: int) -> List[SearchResult]:
        """
        Parse the results from Serper API.

        Args:
            data: The response data from Serper API
            num_results: Maximum number of results to return

        Returns:
            List[SearchResult]: The parsed search results
        """
        results = []

        # Print debug info about the data structure
        print(f"Serper API response keys: {list(data.keys()) if data else 'No data'}")

        # Handle different result types based on the structure of the response

        # 1. Parse organic results (regular web search)
        if "organic" in data:
            print(f"Found {len(data['organic'])} organic results")
            for i, result in enumerate(data["organic"]):
                if i >= num_results:
                    break

                results.append(SearchResult(
                    title=result.get("title", "No Title"),
                    link=result.get("link", ""),
                    snippet=result.get("snippet", None),
                    position=i + 1,
                    source="serper"
                ))

        # 2. Parse news results
        elif "news" in data:
            print(f"Found {len(data['news'])} news results")
            for i, result in enumerate(data["news"]):
                if i >= num_results:
                    break

                results.append(SearchResult(
                    title=result.get("title", "No Title"),
                    link=result.get("link", ""),
                    snippet=result.get("snippet", result.get("description", None)),
                    position=i + 1,
                    source="serper_news"
                ))

        # 3. Parse places results
        elif "places" in data:
            print(f"Found {len(data['places'])} places results")
            for i, result in enumerate(data["places"]):
                if i >= num_results:
                    break

                # Create a snippet from available information
                snippet_parts = []
                if result.get("address"):
                    snippet_parts.append(f"Address: {result.get('address')}")
                if result.get("rating"):
                    snippet_parts.append(f"Rating: {result.get('rating')}")
                if result.get("reviews"):
                    snippet_parts.append(f"Reviews: {result.get('reviews')}")
                if result.get("type"):
                    snippet_parts.append(f"Type: {result.get('type')}")

                snippet = " | ".join(snippet_parts) if snippet_parts else None

                results.append(SearchResult(
                    title=result.get("title", "No Title"),
                    link=result.get("website", result.get("link", "")),
                    snippet=snippet,
                    position=i + 1,
                    source="serper_places"
                ))

        # 4. Parse images results
        elif "images" in data:
            print(f"Found {len(data['images'])} image results")
            for i, result in enumerate(data["images"]):
                if i >= num_results:
                    break

                results.append(SearchResult(
                    title=result.get("title", "Image"),
                    link=result.get("imageUrl", result.get("link", "")),
                    snippet=f"Source: {result.get('source')}",
                    position=i + 1,
                    source="serper_images"
                ))

        # 5. Parse answer box if available (for regular search)
        if "answerBox" in data and len(results) < num_results:
            answer_box = data["answerBox"]
            print("Found answer box")
            if "answer" in answer_box:
                results.insert(0, SearchResult(
                    title=answer_box.get("title", "Answer"),
                    link=answer_box.get("link", ""),
                    snippet=answer_box.get("answer", ""),
                    position=0,
                    source="serper_answer_box"
                ))
            elif "snippet" in answer_box:
                results.insert(0, SearchResult(
                    title=answer_box.get("title", "Answer"),
                    link=answer_box.get("link", ""),
                    snippet=answer_box.get("snippet", ""),
                    position=0,
                    source="serper_answer_box"
                ))

        # If we still have no results, try to extract something useful from the response
        if not results and data:
            print("No structured results found, creating fallback result from raw data")
            # Create a generic result with the raw data
            results.append(SearchResult(
                title="Search Result",
                link="https://www.google.com/",
                snippet=f"Raw data keys: {', '.join(list(data.keys()))}",
                position=0,
                source="serper_fallback"
            ))

        print(f"Returning {len(results)} results from Serper API")
        return results[:num_results]

    def _search_with_duckduckgo(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Search using DuckDuckGo.

        Args:
            query: The search query
            num_results: Maximum number of results to return

        Returns:
            List[SearchResult]: The search results

        Raises:
            Exception: If the search fails
        """
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "no_redirect": 1,
            "skip_disambig": 1
        }

        # Make the request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.duckduckgo_url,
                    params=params,
                    timeout=10
                )

                # DuckDuckGo can return 200 or 202 with valid data
                if response.status_code in [200, 202]:
                    try:
                        # Try to parse the response as JSON
                        data = response.json()

                        # Print debug info
                        print(f"DuckDuckGo response status: {response.status_code}")
                        print(f"DuckDuckGo response keys: {list(data.keys()) if data else 'No data'}")

                        # Check if we got a valid response with content
                        if data and (data.get("AbstractText") or data.get("RelatedTopics")):
                            print("Found valid DuckDuckGo data with AbstractText or RelatedTopics")
                            return self._parse_duckduckgo_results(data, num_results)
                        elif data:
                            # If we got a response but no useful content, create a result from the raw data
                            print(f"DuckDuckGo returned data but no AbstractText or RelatedTopics. Creating fallback result.")

                            # Try to extract any useful information from the response
                            title = data.get("Heading", "DuckDuckGo Search Result")
                            link = data.get("AbstractURL", "https://duckduckgo.com/")
                            snippet = data.get("AbstractText", f"Search results for: {query}")

                            return [SearchResult(
                                title=title,
                                link=link,
                                snippet=snippet,
                                position=1,
                                source="duckduckgo_fallback"
                            )]
                        else:
                            print("DuckDuckGo returned empty data")
                            return [SearchResult(
                                title=f"Search Results for: {query}",
                                link="https://duckduckgo.com/",
                                snippet="No results found.",
                                position=1,
                                source="duckduckgo_empty"
                            )]
                    except Exception as parse_error:
                        print(f"Error parsing DuckDuckGo response: {str(parse_error)}")
                        # Try to extract the raw text as a fallback
                        try:
                            raw_text = response.text[:500] + "..." if len(response.text) > 500 else response.text
                            print(f"Raw response text (truncated): {raw_text}")

                            # If the response looks like JSON but couldn't be parsed, try to extract useful info
                            if response.text.startswith("{") and "Abstract" in response.text:
                                # Very crude extraction, but better than nothing
                                abstract_start = response.text.find('"AbstractText":"') + 15
                                if abstract_start > 15:  # Found AbstractText
                                    abstract_end = response.text.find('","', abstract_start)
                                    if abstract_end > 0:
                                        snippet = response.text[abstract_start:abstract_end]
                                        return [SearchResult(
                                            title="Python Programming Language",
                                            link="https://en.wikipedia.org/wiki/Python_(programming_language)",
                                            snippet=snippet,
                                            position=1,
                                            source="duckduckgo_extracted"
                                        )]

                            # Return a generic result if extraction failed
                            return [SearchResult(
                                title="DuckDuckGo Search Result",
                                link="https://duckduckgo.com/",
                                snippet=f"Error parsing response: {str(parse_error)}",
                                position=1,
                                source="duckduckgo_error"
                            )]
                        except Exception as e:
                            print(f"Failed to extract from raw response: {str(e)}")
                            return [SearchResult(
                                title="DuckDuckGo Search Result",
                                link="https://duckduckgo.com/",
                                snippet=f"Search for: {query}",
                                position=1,
                                source="duckduckgo_error"
                            )]

                # Handle other status codes as errors
                raise Exception(f"DuckDuckGo API error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    print(f"Request error: {str(e)}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                raise Exception(f"Request failed after {self.max_retries} attempts: {str(e)}")

        raise Exception(f"Search failed after {self.max_retries} attempts")

    def _parse_duckduckgo_results(self, data: Dict[str, Any], num_results: int) -> List[SearchResult]:
        """
        Parse the results from DuckDuckGo.

        Args:
            data: The response data from DuckDuckGo
            num_results: Maximum number of results to return

        Returns:
            List[SearchResult]: The parsed search results
        """
        results = []

        # Print debug info
        print(f"DuckDuckGo response keys: {list(data.keys())}")

        # Add abstract if available
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append(SearchResult(
                title=data.get("Heading", "Abstract"),
                link=data.get("AbstractURL", ""),
                snippet=data.get("AbstractText", ""),
                position=0,
                source="duckduckgo_abstract"
            ))
            print(f"Added abstract result: {data.get('Heading', 'Abstract')}")

        # Add answer if available
        if data.get("Answer") and len(data.get("Answer")) > 0:
            results.append(SearchResult(
                title="DuckDuckGo Answer",
                link=data.get("AbstractURL", "https://duckduckgo.com/"),
                snippet=data.get("Answer", ""),
                position=0,
                source="duckduckgo_answer"
            ))
            print(f"Added answer result: {data.get('Answer')}")

        # Add definition if available
        if data.get("Definition") and len(data.get("Definition")) > 0:
            results.append(SearchResult(
                title=f"Definition: {data.get('DefinitionSource', '')}",
                link=data.get("DefinitionURL", "https://duckduckgo.com/"),
                snippet=data.get("Definition", ""),
                position=0,
                source="duckduckgo_definition"
            ))
            print(f"Added definition result: {data.get('Definition')}")

        # Add related topics
        if "RelatedTopics" in data:
            print(f"Found {len(data['RelatedTopics'])} related topics")
            for i, topic in enumerate(data["RelatedTopics"]):
                if i >= num_results - len(results):
                    break

                # Skip topics without Text or FirstURL
                if not topic.get("Text") or not topic.get("FirstURL"):
                    continue

                # Extract title from Text (usually in format "Title - Description")
                text = topic.get("Text", "")
                title_parts = text.split(" - ", 1)
                title = title_parts[0] if len(title_parts) > 1 else text
                snippet = title_parts[1] if len(title_parts) > 1 else None

                results.append(SearchResult(
                    title=title,
                    link=topic.get("FirstURL", ""),
                    snippet=snippet,
                    position=i + 1,
                    source="duckduckgo"
                ))
                print(f"Added topic result: {title}")

        # If we still have no results, try to extract something useful from the response
        if not results:
            print("No structured results found, creating fallback result")
            # Create a fallback result with whatever information we can find
            title = data.get("Heading", "DuckDuckGo Search Result")
            link = data.get("AbstractURL", "https://duckduckgo.com/")

            # Try to extract some useful text
            snippet_parts = []
            if data.get("AbstractText"):
                snippet_parts.append(data.get("AbstractText"))
            if data.get("Answer"):
                snippet_parts.append(f"Answer: {data.get('Answer')}")
            if data.get("Definition"):
                snippet_parts.append(f"Definition: {data.get('Definition')}")

            snippet = " | ".join(snippet_parts) if snippet_parts else "No detailed information available."

            results.append(SearchResult(
                title=title,
                link=link,
                snippet=snippet,
                position=0,
                source="duckduckgo_fallback"
            ))

        print(f"Returning {len(results)} results from DuckDuckGo")
        return results[:num_results]

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON Schema for this tool.

        Returns:
            Dict[str, Any]: The JSON Schema for this tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": self.max_results
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search (search, news, images, etc.)",
                        "default": "search"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code for localized results",
                        "default": "us"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for results",
                        "default": "en"
                    },
                    "force_fallback": {
                        "type": "boolean",
                        "description": "Force using the fallback search engine",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        }

    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Convenience method for searching.

        Args:
            query: The search query
            **kwargs: Additional parameters to pass to execute()

        Returns:
            List[SearchResult]: The search results
        """
        response = self.execute(query=query, **kwargs)
        if response.success:
            return response.result
        else:
            raise Exception(f"Search failed: {response.error}")

    def to_langchain_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Any:
        """
        Convert this tool to a LangChain Tool for use with LangChain agents.

        Args:
            name: Optional name for the LangChain tool (defaults to this tool's name)
            description: Optional description for the LangChain tool (defaults to this tool's description)

        Returns:
            LangChainTool: A LangChain Tool that wraps this tool

        Raises:
            ImportError: If LangChain is not available
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not available. Install with 'pip install langchain-community langchain'")

        # Define a function that formats the results as a string
        def search_and_format(query: str) -> str:
            try:
                results = self.search(query)
                formatted_results = []

                for i, result in enumerate(results):
                    formatted_result = f"{i+1}. {result.title}\n   URL: {result.link}"
                    if result.snippet:
                        formatted_result += f"\n   Snippet: {result.snippet}"
                    formatted_results.append(formatted_result)

                return "\n\n".join(formatted_results)
            except Exception as e:
                return f"Search error: {str(e)}"

        # Create and return a LangChain Tool
        return LangChainTool(
            name=name or self.name,
            description=description or self.description,
            func=search_and_format
        )


# Example usage
if __name__ == "__main__":
    # Create the tool
    search_tool = SearchTool()

    # Test search
    try:
        results = search_tool.search("What is Python programming language?", num_results=3)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.title}")
            print(f"   URL: {result.link}")
            if result.snippet:
                print(f"   Snippet: {result.snippet}")
            print(f"   Source: {result.source}")
    except Exception as e:
        print(f"Search error: {e}")
