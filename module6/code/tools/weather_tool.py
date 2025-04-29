"""
Weather Tool Implementation
--------------------------
This file contains a tool for fetching weather information using the OpenWeatherMap API.
"""

from typing import Dict, Any, Optional, Union, List, Callable
import os
import time
import requests
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from dotenv import load_dotenv

from .base_tool import BaseTool, ToolResponse

# Check if LangChain is available
try:
    from langchain.tools import BaseTool as LangChainBaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Load environment variables
load_dotenv()

class WeatherResult(BaseModel):
    """Model for weather results."""
    temperature: float = Field(..., description="Current temperature in Celsius")
    conditions: str = Field(..., description="Weather conditions description")
    humidity: int = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in meters per second")
    location: str = Field(..., description="Location name")
    units: str = Field("metric", description="Units used for measurements (metric or imperial)")


class WeatherTool(BaseTool):
    """Tool for fetching current weather information using OpenWeatherMap API."""

    def __init__(
        self,
        name: str = "weather",
        description: str = "Get current weather information for a location",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the Weather tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            api_key: OpenWeatherMap API key (if None, will try to get from environment)
            max_retries: Maximum number of retries on rate limit or error
            retry_delay: Initial delay between retries in seconds
        """
        super().__init__(name, description)
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is required. Set it as an argument or as OPENWEATHERMAP_API_KEY environment variable.")

        self.base_url = "https://api.openweathermap.org/data/2.5"
        # Add API version to metadata
        self.metadata["api_version"] = "2.5"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        # Create session with retry strategy
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute the weather tool with the provided parameters.

        Args:
            location: Name of the location to get weather for
            lat: Latitude coordinate (alternative to location)
            lon: Longitude coordinate (alternative to location)
            units: Units for temperature (metric or imperial)

        Returns:
            ToolResponse: The result of the tool execution
        """
        try:
            # Check parameters
            location = kwargs.get("location")
            lat = kwargs.get("lat")
            lon = kwargs.get("lon")
            units = kwargs.get("units", "metric")

            # Validate parameters
            if location is None and (lat is None or lon is None):
                return ToolResponse(
                    success=False,
                    error="Either 'location' or both 'lat' and 'lon' must be provided"
                )

            # Get weather data
            if location is not None:
                result = self.get_weather_by_location(location, units)
            else:
                result = self.get_weather_by_coords(lat, lon, units)

            # Return successful response
            return ToolResponse(
                success=True,
                result=result.model_dump(),
                metadata={
                    "units": units,
                    "source": "OpenWeatherMap",
                    "query_type": "location" if location else "coordinates"
                }
            )

        except ValueError as e:
            # Handle location not found or invalid parameters
            return ToolResponse(
                success=False,
                error=str(e)
            )

        except Exception as e:
            # Handle other errors
            return ToolResponse(
                success=False,
                error=f"Weather tool execution failed: {str(e)}"
            )

    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request with retry logic and rate limit handling.

        Args:
            url: API endpoint URL
            params: Request parameters

        Returns:
            Dict[str, Any]: API response

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit error
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # exponential backoff
                        print(f"Rate limit hit. Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue

                # If we've exhausted retries or it's not a rate limit error
                error_msg = f"API error: {e.response.status_code} - {e.response.text}"
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"{error_msg}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(error_msg)

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"Request failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                raise Exception(f"Request failed after {self.max_retries} attempts: {str(e)}")

    def get_weather_by_coords(self, lat: float, lon: float, units: str = "metric") -> WeatherResult:
        """
        Get weather information by coordinates.

        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            units: Units for temperature (metric or imperial)

        Returns:
            WeatherResult: Weather information
        """
        url = f"{self.base_url}/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": units
        }

        data = self._make_request(url, params)
        return WeatherResult(
            temperature=data['main']['temp'],
            conditions=data['weather'][0]['description'],
            humidity=data['main']['humidity'],
            wind_speed=data['wind']['speed'],
            location=data['name'],
            units=units
        )

    def get_weather_by_location(self, location: str, units: str = "metric") -> WeatherResult:
        """
        Get weather information by location name.

        Args:
            location: Name of the location
            units: Units for temperature (metric or imperial)

        Returns:
            WeatherResult: Weather information

        Raises:
            ValueError: If location is not found
        """
        # First, geocode the location
        geo_url = f"{self.base_url}/geo/1.0/direct"
        geo_params = {
            "q": location,
            "limit": 1,
            "appid": self.api_key
        }

        locations = self._make_request(geo_url, geo_params)
        if not locations:
            raise ValueError(f"Location not found: {location}")

        # Get coordinates
        lat = locations[0]['lat']
        lon = locations[0]['lon']

        # Get weather using coordinates
        return self.get_weather_by_coords(lat, lon, units)

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
                        "location": {
                            "type": "string",
                            "description": "The name of the location to get weather for (e.g., 'New York', 'London')"
                        },
                        "lat": {
                            "type": "number",
                            "description": "Latitude coordinate (alternative to location)"
                        },
                        "lon": {
                            "type": "number",
                            "description": "Longitude coordinate (alternative to location)"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["metric", "imperial"],
                            "description": "Units for temperature (metric: Celsius, imperial: Fahrenheit)",
                            "default": "metric"
                        }
                    },
                    "required": [],
                    "oneOf": [
                        {"required": ["location"]},
                        {"required": ["lat", "lon"]}
                    ]
                }
            }
        }

    def get_current_weather(self, location_or_coords: Union[str, Dict[str, float]], units: str = "metric") -> Dict[str, Any]:
        """
        Convenience method to get current weather for a location or coordinates.

        Args:
            location_or_coords: Either a location name (str) or coordinates dict with 'lat' and 'lon' keys
            units: Units for temperature (metric or imperial)

        Returns:
            Dict[str, Any]: Weather information
        """
        try:
            if isinstance(location_or_coords, str):
                result = self.get_weather_by_location(location_or_coords, units)
            else:
                result = self.get_weather_by_coords(
                    location_or_coords.get('lat'),
                    location_or_coords.get('lon'),
                    units
                )
            return result.model_dump()
        except Exception as e:
            raise ValueError(f"Failed to get weather: {str(e)}")

    def to_langchain_tool(self, name: Optional[str] = None, description: Optional[str] = None) -> Any:
        """
        Convert this tool to a LangChain tool.

        Args:
            name: Optional name for the LangChain tool
            description: Optional description for the LangChain tool

        Returns:
            A LangChain tool

        Raises:
            ImportError: If LangChain is not installed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install it with 'pip install langchain'"
            )

        # For simplicity, we'll use the built-in OpenWeatherMap tool from LangChain
        try:
            from langchain_community.agent_toolkits.load_tools import load_tools

            # Load the built-in OpenWeatherMap tool
            tools = load_tools(["openweathermap-api"])
            if tools:
                print("✅ Using LangChain's built-in OpenWeatherMap tool")
                return tools[0]
            else:
                raise ImportError("Failed to load OpenWeatherMap tool from LangChain")
        except ImportError as e:
            raise ImportError(f"Error loading LangChain tools: {str(e)}. Make sure you have installed 'pyowm' with 'pip install pyowm'.")


# Example usage
if __name__ == "__main__":
    # Create the tool
    weather_tool = WeatherTool()

    # Test with location name
    try:
        response = weather_tool.execute(location="London")
        print("Weather in London:")
        print(response.result)
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")

    # Test with coordinates
    try:
        response = weather_tool.execute(lat=40.7128, lon=-74.0060)
        print("Weather at coordinates (New York):")
        print(response.result)
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
