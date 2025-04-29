"""
LangChain Weather Tool Integration
---------------------------------
This file provides integration between our WeatherTool and LangChain.
"""

from typing import Dict, Any, Optional, List, Type
import os
from dotenv import load_dotenv
from langchain.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field

from .weather_tool import WeatherTool

# Load environment variables
load_dotenv()

class WeatherInput(BaseModel):
    """Input for the weather tool."""
    location: str = Field(..., description="The city and country, e.g. 'London,GB'")

class LangChainWeatherTool(LangChainBaseTool):
    """LangChain tool for getting weather information."""

    name: str = "OpenWeatherMap"
    description: str = "Useful for getting weather information for a specific location. Input should be a string representing the city and country code, e.g., 'London,GB'."
    args_schema: Type[BaseModel] = WeatherInput

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the tool with an API key."""
        super().__init__()
        self.weather_tool = WeatherTool(api_key=api_key)

    def _run(self, location: str) -> str:
        """Run the tool with the provided location."""
        try:
            # Parse location (might include country code)
            location_parts = location.split(',')
            city = location_parts[0].strip()

            # Get weather data
            response = self.weather_tool.execute(location=city)

            if not response.success:
                return f"Error getting weather: {response.error}"

            weather_data = response.result

            # Format the response in a human-readable way
            formatted_response = f"In {location}, the current weather is as follows:\n"
            formatted_response += f"Detailed status: {weather_data['conditions']}\n"
            formatted_response += f"Wind speed: {weather_data['wind_speed']} m/s\n"
            formatted_response += f"Humidity: {weather_data['humidity']}%\n"
            formatted_response += f"Temperature: {weather_data['temperature']}°C\n"

            return formatted_response

        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, location: str) -> str:
        """Run the tool asynchronously with the provided location."""
        # For simplicity, we're just calling the synchronous version
        return self._run(location)


def get_langchain_weather_tool(api_key: Optional[str] = None) -> LangChainWeatherTool:
    """
    Get a LangChain-compatible weather tool.

    Args:
        api_key: OpenWeatherMap API key (if None, will try to get from environment)

    Returns:
        LangChainWeatherTool: A LangChain-compatible weather tool
    """
    return LangChainWeatherTool(api_key=api_key)
