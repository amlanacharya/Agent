"""
Tools Package for Module 6
--------------------------
This package contains various tools for integration with language models.
"""

from .base_tool import BaseTool, ToolResponse
from .openai_tool import OpenAITool
from .groq_tool import GroqTool
from .search_tool import SearchTool, SearchResult
from .weather_tool import WeatherTool, WeatherResult
from .alpha_vantage_tool import AlphaVantageTool, StockQuote, ExchangeRate

__all__ = [
    'BaseTool',
    'ToolResponse',
    'OpenAITool',
    'GroqTool',
    'SearchTool',
    'SearchResult',
    'WeatherTool',
    'WeatherResult',
    'AlphaVantageTool',
    'StockQuote',
    'ExchangeRate'
]
