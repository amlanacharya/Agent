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
from .alpha_vantage_tool import AlphaVantageTool

# Import the registry
try:
    from ..registry import ToolRegistry, ToolCategory
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("Tool Registry not available. Make sure the registry package is installed.")

__all__ = [
    'BaseTool',
    'ToolResponse',
    'OpenAITool',
    'GroqTool',
    'SearchTool',
    'SearchResult',
    'WeatherTool',
    'WeatherResult',
    'AlphaVantageTool'
]

# Add registry to __all__ if available
if REGISTRY_AVAILABLE:
    __all__.extend(['ToolRegistry', 'ToolCategory'])
