"""
Module 6: Tool Integration & Function Calling
-------------------------------------------
This module focuses on building and integrating tools with language models
to create powerful agentic systems.
"""

# Import tools
try:
    from .code.tools import (
        BaseTool, ToolResponse,
        OpenAITool, GroqTool,
        SearchTool, SearchResult,
        WeatherTool, WeatherResult,
        AlphaVantageTool
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# Import registry
try:
    from .code.registry import ToolRegistry, ToolCategory
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

# Define __all__ based on what's available
__all__ = []

if TOOLS_AVAILABLE:
    __all__.extend([
        'BaseTool', 'ToolResponse',
        'OpenAITool', 'GroqTool',
        'SearchTool', 'SearchResult',
        'WeatherTool', 'WeatherResult',
        'AlphaVantageTool'
    ])

if REGISTRY_AVAILABLE:
    __all__.extend(['ToolRegistry', 'ToolCategory'])
