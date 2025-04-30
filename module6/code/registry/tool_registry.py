"""
Tool Registry Implementation
--------------------------
This file contains the implementation of the Tool Registry system for managing tools.
"""

from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum, auto
import time
import json

# Import the BaseTool class
from module6.code.tools.base_tool import BaseTool

# Try to import LangChain components
try:
    from langchain.tools import Tool as LangChainTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Install with 'pip install langchain'")


class ToolCategory(Enum):
    """Enum for tool categories."""
    CONTENT_GENERATION = auto()
    INFORMATION_RETRIEVAL = auto()
    DATA_ANALYSIS = auto()
    EXTERNAL_API = auto()
    UTILITY = auto()
    OTHER = auto()


class ToolRegistry:
    """
    A registry for managing and discovering tools.
    
    The ToolRegistry provides a central place to register, discover, and manage tools.
    It supports categorization, filtering, and conversion to LangChain tools.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        # Dictionary mapping tool names to tool instances
        self.tools: Dict[str, BaseTool] = {}
        
        # Dictionary mapping category names to sets of tool names
        self.categories: Dict[Union[str, ToolCategory], Set[str]] = {}
        
        # Dictionary mapping tool names to their metadata
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default categories
        for category in ToolCategory:
            self.categories[category] = set()
            self.categories[category.name.lower()] = set()
    
    def register_tool(self, tool: BaseTool, categories: Optional[List[Union[str, ToolCategory]]] = None) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: The tool to register
            categories: Optional list of categories to assign to the tool
        
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self.tools:
            raise ValueError(f"A tool with the name '{tool.name}' is already registered")
        
        # Store the tool
        self.tools[tool.name] = tool
        
        # Store tool metadata
        self.tool_metadata[tool.name] = {
            "registered_at": time.time(),
            "description": tool.description,
            **tool.metadata
        }
        
        # Add to categories
        if categories:
            for category in categories:
                self.add_tool_to_category(tool.name, category)
        
        print(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: The name of the tool to unregister
            
        Returns:
            bool: True if the tool was unregistered, False if it wasn't found
        """
        if tool_name not in self.tools:
            return False
        
        # Remove from tools dictionary
        del self.tools[tool_name]
        
        # Remove from metadata
        if tool_name in self.tool_metadata:
            del self.tool_metadata[tool_name]
        
        # Remove from all categories
        for category in self.categories:
            if tool_name in self.categories[category]:
                self.categories[category].remove(tool_name)
        
        print(f"Unregistered tool: {tool_name}")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: The name of the tool to get
            
        Returns:
            Optional[BaseTool]: The tool if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List[str]: List of registered tool names
        """
        return list(self.tools.keys())
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            List[BaseTool]: List of all registered tools
        """
        return list(self.tools.values())
    
    def add_tool_to_category(self, tool_name: str, category: Union[str, ToolCategory]) -> bool:
        """
        Add a tool to a category.
        
        Args:
            tool_name: The name of the tool to add
            category: The category to add the tool to
            
        Returns:
            bool: True if the tool was added to the category, False otherwise
            
        Raises:
            ValueError: If the tool is not registered
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        
        # Convert ToolCategory enum to string if needed
        if isinstance(category, ToolCategory):
            category_name = category.name.lower()
            enum_category = category
        else:
            category_name = category.lower()
            try:
                enum_category = ToolCategory[category.upper()]
            except KeyError:
                enum_category = None
        
        # Create category if it doesn't exist
        if category_name not in self.categories:
            self.categories[category_name] = set()
        
        # Add tool to string category
        self.categories[category_name].add(tool_name)
        
        # Add tool to enum category if it exists
        if enum_category and enum_category not in self.categories:
            self.categories[enum_category] = set()
        
        if enum_category:
            self.categories[enum_category].add(tool_name)
        
        return True
    
    def remove_tool_from_category(self, tool_name: str, category: Union[str, ToolCategory]) -> bool:
        """
        Remove a tool from a category.
        
        Args:
            tool_name: The name of the tool to remove
            category: The category to remove the tool from
            
        Returns:
            bool: True if the tool was removed from the category, False otherwise
        """
        # Convert ToolCategory enum to string if needed
        if isinstance(category, ToolCategory):
            category_name = category.name.lower()
            enum_category = category
        else:
            category_name = category.lower()
            try:
                enum_category = ToolCategory[category.upper()]
            except KeyError:
                enum_category = None
        
        # Check if category exists
        if category_name not in self.categories:
            return False
        
        # Remove tool from string category
        if tool_name in self.categories[category_name]:
            self.categories[category_name].remove(tool_name)
            
        # Remove tool from enum category if it exists
        if enum_category and enum_category in self.categories and tool_name in self.categories[enum_category]:
            self.categories[enum_category].remove(tool_name)
            
        return True
    
    def get_tools_by_category(self, category: Union[str, ToolCategory]) -> List[BaseTool]:
        """
        Get all tools in a category.
        
        Args:
            category: The category to get tools for
            
        Returns:
            List[BaseTool]: List of tools in the category
        """
        # Convert ToolCategory enum to string if needed
        if isinstance(category, ToolCategory):
            category_name = category.name.lower()
        else:
            category_name = category.lower()
        
        # Check if category exists
        if category_name not in self.categories:
            return []
        
        # Get tools in category
        tool_names = self.categories[category_name]
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_categories(self) -> List[str]:
        """
        Get all category names.
        
        Returns:
            List[str]: List of category names
        """
        # Filter out enum categories and empty categories
        return [cat for cat in self.categories.keys() 
                if isinstance(cat, str) and self.categories[cat]]
    
    def get_tool_categories(self, tool_name: str) -> List[str]:
        """
        Get all categories a tool belongs to.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            List[str]: List of category names the tool belongs to
        """
        if tool_name not in self.tools:
            return []
        
        # Find all string categories containing the tool
        return [cat for cat in self.categories.keys() 
                if isinstance(cat, str) and tool_name in self.categories[cat]]
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        Get schemas for all registered tools.
        
        Returns:
            List[Dict[str, Any]]: List of tool schemas
        """
        return [tool.get_schema() for tool in self.tools.values()]
    
    def to_langchain_tools(self) -> List[Any]:
        """
        Convert all registered tools to LangChain tools.
        
        Returns:
            List[Any]: List of LangChain tools
            
        Raises:
            ImportError: If LangChain is not available
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not available. Install with 'pip install langchain'")
        
        langchain_tools = []
        
        for tool in self.tools.values():
            # Check if the tool has a to_langchain_tool method
            if hasattr(tool, 'to_langchain_tool') and callable(getattr(tool, 'to_langchain_tool')):
                try:
                    langchain_tool = tool.to_langchain_tool()
                    langchain_tools.append(langchain_tool)
                except Exception as e:
                    print(f"Error converting tool '{tool.name}' to LangChain tool: {e}")
            else:
                print(f"Tool '{tool.name}' does not have a to_langchain_tool method")
        
        return langchain_tools
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the registry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the registry
        """
        # Convert tools to dictionaries
        tools_dict = {name: tool.to_dict() for name, tool in self.tools.items()}
        
        # Convert categories to dictionaries (only string categories)
        categories_dict = {cat: list(tools) for cat, tools in self.categories.items() 
                          if isinstance(cat, str) and tools}
        
        return {
            "tools": tools_dict,
            "categories": categories_dict,
            "metadata": {
                "tool_count": len(self.tools),
                "category_count": len(self.get_categories())
            }
        }
    
    def to_json(self) -> str:
        """
        Convert the registry to a JSON string.
        
        Returns:
            str: JSON representation of the registry
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """String representation of the registry."""
        tools_str = ", ".join(self.list_tools())
        return f"ToolRegistry with {len(self.tools)} tools: {tools_str}"
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self.tools)
