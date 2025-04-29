"""
Base Tool Implementation
-----------------------
This file contains the base Tool class that all tools should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import json
import time
from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    """Base model for tool responses."""
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Any = Field(None, description="The result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if the tool execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the execution")


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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.get_schema(),
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"
