"""
Unit tests for the Tool Registry.
"""

import unittest
import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to the path so we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

# Import the tools and registry
from module6.code.registry import ToolRegistry, ToolCategory
from module6.code.tools.base_tool import BaseTool, ToolResponse


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, name="mock_tool", description="A mock tool for testing"):
        super().__init__(name, description)
    
    def execute(self, **kwargs) -> ToolResponse:
        """Execute the mock tool."""
        return ToolResponse(success=True, result="Mock result")
    
    def get_schema(self):
        """Get the schema for the mock tool."""
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "A test parameter"}
            }
        }
    
    def to_langchain_tool(self):
        """Convert to a LangChain tool."""
        return MagicMock()


class TestToolRegistry(unittest.TestCase):
    """Test cases for the Tool Registry."""
    
    def setUp(self):
        """Set up the test environment."""
        self.registry = ToolRegistry()
        self.tool1 = MockTool("tool1", "Test tool 1")
        self.tool2 = MockTool("tool2", "Test tool 2")
        self.tool3 = MockTool("tool3", "Test tool 3")
    
    def test_register_tool(self):
        """Test registering a tool."""
        self.registry.register_tool(self.tool1)
        self.assertIn("tool1", self.registry.list_tools())
        self.assertEqual(self.registry.get_tool("tool1"), self.tool1)
    
    def test_register_duplicate_tool(self):
        """Test registering a duplicate tool."""
        self.registry.register_tool(self.tool1)
        with self.assertRaises(ValueError):
            self.registry.register_tool(self.tool1)
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        self.registry.register_tool(self.tool1)
        self.assertTrue(self.registry.unregister_tool("tool1"))
        self.assertNotIn("tool1", self.registry.list_tools())
    
    def test_unregister_nonexistent_tool(self):
        """Test unregistering a nonexistent tool."""
        self.assertFalse(self.registry.unregister_tool("nonexistent"))
    
    def test_get_tool(self):
        """Test getting a tool."""
        self.registry.register_tool(self.tool1)
        self.assertEqual(self.registry.get_tool("tool1"), self.tool1)
        self.assertIsNone(self.registry.get_tool("nonexistent"))
    
    def test_list_tools(self):
        """Test listing tools."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        tools = self.registry.list_tools()
        self.assertIn("tool1", tools)
        self.assertIn("tool2", tools)
        self.assertEqual(len(tools), 2)
    
    def test_get_all_tools(self):
        """Test getting all tools."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        tools = self.registry.get_all_tools()
        self.assertIn(self.tool1, tools)
        self.assertIn(self.tool2, tools)
        self.assertEqual(len(tools), 2)
    
    def test_add_tool_to_category(self):
        """Test adding a tool to a category."""
        self.registry.register_tool(self.tool1)
        self.registry.add_tool_to_category("tool1", ToolCategory.UTILITY)
        self.registry.add_tool_to_category("tool1", "custom_category")
        
        utility_tools = self.registry.get_tools_by_category(ToolCategory.UTILITY)
        custom_tools = self.registry.get_tools_by_category("custom_category")
        
        self.assertIn(self.tool1, utility_tools)
        self.assertIn(self.tool1, custom_tools)
    
    def test_add_nonexistent_tool_to_category(self):
        """Test adding a nonexistent tool to a category."""
        with self.assertRaises(ValueError):
            self.registry.add_tool_to_category("nonexistent", ToolCategory.UTILITY)
    
    def test_remove_tool_from_category(self):
        """Test removing a tool from a category."""
        self.registry.register_tool(self.tool1)
        self.registry.add_tool_to_category("tool1", ToolCategory.UTILITY)
        self.registry.remove_tool_from_category("tool1", ToolCategory.UTILITY)
        
        utility_tools = self.registry.get_tools_by_category(ToolCategory.UTILITY)
        self.assertNotIn(self.tool1, utility_tools)
    
    def test_get_tools_by_category(self):
        """Test getting tools by category."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        self.registry.add_tool_to_category("tool1", ToolCategory.UTILITY)
        self.registry.add_tool_to_category("tool2", ToolCategory.UTILITY)
        
        utility_tools = self.registry.get_tools_by_category(ToolCategory.UTILITY)
        self.assertEqual(len(utility_tools), 2)
        self.assertIn(self.tool1, utility_tools)
        self.assertIn(self.tool2, utility_tools)
    
    def test_get_categories(self):
        """Test getting categories."""
        self.registry.register_tool(self.tool1)
        self.registry.add_tool_to_category("tool1", ToolCategory.UTILITY)
        self.registry.add_tool_to_category("tool1", "custom_category")
        
        categories = self.registry.get_categories()
        self.assertIn("utility", categories)
        self.assertIn("custom_category", categories)
    
    def test_get_tool_categories(self):
        """Test getting tool categories."""
        self.registry.register_tool(self.tool1)
        self.registry.add_tool_to_category("tool1", ToolCategory.UTILITY)
        self.registry.add_tool_to_category("tool1", "custom_category")
        
        categories = self.registry.get_tool_categories("tool1")
        self.assertIn("utility", categories)
        self.assertIn("custom_category", categories)
    
    def test_get_all_schemas(self):
        """Test getting all schemas."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        
        schemas = self.registry.get_all_schemas()
        self.assertEqual(len(schemas), 2)
        
        # Check that each schema has the expected structure
        for schema in schemas:
            self.assertIn("type", schema)
            self.assertIn("properties", schema)
    
    @patch("module6.code.registry.tool_registry.LANGCHAIN_AVAILABLE", True)
    def test_to_langchain_tools(self):
        """Test converting to LangChain tools."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        
        with patch.object(self.tool1, "to_langchain_tool") as mock_to_langchain1:
            with patch.object(self.tool2, "to_langchain_tool") as mock_to_langchain2:
                mock_to_langchain1.return_value = MagicMock()
                mock_to_langchain2.return_value = MagicMock()
                
                langchain_tools = self.registry.to_langchain_tools()
                self.assertEqual(len(langchain_tools), 2)
                
                mock_to_langchain1.assert_called_once()
                mock_to_langchain2.assert_called_once()
    
    @patch("module6.code.registry.tool_registry.LANGCHAIN_AVAILABLE", False)
    def test_to_langchain_tools_not_available(self):
        """Test converting to LangChain tools when LangChain is not available."""
        self.registry.register_tool(self.tool1)
        
        with self.assertRaises(ImportError):
            self.registry.to_langchain_tools()
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        self.registry.register_tool(self.tool1)
        self.registry.add_tool_to_category("tool1", ToolCategory.UTILITY)
        
        registry_dict = self.registry.to_dict()
        self.assertIn("tools", registry_dict)
        self.assertIn("categories", registry_dict)
        self.assertIn("metadata", registry_dict)
        
        self.assertIn("tool1", registry_dict["tools"])
        self.assertIn("utility", registry_dict["categories"])
        self.assertEqual(registry_dict["metadata"]["tool_count"], 1)
    
    def test_to_json(self):
        """Test converting to JSON."""
        self.registry.register_tool(self.tool1)
        
        registry_json = self.registry.to_json()
        self.assertIsInstance(registry_json, str)
        self.assertIn("tool1", registry_json)
    
    def test_str(self):
        """Test string representation."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        
        registry_str = str(self.registry)
        self.assertIn("ToolRegistry", registry_str)
        self.assertIn("2 tools", registry_str)
        self.assertIn("tool1", registry_str)
        self.assertIn("tool2", registry_str)
    
    def test_len(self):
        """Test length."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        
        self.assertEqual(len(self.registry), 2)


if __name__ == "__main__":
    unittest.main()
