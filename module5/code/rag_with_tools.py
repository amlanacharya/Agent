"""
RAG with Tools Integration Example

This module demonstrates how to integrate tools with a RAG system,
showing the evolution from pure information retrieval to action execution.
"""

import json
import math
import datetime
from typing import Dict, List, Any, Optional, Union, Callable

# Simulated components for demonstration purposes
class RAGSystem:
    """Simple RAG system for demonstration purposes."""
    
    def __init__(self, documents: List[str]):
        """Initialize with a list of documents."""
        self.documents = documents
        print("RAG System initialized with", len(documents), "documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Simulate document retrieval based on query."""
        # In a real system, this would use embeddings and vector search
        results = []
        for doc in self.documents:
            # Simple keyword matching for demonstration
            if any(term in doc.lower() for term in query.lower().split()):
                results.append(doc)
                if len(results) >= top_k:
                    break
        
        print(f"Retrieved {len(results)} documents for query: '{query}'")
        return results
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response based on the query and retrieved context."""
        # In a real system, this would use an LLM
        if not context:
            return "I don't have enough information to answer that question."
        
        # Simple response generation for demonstration
        response = f"Based on the information I have:\n\n"
        for i, doc in enumerate(context, 1):
            response += f"{i}. {doc}\n"
        
        print(f"Generated response for query: '{query}'")
        return response


class Tool:
    """Base class for tools that can be used by an agent."""
    
    def __init__(self, name: str, description: str):
        """Initialize the tool with a name and description."""
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the provided parameters."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON Schema for this tool."""
        raise NotImplementedError("Subclasses must implement get_schema method")


class CalculatorTool(Tool):
    """A tool for performing mathematical calculations."""
    
    def __init__(self):
        """Initialize the calculator tool."""
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
    
    def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """
        Execute a mathematical operation.
        
        Args:
            operation: The operation to perform (add, subtract, multiply, divide, power)
            a: The first number
            b: The second number
            
        Returns:
            A dictionary with the result and a description
        """
        result = None
        if operation == "add":
            result = a + b
            description = f"The sum of {a} and {b} is {result}"
        elif operation == "subtract":
            result = a - b
            description = f"The difference between {a} and {b} is {result}"
        elif operation == "multiply":
            result = a * b
            description = f"The product of {a} and {b} is {result}"
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
            description = f"The result of dividing {a} by {b} is {result}"
        elif operation == "power":
            result = math.pow(a, b)
            description = f"{a} raised to the power of {b} is {result}"
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        print(f"Calculator executed: {operation}({a}, {b}) = {result}")
        return {"result": result, "description": description}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON Schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "power"],
                        "description": "The mathematical operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }


class DateTimeTool(Tool):
    """A tool for getting current date and time information."""
    
    def __init__(self):
        """Initialize the date/time tool."""
        super().__init__(
            name="datetime",
            description="Get current date and time information"
        )
    
    def execute(self, format_type: str = "full") -> Dict[str, Any]:
        """
        Get the current date and time.
        
        Args:
            format_type: The format to return (full, date, time, year, month, day)
            
        Returns:
            A dictionary with the requested date/time information
        """
        now = datetime.datetime.now()
        
        if format_type == "full":
            result = now.strftime("%Y-%m-%d %H:%M:%S")
            description = f"The current date and time is {result}"
        elif format_type == "date":
            result = now.strftime("%Y-%m-%d")
            description = f"The current date is {result}"
        elif format_type == "time":
            result = now.strftime("%H:%M:%S")
            description = f"The current time is {result}"
        elif format_type == "year":
            result = now.year
            description = f"The current year is {result}"
        elif format_type == "month":
            result = now.month
            description = f"The current month is {result}"
        elif format_type == "day":
            result = now.day
            description = f"The current day is {result}"
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        print(f"DateTime executed: format_type={format_type}, result={result}")
        return {"result": result, "description": description}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON Schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "format_type": {
                        "type": "string",
                        "enum": ["full", "date", "time", "year", "month", "day"],
                        "description": "The format to return",
                        "default": "full"
                    }
                },
                "required": []
            }
        }


class ToolRegistry:
    """A registry of tools that can be used by an agent."""
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the registry."""
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self.tools.values()]


class HybridAgent:
    """An agent that combines RAG with tools."""
    
    def __init__(self, rag_system: RAGSystem, tool_registry: ToolRegistry):
        """
        Initialize the hybrid agent.
        
        Args:
            rag_system: The RAG system for information retrieval
            tool_registry: The tool registry for action execution
        """
        self.rag_system = rag_system
        self.tool_registry = tool_registry
        print("Hybrid Agent initialized with RAG system and Tool Registry")
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine if it requires information or action.
        
        In a real system, this would use an LLM to classify the query.
        Here we use simple keyword matching for demonstration.
        
        Args:
            query: The user query
            
        Returns:
            A dictionary with the query type and other analysis results
        """
        # Simple keyword-based classification for demonstration
        action_keywords = ["calculate", "compute", "what is", "find", "get", "current", "time", "date"]
        information_keywords = ["explain", "describe", "what are", "tell me about", "information on"]
        
        is_action = any(keyword in query.lower() for keyword in action_keywords)
        is_information = any(keyword in query.lower() for keyword in information_keywords)
        
        if is_action and not is_information:
            query_type = "action"
        elif is_information and not is_action:
            query_type = "information"
        elif is_action and is_information:
            query_type = "hybrid"
        else:
            query_type = "unknown"
        
        print(f"Query '{query}' classified as: {query_type}")
        return {"type": query_type, "query": query}
    
    def select_tool(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Select an appropriate tool for a query.
        
        In a real system, this would use an LLM to select the tool and parameters.
        Here we use simple keyword matching for demonstration.
        
        Args:
            query: The user query
            
        Returns:
            A dictionary with the selected tool and parameters, or None if no tool is selected
        """
        query_lower = query.lower()
        
        # Simple tool selection logic for demonstration
        if any(word in query_lower for word in ["calculate", "compute", "sum", "add", "subtract", "multiply", "divide"]):
            tool_name = "calculator"
            
            # Simple parameter extraction for demonstration
            import re
            numbers = re.findall(r'\d+', query)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
            else:
                a, b = 0, 0
            
            if "add" in query_lower or "sum" in query_lower or "plus" in query_lower:
                operation = "add"
            elif "subtract" in query_lower or "minus" in query_lower:
                operation = "subtract"
            elif "multiply" in query_lower or "times" in query_lower:
                operation = "multiply"
            elif "divide" in query_lower or "divided by" in query_lower:
                operation = "divide"
            elif "power" in query_lower or "raised to" in query_lower:
                operation = "power"
            else:
                operation = "add"  # Default
            
            return {
                "tool_name": tool_name,
                "parameters": {
                    "operation": operation,
                    "a": a,
                    "b": b
                }
            }
        
        elif any(word in query_lower for word in ["time", "date", "day", "month", "year", "current"]):
            tool_name = "datetime"
            
            # Simple parameter extraction for demonstration
            if "date" in query_lower and "time" not in query_lower:
                format_type = "date"
            elif "time" in query_lower and "date" not in query_lower:
                format_type = "time"
            elif "year" in query_lower:
                format_type = "year"
            elif "month" in query_lower:
                format_type = "month"
            elif "day" in query_lower:
                format_type = "day"
            else:
                format_type = "full"  # Default
            
            return {
                "tool_name": tool_name,
                "parameters": {
                    "format_type": format_type
                }
            }
        
        return None
    
    def process_query(self, query: str) -> str:
        """
        Process a user query using RAG, tools, or both.
        
        Args:
            query: The user query
            
        Returns:
            The response to the query
        """
        # Analyze the query to determine its type
        analysis = self.analyze_query(query)
        query_type = analysis["type"]
        
        if query_type == "information":
            # Use RAG for information queries
            documents = self.rag_system.retrieve(query)
            response = self.rag_system.generate_response(query, documents)
            return response
        
        elif query_type == "action":
            # Use tools for action queries
            tool_selection = self.select_tool(query)
            if tool_selection:
                tool_name = tool_selection["tool_name"]
                parameters = tool_selection["parameters"]
                
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    try:
                        result = tool.execute(**parameters)
                        return result["description"]
                    except Exception as e:
                        return f"Error executing tool: {str(e)}"
                else:
                    return f"Tool '{tool_name}' not found"
            else:
                # Fall back to RAG if no tool is selected
                documents = self.rag_system.retrieve(query)
                response = self.rag_system.generate_response(query, documents)
                return response
        
        elif query_type == "hybrid":
            # Use both RAG and tools for hybrid queries
            tool_selection = self.select_tool(query)
            documents = self.rag_system.retrieve(query)
            
            if tool_selection and documents:
                tool_name = tool_selection["tool_name"]
                parameters = tool_selection["parameters"]
                
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    try:
                        tool_result = tool.execute(**parameters)
                        rag_response = self.rag_system.generate_response(query, documents)
                        
                        # Combine the results
                        response = f"{tool_result['description']}\n\nAdditional information:\n{rag_response}"
                        return response
                    except Exception as e:
                        # Fall back to RAG if tool execution fails
                        rag_response = self.rag_system.generate_response(query, documents)
                        return f"Tool execution failed: {str(e)}\n\nHere's what I found instead:\n{rag_response}"
                else:
                    # Fall back to RAG if tool is not found
                    return self.rag_system.generate_response(query, documents)
            elif tool_selection:
                # Use tool only if no relevant documents
                tool_name = tool_selection["tool_name"]
                parameters = tool_selection["parameters"]
                
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    try:
                        result = tool.execute(**parameters)
                        return result["description"]
                    except Exception as e:
                        return f"Error executing tool: {str(e)}"
                else:
                    return f"Tool '{tool_name}' not found and no relevant information available"
            elif documents:
                # Use RAG only if no relevant tool
                return self.rag_system.generate_response(query, documents)
            else:
                return "I don't have enough information or capabilities to handle this query"
        
        else:  # Unknown query type
            # Try both approaches and see if either works
            tool_selection = self.select_tool(query)
            documents = self.rag_system.retrieve(query)
            
            if tool_selection:
                tool_name = tool_selection["tool_name"]
                parameters = tool_selection["parameters"]
                
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    try:
                        result = tool.execute(**parameters)
                        return result["description"]
                    except Exception:
                        # Silently fail and try RAG instead
                        pass
            
            if documents:
                return self.rag_system.generate_response(query, documents)
            
            return "I'm not sure how to help with that query"


def main():
    """Demonstrate the hybrid agent with RAG and tools."""
    # Sample documents for the RAG system
    documents = [
        "The calculator is a tool used for performing arithmetic operations.",
        "Basic calculators can add, subtract, multiply, and divide.",
        "Scientific calculators can perform more complex operations like exponents and logarithms.",
        "The first mechanical calculators were created in the 17th century.",
        "Modern electronic calculators became common in the 1970s.",
        "Time is measured using various units including seconds, minutes, hours, days, weeks, months, and years.",
        "A calendar is a system of organizing days for social, religious, commercial, or administrative purposes.",
        "The Gregorian calendar is the most widely used civil calendar in the world.",
        "Coordinated Universal Time (UTC) is the primary time standard by which the world regulates clocks and time.",
        "A digital clock displays the time digitally, as opposed to an analog clock."
    ]
    
    # Create the RAG system
    rag_system = RAGSystem(documents)
    
    # Create the tool registry
    tool_registry = ToolRegistry()
    
    # Register tools
    calculator_tool = CalculatorTool()
    datetime_tool = DateTimeTool()
    tool_registry.register_tool(calculator_tool)
    tool_registry.register_tool(datetime_tool)
    
    # Create the hybrid agent
    agent = HybridAgent(rag_system, tool_registry)
    
    # Example queries
    queries = [
        "Tell me about calculators",  # Information query
        "Calculate 5 plus 3",  # Action query
        "What is the current time?",  # Action query
        "Explain how time is measured and tell me the current date",  # Hybrid query
        "What is 10 divided by 2?",  # Action query
        "Tell me about digital clocks and show me the current time",  # Hybrid query
    ]
    
    # Process each query
    print("\n" + "="*80)
    print("HYBRID AGENT DEMONSTRATION")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: \"{query}\"")
        print("-" * 40)
        response = agent.process_query(query)
        print(f"Response: {response}")
        print("-" * 40)


if __name__ == "__main__":
    main()
