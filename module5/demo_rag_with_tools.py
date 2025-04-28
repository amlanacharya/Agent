"""
Demo script for RAG with Tools integration.

This script demonstrates how a hybrid agent can combine RAG capabilities
with tool use to handle both information and action requests.
"""

import os
import sys

# Add the module directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the RAG with Tools components
from code.rag_with_tools import (
    RAGSystem,
    Tool,
    CalculatorTool,
    DateTimeTool,
    ToolRegistry,
    HybridAgent
)


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"{text.center(80)}")
    print("="*80)


def print_section(text):
    """Print a formatted section header."""
    print("\n" + "-"*80)
    print(f"{text}")
    print("-"*80)


def main():
    """Run the RAG with Tools demonstration."""
    print_header("RAG WITH TOOLS DEMONSTRATION")
    
    # Step 1: Create a RAG system with sample documents
    print_section("Step 1: Creating RAG System")
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
    
    rag_system = RAGSystem(documents)
    print("RAG System created with 10 sample documents")
    
    # Step 2: Create and register tools
    print_section("Step 2: Creating and Registering Tools")
    tool_registry = ToolRegistry()
    
    calculator_tool = CalculatorTool()
    datetime_tool = DateTimeTool()
    
    tool_registry.register_tool(calculator_tool)
    tool_registry.register_tool(datetime_tool)
    
    print(f"Available tools: {', '.join(tool_registry.list_tools())}")
    
    # Step 3: Create the hybrid agent
    print_section("Step 3: Creating Hybrid Agent")
    agent = HybridAgent(rag_system, tool_registry)
    print("Hybrid Agent created, combining RAG and Tools")
    
    # Step 4: Process information queries (RAG only)
    print_section("Step 4: Processing Information Queries (RAG only)")
    info_queries = [
        "Tell me about calculators",
        "Explain how time is measured",
        "What is a digital clock?"
    ]
    
    for query in info_queries:
        print(f"\nQuery: \"{query}\"")
        response = agent.process_query(query)
        print(f"Response: {response}")
    
    # Step 5: Process action queries (Tools only)
    print_section("Step 5: Processing Action Queries (Tools only)")
    action_queries = [
        "Calculate 5 plus 3",
        "What is 10 divided by 2?",
        "What is the current time?",
        "What is today's date?"
    ]
    
    for query in action_queries:
        print(f"\nQuery: \"{query}\"")
        response = agent.process_query(query)
        print(f"Response: {response}")
    
    # Step 6: Process hybrid queries (RAG + Tools)
    print_section("Step 6: Processing Hybrid Queries (RAG + Tools)")
    hybrid_queries = [
        "Explain how time is measured and tell me the current date",
        "Tell me about calculators and calculate 7 times 8",
        "What is a digital clock and what time is it now?"
    ]
    
    for query in hybrid_queries:
        print(f"\nQuery: \"{query}\"")
        response = agent.process_query(query)
        print(f"Response: {response}")
    
    # Step 7: Interactive mode
    print_section("Step 7: Interactive Mode")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        response = agent.process_query(query)
        print(f"Response: {response}")
    
    print_header("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    main()
