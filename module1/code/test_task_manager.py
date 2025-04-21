"""
Test Script for Task Manager Agent
-------------------------------
This script demonstrates how to test the TaskManagerAgent implementation.
"""

import os
import time
import shutil

# Adjust the import path based on how you're running the script
try:
    # When running from the module1/code directory
    from task_manager_agent import TaskManagerAgent
except ImportError:
    # When running from the project root
    from module1.code.task_manager_agent import TaskManagerAgent

def test_task_manager():
    """Test the TaskManagerAgent with various inputs"""
    # Create a clean test directory
    test_dir = "test_task_manager_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize the agent
    agent = TaskManagerAgent(storage_dir=test_dir)
    
    print("=== Task Manager Agent Test ===\n")
    
    # Test task creation
    print("Testing task creation...")
    inputs = [
        "Add a new task: Finish the project proposal by Friday",
        "I need to call John tomorrow, high priority",
        "Remind me to buy groceries next week"
    ]
    
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
    
    # Test task queries
    print("\nTesting task queries...")
    query = "Show me all my tasks"
    print(f"\nUser: {query}")
    response = agent.agent_loop(query)
    print(f"Agent: {response}")
    
    # Test task updates
    print("\nTesting task updates...")
    update = "Mark the project proposal as completed"
    print(f"\nUser: {update}")
    response = agent.agent_loop(update)
    print(f"Agent: {response}")
    
    # Test preference setting
    print("\nTesting preference setting...")
    preference = "I prefer to sort tasks by priority"
    print(f"\nUser: {preference}")
    response = agent.agent_loop(preference)
    print(f"Agent: {response}")
    
    # Test daily summary
    print("\nTesting daily summary...")
    summary = agent.get_daily_summary()
    print(f"Daily Summary:\n{summary}")
    
    # Test task statistics
    print("\nTask Statistics:")
    stats = agent.get_tasks_summary()
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"By status: {stats['by_status']}")
    print(f"By priority: {stats['by_priority']}")
    
    # Test reasoning steps
    print("\nReasoning Steps for Last Interaction:")
    steps = agent.get_reasoning_steps()
    for step in steps:
        print(f"- {step['step']}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)

def test_conversation_flow():
    """Test a more natural conversation flow with the agent"""
    # Create a clean test directory
    test_dir = "test_conversation_flow"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize the agent
    agent = TaskManagerAgent(storage_dir=test_dir)
    
    print("\n=== Conversation Flow Test ===\n")
    
    # Simulate a conversation
    conversation = [
        "Hi, I need help managing my tasks",
        "I need to finish a report by tomorrow, it's high priority",
        "Also remind me to call John next week",
        "What tasks do I have now?",
        "Mark the report as in progress",
        "I prefer dark theme",
        "Thanks for your help!"
    ]
    
    for user_input in conversation:
        print(f"\nUser: {user_input}")
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)

def test_error_handling():
    """Test how the agent handles edge cases and errors"""
    # Create a clean test directory
    test_dir = "test_error_handling"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize the agent
    agent = TaskManagerAgent(storage_dir=test_dir)
    
    print("\n=== Error Handling Test ===\n")
    
    # Test with empty input
    print("Testing empty input...")
    response = agent.agent_loop("")
    print(f"Agent: {response}")
    
    # Test with ambiguous input
    print("\nTesting ambiguous input...")
    response = agent.agent_loop("update it")
    print(f"Agent: {response}")
    
    # Test with non-task related input
    print("\nTesting non-task related input...")
    response = agent.agent_loop("What's the weather like today?")
    print(f"Agent: {response}")
    
    # Test with very long input
    print("\nTesting very long input...")
    long_input = "I need to " + "really " * 50 + "finish this task"
    response = agent.agent_loop(long_input)
    print(f"Agent: {response}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_task_manager()
    test_conversation_flow()
    test_error_handling()
