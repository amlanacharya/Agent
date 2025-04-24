"""
Test Script for Prompt-Driven Agent
---------------------------------
This script demonstrates how to test the PromptDrivenAgent implementation.
"""

# Adjust the import path based on how you're running the script
try:
    # When running from the module1/code directory
    from prompt_driven_agent import PromptDrivenAgent
    from prompt_template import PromptTemplate
except ImportError:
    # When running from the project root
    from module1.code.prompt_driven_agent import PromptDrivenAgent
    from module1.code.prompt_template import PromptTemplate

def test_prompt_driven_agent():
    """Test the basic functionality of the PromptDrivenAgent"""
    agent = PromptDrivenAgent()
    
    # Test with different inputs
    inputs = [
        "Add a new task: Call John about the project tomorrow",
        "I need to prepare for the presentation next week",
        "Show me my tasks",
        "Help me understand what you can do"
    ]
    
    print("=== Prompt-Driven Agent Test ===")
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
        print("-" * 50)
    
    # Show the agent's tasks
    print("\n=== Agent Tasks ===")
    for i, task in enumerate(agent.state['tasks']):
        print(f"Task {i+1}: {task['description']}")
        print(f"  Date: {task['date']}")
        print(f"  Priority: {task['priority']}")
        print()

def test_prompt_templates():
    """Test the prompt templates used by the agent"""
    agent = PromptDrivenAgent()
    
    print("=== Prompt Template Examples ===")
    
    # Test greeting template
    greeting_prompt = agent.prompt_library.format_prompt(
        "greeting",
        assistant_type="task management",
        assistant_name="TaskBot",
        user_name="Alice",
        tone="friendly"
    )
    print("Greeting Template:")
    print(greeting_prompt)
    print("-" * 50)
    
    # Test task creation template
    task_prompt = agent.prompt_library.format_prompt(
        "task_creation",
        description="Finish the quarterly report",
        due_date="next Friday",
        priority="high",
        tone="professional"
    )
    print("\nTask Creation Template:")
    print(task_prompt)
    print("-" * 50)
    
    # Test chain-of-thought template
    cot_prompt = agent.prompt_library.format_prompt(
        "cot_planning",
        objective="Organize team documentation",
        task_description="Create a centralized system for team documentation that is easy to search and maintain"
    )
    print("\nChain-of-Thought Template:")
    print(cot_prompt)
    print("-" * 50)

if __name__ == "__main__":
    test_prompt_driven_agent()
    test_prompt_templates()
