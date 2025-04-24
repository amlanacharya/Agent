"""
Demo Script for Module 1 Agents
----------------------------
This script provides a unified interface to demonstrate all agent implementations from Module 1.
"""

import time
import argparse

# Import all agent implementations
try:
    from module1.code.simple_agent import SimpleAgent
    from module1.code.prompt_driven_agent import PromptDrivenAgent
    from module1.code.stateful_agent import StatefulAgent
    from module1.code.task_manager_agent import TaskManagerAgent
except ImportError:
    # When running from the module1 directory
    from code.simple_agent import SimpleAgent
    from code.prompt_driven_agent import PromptDrivenAgent
    from code.stateful_agent import StatefulAgent
    from code.task_manager_agent import TaskManagerAgent

def demo_simple_agent():
    """Demonstrate the SimpleAgent implementation"""
    print("\n" + "="*50)
    print("🤖 Simple Agent Demo")
    print("="*50)
    
    agent = SimpleAgent()
    
    print("This agent demonstrates the basic sense-think-act loop.")
    print("Try typing 'hello', asking questions, or saying 'bye'.")
    print("-"*50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting Simple Agent demo...")
            break
            
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")

def demo_prompt_driven_agent():
    """Demonstrate the PromptDrivenAgent implementation"""
    print("\n" + "="*50)
    print("🤖 Prompt-Driven Agent Demo")
    print("="*50)
    
    agent = PromptDrivenAgent()
    
    print("This agent uses prompt templates to generate more sophisticated responses.")
    print("Try adding tasks, asking about your tasks, or requesting help.")
    print("-"*50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting Prompt-Driven Agent demo...")
            break
            
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")

def demo_stateful_agent():
    """Demonstrate the StatefulAgent implementation"""
    print("\n" + "="*50)
    print("🤖 Stateful Agent Demo")
    print("="*50)
    
    # Create a temporary directory for the demo
    import os
    import shutil
    test_dir = "demo_stateful_agent_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    
    agent = StatefulAgent(storage_dir=test_dir)
    
    print("This agent maintains state between interactions.")
    print("Try telling it your name, adding tasks, or asking about previous interactions.")
    print("-"*50)
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting Stateful Agent demo...")
                break
                
            response = agent.agent_loop(user_input)
            print(f"Agent: {response}")
            
            # Show state summary occasionally
            if len(agent.state_manager.get_conversation_history()) % 3 == 0:
                print("\n[State Summary]")
                summary = agent.get_state_summary()
                print(f"User: {summary['user_profile']['name'] or 'Unknown'}")
                print(f"Tasks: {summary['task_count']}")
                print(f"Conversation length: {summary['conversation_length']}")
                print()
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def demo_task_manager():
    """Demonstrate the TaskManagerAgent implementation"""
    print("\n" + "="*50)
    print("🤖 Task Manager Agent Demo")
    print("="*50)
    
    # Create a temporary directory for the demo
    import os
    import shutil
    test_dir = "demo_task_manager_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    agent = TaskManagerAgent(storage_dir=test_dir)
    
    print("This is the complete Task Manager implementation.")
    print("Try commands like:")
    print("- Add a new task: Finish the report by Friday")
    print("- Show my tasks")
    print("- Mark the report task as completed")
    print("- I prefer dark theme")
    print("-"*50)
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting Task Manager demo...")
                break
                
            response = agent.agent_loop(user_input)
            print(f"Agent: {response}")
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(description="Demo for Module 1 Agents")
    parser.add_argument('agent', nargs='?', choices=['simple', 'prompt', 'stateful', 'task'], 
                        help='Which agent to demo (simple, prompt, stateful, or task)')
    
    args = parser.parse_args()
    
    if args.agent == 'simple':
        demo_simple_agent()
    elif args.agent == 'prompt':
        demo_prompt_driven_agent()
    elif args.agent == 'stateful':
        demo_stateful_agent()
    elif args.agent == 'task':
        demo_task_manager()
    else:
        # If no agent specified, show menu
        while True:
            print("\n🤖 Module 1: Agent Demos")
            print("1. Simple Agent (Basic sense-think-act loop)")
            print("2. Prompt-Driven Agent (Using prompt templates)")
            print("3. Stateful Agent (With state management)")
            print("4. Task Manager Agent (Complete implementation)")
            print("0. Exit")
            
            choice = input("\nSelect an agent to demo (0-4): ")
            
            if choice == '1':
                demo_simple_agent()
            elif choice == '2':
                demo_prompt_driven_agent()
            elif choice == '3':
                demo_stateful_agent()
            elif choice == '4':
                demo_task_manager()
            elif choice == '0':
                print("Exiting demo...")
                break
            else:
                print("Invalid choice. Please select 0-4.")

if __name__ == "__main__":
    main()
