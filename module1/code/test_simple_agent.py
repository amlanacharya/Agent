"""
Test Script for Simple Agent
---------------------------
This script demonstrates how to test the SimpleAgent implementation.
"""

# Adjust the import path based on how you're running the script
try:
    # When running from the module1/code directory
    from simple_agent import SimpleAgent
except ImportError:
    # When running from the project root
    from module1.code.simple_agent import SimpleAgent

def test_agent():
    """Run a series of tests on the SimpleAgent"""
    agent = SimpleAgent()

    # Test with different inputs
    inputs = [
        "Hello there!",
        "What's your name?",
        "How does this work?",
        "Goodbye!"
    ]

    print("=== Simple Agent Test ===")
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
        print("-" * 30)

    # Show the agent's internal state
    print("\n=== Agent State ===")
    for key, value in agent.state.items():
        print(f"{key}: {value}")

def test_conversation_flow():
    """Test a more natural conversation flow"""
    agent = SimpleAgent()

    print("\n=== Conversation Flow Test ===")

    # Simulate a conversation
    conversation = [
        "Hello, agent!",
        "Can you help me with something?",
        "I need to organize my tasks",
        "Thanks for your help",
        "Bye now"
    ]

    for user_input in conversation:
        print(f"\nUser: {user_input}")
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    test_agent()
    test_conversation_flow()
