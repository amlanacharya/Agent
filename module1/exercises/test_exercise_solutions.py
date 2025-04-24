"""
Test Script for Lesson 1 Exercise Solutions
----------------------------------------
This script demonstrates how to test the exercise solutions from Lesson 1.
"""

# Adjust the import path based on how you're running the script
try:
    # When running from the module1/exercises directory
    from exercise_solutions import EnhancedAgent
except ImportError:
    # When running from the project root
    from module1.exercises.exercise_solutions import EnhancedAgent

def test_enhanced_agent():
    """Test the EnhancedAgent implementation"""
    agent = EnhancedAgent()
    
    print("=== Enhanced Agent Test ===")
    
    # Test basic responses
    inputs = [
        "Hello there!",
        "What's your name?",
        "help",
        "my name is Alice",
        "What time is it?",
        "I need to organize my tasks",
        "history",
        "clear",
        "Goodbye!"
    ]
    
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        response = agent.agent_loop(user_input)
        print(f"Agent: {response}")
        print("-" * 50)
    
    # Test intent recognition
    print("\n=== Intent Recognition Test ===")
    test_inputs = {
        "statement": "I like programming.",
        "question": "How does this work?",
        "command": "help me with this"
    }
    
    for intent, text in test_inputs.items():
        processed = agent.sense(text)
        detected = processed.get('intent', 'unknown')
        print(f"Input: '{text}'")
        print(f"Expected intent: {intent}")
        print(f"Detected intent: {detected}")
        print(f"Match: {'✓' if intent == detected else '✗'}")
        print("-" * 50)

if __name__ == "__main__":
    test_enhanced_agent()
