"""
Test Script for Lesson 2 Exercise Solutions
----------------------------------------
This script demonstrates how to test the prompt engineering exercise solutions from Lesson 2.
"""

# Adjust the import path based on how you're running the script
try:
    # When running from the module1/exercises directory
    from prompt_exercises import TaskParserExercise, RoleBasedPromptsExercise, ChainOfThoughtExercise
except ImportError:
    # When running from the project root
    from module1.exercises.prompt_exercises import TaskParserExercise, RoleBasedPromptsExercise, ChainOfThoughtExercise

def test_task_parser():
    """Test the TaskParserExercise implementation"""
    task_parser = TaskParserExercise()
    
    print("=== Task Parser Exercise Test ===")
    
    # Test with different task descriptions
    inputs = [
        "Schedule a meeting with the team tomorrow at 2pm",
        "I need to submit the report by Friday, it's urgent",
        "Remind me to call John next Monday morning",
        "Buy groceries sometime this weekend"
    ]
    
    for user_input in inputs:
        print(f"\nInput: {user_input}")
        prompt = task_parser.parse_task(user_input)
        print("Generated Prompt:")
        print(prompt)
        print("-" * 50)

def test_role_based_prompts():
    """Test the RoleBasedPromptsExercise implementation"""
    role_prompts = RoleBasedPromptsExercise()
    
    print("\n=== Role-Based Prompts Exercise Test ===")
    
    # Test with different roles and queries
    roles = ["professional", "friendly", "technical"]
    queries = [
        "I need to reschedule our meeting",
        "Can you help me with this task?",
        "What's the status of the project?"
    ]
    
    for role in roles:
        print(f"\n[Testing {role.upper()} role]")
        for query in queries:
            print(f"\nUser query: {query}")
            prompt = role_prompts.get_role_prompt(role, query)
            print("Generated Prompt:")
            print(prompt)
            print("-" * 30)

def test_chain_of_thought():
    """Test the ChainOfThoughtExercise implementation"""
    cot = ChainOfThoughtExercise()
    
    print("\n=== Chain-of-Thought Exercise Test ===")
    
    # Test with different problems
    problems = [
        "How can I improve team communication?",
        "What's the best way to organize project documentation?",
        "How should I prioritize tasks for the week?"
    ]
    
    for problem in problems:
        print(f"\nProblem: {problem}")
        prompt = cot.generate_cot_prompt(problem)
        print("Generated Prompt:")
        print(prompt)
        print("-" * 50)

if __name__ == "__main__":
    test_task_parser()
    test_role_based_prompts()
    test_chain_of_thought()
