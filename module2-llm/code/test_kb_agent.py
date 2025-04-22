"""
Test Script for Knowledge Base Assistant with Groq API
--------------------------------------------------
This script demonstrates how to test the knowledge base assistant implementation
with Groq API integration.
"""

import os
import time
import json
import shutil
from datetime import datetime

# Adjust the import path based on how you're running the script
try:
    # When running from the module2-llm/code directory
    from kb_agent import KnowledgeBaseAssistant
    from knowledge_base import KnowledgeBase
except ImportError:
    # When running from the project root
    from module2_llm.code.kb_agent import KnowledgeBaseAssistant
    from module2_llm.code.knowledge_base import KnowledgeBase


def test_basic_functionality():
    """Test basic knowledge base assistant functionality"""
    print("\n=== Testing Basic Functionality ===\n")
    
    # Create a temporary directory for testing
    test_dir = "test_kb_assistant"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant(test_dir)
    
    # Add some knowledge
    print("Adding knowledge to the assistant...")
    assistant.add_to_knowledge_base("Python is a high-level programming language known for its readability and simplicity.")
    assistant.add_to_knowledge_base("Python was created by Guido van Rossum and first released in 1991.")
    assistant.add_to_knowledge_base("Python is widely used in data science, machine learning, web development, and automation.")
    assistant.add_to_knowledge_base("Machine learning is a subset of artificial intelligence that enables systems to learn from data.")
    assistant.add_to_knowledge_base("Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.")
    
    # Test answering questions
    print("\nTesting question answering...")
    questions = [
        "What is Python?",
        "Who created Python?",
        "What is machine learning?"
    ]
    
    for question in questions:
        print(f"\nQuestion: '{question}'")
        answer = assistant.answer_question(question)
        print(f"Answer: {answer}")
    
    # Test handling statements
    print("\nTesting statement handling...")
    statements = [
        "Python is also the name of a snake.",
        "The Python programming language is named after Monty Python."
    ]
    
    for statement in statements:
        print(f"\nStatement: '{statement}'")
        response = assistant.handle_input(statement)
        print(f"Response: {response}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)


def test_conversation_context():
    """Test conversation context handling"""
    print("\n=== Testing Conversation Context ===\n")
    
    # Create a temporary directory for testing
    test_dir = "test_kb_assistant_context"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant(test_dir)
    
    # Add some knowledge
    print("Adding knowledge to the assistant...")
    assistant.add_to_knowledge_base("Python is a high-level programming language known for its readability and simplicity.")
    assistant.add_to_knowledge_base("Python was created by Guido van Rossum and first released in 1991.")
    assistant.add_to_knowledge_base("Python is widely used in data science, machine learning, web development, and automation.")
    
    # Simulate a conversation with context
    print("\nSimulating a conversation with context...")
    
    print("\nUser: What is Python?")
    response = assistant.handle_input("What is Python?")
    print(f"Assistant: {response}")
    
    print("\nUser: Who created it?")  # "it" refers to Python from context
    response = assistant.handle_input("Who created it?")
    print(f"Assistant: {response}")
    
    print("\nUser: When was it released?")  # "it" still refers to Python
    response = assistant.handle_input("When was it released?")
    print(f"Assistant: {response}")
    
    # Get conversation history
    print("\nRetrieving conversation history...")
    history = assistant.get_conversation_history(6)
    print(f"Retrieved {len(history)} conversation turns:")
    for i, turn in enumerate(history):
        role = turn['content']['role']
        content = turn['content']['content']
        print(f"{i+1}. {role}: {content[:50]}...")
    
    # Clean up test directory
    shutil.rmtree(test_dir)


def test_learning_modes():
    """Test different learning modes"""
    print("\n=== Testing Learning Modes ===\n")
    
    # Create a temporary directory for testing
    test_dir = "test_kb_assistant_learning"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant(test_dir)
    
    # Test passive learning mode
    print("\nTesting passive learning mode...")
    assistant.update_settings({"learning_mode": "passive"})
    
    # Add a fact-like statement
    print("\nUser: Python is an interpreted language.")
    response = assistant.handle_input("Python is an interpreted language.")
    print(f"Assistant: {response}")
    
    # Add a non-fact-like statement
    print("\nUser: I like programming in Python.")
    response = assistant.handle_input("I like programming in Python.")
    print(f"Assistant: {response}")
    
    # Test active learning mode
    print("\nTesting active learning mode...")
    assistant.update_settings({"learning_mode": "active"})
    
    # Add a statement
    print("\nUser: Python supports multiple programming paradigms.")
    response = assistant.handle_input("Python supports multiple programming paradigms.")
    print(f"Assistant: {response}")
    
    # Test learning mode off
    print("\nTesting learning mode off...")
    assistant.update_settings({"learning_mode": "off"})
    
    # Add a fact-like statement
    print("\nUser: Python has a large standard library.")
    response = assistant.handle_input("Python has a large standard library.")
    print(f"Assistant: {response}")
    
    # Get knowledge base summary
    print("\nGetting knowledge base summary...")
    summary = assistant.get_knowledge_base_summary()
    print(f"Summary: {summary}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)


def test_explanation_capabilities():
    """Test explanation capabilities"""
    print("\n=== Testing Explanation Capabilities ===\n")
    
    # Create a temporary directory for testing
    test_dir = "test_kb_assistant_explanation"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant(test_dir)
    
    # Add some knowledge
    print("Adding knowledge to the assistant...")
    assistant.add_to_knowledge_base("Python is a high-level programming language known for its readability and simplicity.")
    assistant.add_to_knowledge_base("Python was created by Guido van Rossum and first released in 1991.")
    assistant.add_to_knowledge_base("Python is widely used in data science, machine learning, web development, and automation.")
    assistant.add_to_knowledge_base("Machine learning is a subset of artificial intelligence that enables systems to learn from data.")
    assistant.add_to_knowledge_base("Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.")
    
    # Test explanation
    print("\nTesting answer explanation...")
    question = "What is Python used for?"
    print(f"Question: '{question}'")
    
    explanation = assistant.explain_answer(question)
    
    print(f"\nAnswer: {explanation['answer']}")
    print(f"Confidence: {explanation['confidence']:.2f}")
    print(f"\nRetrieval explanation: {explanation['retrieval_explanation']}")
    print(f"\nReasoning explanation: {explanation['reasoning_explanation']}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)


def test_uncertainty_handling():
    """Test uncertainty handling"""
    print("\n=== Testing Uncertainty Handling ===\n")
    
    # Create a temporary directory for testing
    test_dir = "test_kb_assistant_uncertainty"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a knowledge base assistant
    assistant = KnowledgeBaseAssistant(test_dir)
    
    # Add some knowledge
    print("Adding knowledge to the assistant...")
    assistant.add_to_knowledge_base("Python is a high-level programming language known for its readability and simplicity.")
    assistant.add_to_knowledge_base("Python was created by Guido van Rossum and first released in 1991.")
    
    # Test high confidence question
    print("\nTesting high confidence question...")
    high_confidence_question = "What is Python?"
    print(f"Question: '{high_confidence_question}'")
    answer = assistant.answer_question(high_confidence_question)
    print(f"Answer: {answer}")
    
    # Test medium confidence question
    print("\nTesting medium confidence question...")
    medium_confidence_question = "What are some features of Python?"
    print(f"Question: '{medium_confidence_question}'")
    answer = assistant.answer_question(medium_confidence_question)
    print(f"Answer: {answer}")
    
    # Test low confidence question
    print("\nTesting low confidence question...")
    low_confidence_question = "What is the best way to learn Python?"
    print(f"Question: '{low_confidence_question}'")
    answer = assistant.answer_question(low_confidence_question)
    print(f"Answer: {answer}")
    
    # Clean up test directory
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("Testing Knowledge Base Assistant with Groq API")
    print("=" * 50)
    print("Note: These tests require a valid Groq API key.")
    print("If you haven't set up your API key, please do so before running these tests.")
    print("=" * 50)
    
    try:
        # Run individual tests
        test_basic_functionality()
        test_conversation_context()
        test_learning_modes()
        test_explanation_capabilities()
        test_uncertainty_handling()
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Make sure your Groq API key is set up correctly.")
