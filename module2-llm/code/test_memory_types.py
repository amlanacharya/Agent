"""
Test Script for Memory Types with LLM Integration
--------------------------------------------
This script demonstrates how to test the memory implementations with Groq API integration.
"""

import os
import time
import json
from datetime import datetime

# Adjust the import path based on how you're running the script
try:
    # When running from the module2-llm/code directory
    from memory_types import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemory, AgentMemorySystem
except ImportError:
    # When running from the project root
    from module2_llm.code.memory_types import WorkingMemory, ShortTermMemory, LongTermMemory, EpisodicMemory, AgentMemorySystem


def test_working_memory():
    """Test working memory functionality"""
    print("\n=== Testing Working Memory ===\n")
    
    # Create working memory
    working_memory = WorkingMemory(capacity=3)
    
    # Add items
    working_memory.add("User is asking about machine learning")
    working_memory.add("User mentioned they are a beginner")
    working_memory.add("User wants resources for getting started")
    
    # Get all items
    print("All items in working memory:")
    for i, item in enumerate(working_memory.get_all()):
        print(f"{i+1}. {item['content']}")
    
    # Test summarization with LLM
    print("\nWorking memory summary:")
    summary = working_memory.summarize()
    print(summary)
    
    # Test capacity limit
    print("\nTesting capacity limit...")
    working_memory.add("This should replace the oldest item")
    working_memory.add("This should also replace an older item")
    
    print("\nAfter adding more items:")
    for i, item in enumerate(working_memory.get_all()):
        print(f"{i+1}. {item['content']}")
    
    # Clear memory
    working_memory.clear()
    print("\nAfter clearing memory:")
    print(f"Items in memory: {len(working_memory.get_all())}")


def test_short_term_memory():
    """Test short-term memory functionality"""
    print("\n=== Testing Short-Term Memory ===\n")
    
    # Create short-term memory
    short_term = ShortTermMemory(capacity=5)
    
    # Add conversation turns
    short_term.add({"role": "user", "content": "What is machine learning?"})
    short_term.add({"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."})
    short_term.add({"role": "user", "content": "What are some applications of it?"})
    short_term.add({"role": "assistant", "content": "Applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles."})
    
    # Get recent items
    print("Recent conversation:")
    for i, item in enumerate(short_term.get_recent(3)):
        print(f"{item['content']['role']}: {item['content']['content']}")
    
    # Test key information extraction with LLM
    print("\nExtracting key information about 'applications':")
    info = short_term.extract_key_information("What applications of machine learning were mentioned?")
    print(info)
    
    # Add more items and test capacity
    print("\nTesting capacity limit...")
    for i in range(5):
        short_term.add({"role": "user", "content": f"Test message {i+1}"})
    
    print("\nAfter adding more items:")
    print(f"Items in memory: {len(short_term.get_all())}")
    print("First few items:")
    for i, item in enumerate(short_term.get_recent(3)):
        print(f"{i+1}. {item['content']}")


def test_long_term_memory():
    """Test long-term memory functionality"""
    print("\n=== Testing Long-Term Memory ===\n")
    
    # Create a temporary file for testing
    test_file = "test_long_term_memory.json"
    
    # Create long-term memory
    long_term = LongTermMemory(test_file)
    
    # Add facts
    long_term.add_fact("Machine learning is a subset of artificial intelligence.")
    long_term.add_fact("Neural networks are inspired by the human brain.")
    long_term.add_fact("Deep learning is a subset of machine learning that uses neural networks with many layers.")
    
    # Add concepts
    long_term.add_concept("supervised learning", {
        "definition": "A type of machine learning where the model is trained on labeled data.",
        "examples": ["classification", "regression"]
    })
    
    long_term.add_concept("unsupervised learning", {
        "definition": "A type of machine learning where the model finds patterns in unlabeled data.",
        "examples": ["clustering", "dimensionality reduction"]
    })
    
    # Get facts
    print("Facts in long-term memory:")
    for i, fact in enumerate(long_term.get_facts()):
        print(f"{i+1}. {fact['content']}")
    
    # Get concepts
    print("\nConcepts in long-term memory:")
    for name, concept in long_term.get_all_concepts().items():
        print(f"- {name}: {concept['info']}")
    
    # Test search with LLM
    print("\nSearching for 'neural networks':")
    results = long_term.search("neural networks")
    for result in results:
        print(f"- {result['content']}")
    
    # Clean up test file
    if os.path.exists(test_file):
        os.remove(test_file)


def test_episodic_memory():
    """Test episodic memory functionality"""
    print("\n=== Testing Episodic Memory ===\n")
    
    # Create episodic memory
    episodic = EpisodicMemory()
    
    # Add episodes
    episodic.add_episode({
        "user_input": "How do I start learning machine learning?",
        "agent_response": "I recommend starting with Python basics, then moving to libraries like NumPy, Pandas, and scikit-learn.",
        "metadata": {"topic": "machine learning", "intent": "learning_path"}
    })
    
    episodic.add_episode({
        "user_input": "What's the difference between supervised and unsupervised learning?",
        "agent_response": "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.",
        "metadata": {"topic": "machine learning", "intent": "concept_clarification"}
    })
    
    episodic.add_episode({
        "user_input": "Can you recommend a good book on deep learning?",
        "agent_response": "I recommend 'Deep Learning' by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.",
        "metadata": {"topic": "deep learning", "intent": "resource_request"}
    })
    
    # Get recent episodes
    print("Recent episodes:")
    for i, episode in enumerate(episodic.get_episodes(2)):
        print(f"Episode {i+1}:")
        print(f"User: {episode['user_input']}")
        print(f"Assistant: {episode['agent_response']}")
        print(f"Topic: {episode['metadata']['topic']}")
        print()
    
    # Test search with LLM
    print("Searching for episodes about 'learning resources':")
    results = episodic.search_episodes("learning resources")
    
    if isinstance(results, list) and results and isinstance(results[0], dict):
        for i, result in enumerate(results):
            if 'user_input' in result:
                print(f"Result {i+1}:")
                print(f"User: {result.get('user_input', 'N/A')}")
                print(f"Assistant: {result.get('agent_response', 'N/A')}")
                print()
            else:
                print(f"Result {i+1}: {result}")
    else:
        print("Search results:", results)


def test_agent_memory_system():
    """Test the integrated agent memory system"""
    print("\n=== Testing Agent Memory System ===\n")
    
    # Create a temporary directory for testing
    test_dir = "test_agent_memory"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create memory system
    memory_system = AgentMemorySystem(test_dir)
    
    # Process some inputs and responses
    print("Processing conversation...")
    
    memory_system.process_input("Hi, I'm John. I'm interested in learning about artificial intelligence.")
    memory_system.store_response("Hello John! I'd be happy to help you learn about AI. What specific areas are you interested in?")
    
    memory_system.process_input("I'm particularly interested in natural language processing and how chatbots work.")
    memory_system.store_response("Great choice! NLP is a fascinating field. Chatbots typically use techniques like intent recognition, entity extraction, and language generation.")
    
    memory_system.process_input("What libraries should I learn for NLP?")
    memory_system.store_response("For NLP in Python, I recommend starting with NLTK and spaCy. For more advanced work, look into Hugging Face's transformers library.")
    
    # Create an episode
    memory_system.create_episode(
        "Can you recommend a project for me to practice NLP?",
        "A great starter project would be building a simple text classifier to categorize news articles or product reviews. This will help you learn text preprocessing, feature extraction, and model training.",
        {"topic": "nlp", "intent": "project_recommendation"}
    )
    
    # Retrieve context for a query
    print("\nRetrieving context for 'What is John interested in?'")
    context = memory_system.retrieve_relevant_context("What is John interested in?")
    
    print("\nIntegrated Context:")
    print(context.get('integrated_context', 'No integrated context available.'))
    
    # Retrieve context for another query
    print("\nRetrieving context for 'NLP libraries'")
    context = memory_system.retrieve_relevant_context("NLP libraries")
    
    print("\nIntegrated Context:")
    print(context.get('integrated_context', 'No integrated context available.'))
    
    # Clean up test directory
    for file in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, file))
    os.rmdir(test_dir)


if __name__ == "__main__":
    print("Testing Memory Types with LLM Integration")
    print("=" * 50)
    print("Note: These tests require a valid Groq API key.")
    print("If you haven't set up your API key, please do so before running these tests.")
    print("=" * 50)
    
    try:
        # Run individual tests
        test_working_memory()
        test_short_term_memory()
        test_long_term_memory()
        test_episodic_memory()
        test_agent_memory_system()
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Make sure your Groq API key is set up correctly.")
