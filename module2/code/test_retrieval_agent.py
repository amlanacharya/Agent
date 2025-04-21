"""
Test Script for Retrieval Patterns
--------------------------------
This script demonstrates and tests the retrieval patterns
implemented in retrieval_agent.py.
"""

import os
import time
import shutil
from vector_store import RetrievalMemory
from retrieval_agent import (
    recency_based_retrieval,
    conversation_aware_retrieval,
    multi_query_retrieval
)

def setup_test_memory():
    """Set up a test memory system with sample data"""
    # Create a test directory
    test_dir = "test_retrieval_memory"
    
    # Remove the directory if it exists
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Create retrieval memory
    memory = RetrievalMemory(storage_dir=test_dir)
    
    # Add some test data with timestamps
    # Recent items (within the last day)
    current_time = time.time()
    one_hour_ago = current_time - 3600
    six_hours_ago = current_time - 21600
    one_day_ago = current_time - 86400
    two_days_ago = current_time - 172800
    
    # Store items with different timestamps
    memory.store(
        "Artificial intelligence is the simulation of human intelligence by machines.",
        {"category": "AI", "importance": "high", "timestamp": one_hour_ago}
    )
    
    memory.store(
        "Machine learning is a subset of AI that enables systems to learn from data.",
        {"category": "AI", "importance": "high", "timestamp": six_hours_ago}
    )
    
    memory.store(
        "Neural networks are computing systems inspired by biological neural networks.",
        {"category": "AI", "importance": "medium", "timestamp": one_day_ago}
    )
    
    memory.store(
        "Python is a high-level programming language known for its readability.",
        {"category": "programming", "importance": "medium", "timestamp": two_days_ago}
    )
    
    memory.store(
        "JavaScript is a scripting language used primarily for web development.",
        {"category": "programming", "importance": "medium", "timestamp": two_days_ago}
    )
    
    return memory, test_dir

def test_recency_based_retrieval():
    """Test recency-based retrieval"""
    print("\n=== Testing Recency-Based Retrieval ===")
    
    # Set up test memory
    memory, test_dir = setup_test_memory()
    print(f"Created test memory with {len(memory)} items")
    
    # Test with different max_age settings
    print("\nTesting with max_age_hours=12 (should return only recent items):")
    results = recency_based_retrieval("AI", memory, max_age_hours=12, top_k=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:50]}... (score: {result.get('combined_score', 0):.4f})")
        print(f"   Original score: {result.get('score', 0):.4f}")
        timestamp = result.get('metadata', {}).get('timestamp', 0)
        age_hours = (time.time() - timestamp) / 3600
        print(f"   Age: {age_hours:.2f} hours")
    
    # Test with a longer time window
    print("\nTesting with max_age_hours=48 (should return more items):")
    results = recency_based_retrieval("AI", memory, max_age_hours=48, top_k=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:50]}... (score: {result.get('combined_score', 0):.4f})")
        print(f"   Original score: {result.get('score', 0):.4f}")
        timestamp = result.get('metadata', {}).get('timestamp', 0)
        age_hours = (time.time() - timestamp) / 3600
        print(f"   Age: {age_hours:.2f} hours")
    
    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Recency-based retrieval test completed.")

def test_conversation_aware_retrieval():
    """Test conversation-aware retrieval"""
    print("\n=== Testing Conversation-Aware Retrieval ===")
    
    # Set up test memory
    memory, test_dir = setup_test_memory()
    print(f"Created test memory with {len(memory)} items")
    
    # Create a sample conversation history
    conversation_history = [
        {
            "user_input": "Tell me about programming languages",
            "agent_response": "Programming languages are formal languages used to create instructions for computers."
        },
        {
            "user_input": "What's the difference between Python and JavaScript?",
            "agent_response": "Python is often used for backend development and data science, while JavaScript is primarily for web development."
        },
        {
            "user_input": "Which one should I learn first?",
            "agent_response": "It depends on your goals. Python is often recommended for beginners due to its readability."
        }
    ]
    
    # Test with a query that should be enhanced by conversation context
    print("\nTesting with conversation context about programming languages:")
    query = "Which language is better?"
    
    print(f"Original query: '{query}'")
    print("Recent conversation:")
    for i, turn in enumerate(conversation_history[-3:]):
        print(f"  Turn {i+1} - User: {turn['user_input']}")
    
    # Get results with conversation context
    results = conversation_aware_retrieval(query, conversation_history, memory, top_k=2)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:50]}... (score: {result.get('score', 0):.4f})")
        print(f"   Enhanced with: {result.get('enhanced_with', {})}")
    
    # Compare with results without conversation context
    print("\nComparing with regular retrieval (without conversation context):")
    regular_results = memory.retrieve(query, top_k=2)
    
    print(f"Found {len(regular_results)} results:")
    for i, result in enumerate(regular_results):
        print(f"{i+1}. {result['text'][:50]}... (score: {result.get('score', 0):.4f})")
    
    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Conversation-aware retrieval test completed.")

def test_multi_query_retrieval():
    """Test multi-query retrieval"""
    print("\n=== Testing Multi-Query Retrieval ===")
    
    # Set up test memory
    memory, test_dir = setup_test_memory()
    print(f"Created test memory with {len(memory)} items")
    
    # Test with a query that can have multiple variations
    query = "How do artificial intelligence systems work?"
    print(f"Original query: '{query}'")
    
    # Get results with multi-query approach
    results = multi_query_retrieval(query, memory, top_k=3)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:50]}... (score: {result.get('score', 0):.4f})")
        print(f"   Found by query: '{result.get('found_by_query', 'unknown')}'")
    
    # Compare with results from a single query
    print("\nComparing with regular retrieval (single query):")
    regular_results = memory.retrieve(query, top_k=3)
    
    print(f"Found {len(regular_results)} results:")
    for i, result in enumerate(regular_results):
        print(f"{i+1}. {result['text'][:50]}... (score: {result.get('score', 0):.4f})")
    
    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Multi-query retrieval test completed.")

def run_all_tests():
    """Run all retrieval pattern tests"""
    print("=== Retrieval Patterns Tests ===")
    
    test_recency_based_retrieval()
    test_conversation_aware_retrieval()
    test_multi_query_retrieval()
    
    print("\nAll retrieval pattern tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
