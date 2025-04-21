"""
Test Script for Advanced Retrieval Classes
---------------------------------------
This script demonstrates and tests the advanced retrieval classes
implemented in retrieval_agent.py.
"""

import os
import time
import shutil
from memory_types import EpisodicMemory
from vector_store import SimpleVectorDB, RetrievalMemory
from retrieval_agent import (
    ConversationMemory,
    HybridRetrievalSystem,
    ContextAwareRetrieval,
    RelevanceScorer
)

def setup_test_memory():
    """Set up a test memory system with sample data"""
    # Create a test directory
    test_dir = "test_advanced_retrieval"
    
    # Remove the directory if it exists
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Create retrieval memory
    memory = RetrievalMemory(storage_dir=test_dir)
    
    # Add some test data
    memory.store(
        "Artificial intelligence is the simulation of human intelligence by machines.",
        {"category": "AI", "importance": "high", "timestamp": time.time() - 3600}  # 1 hour ago
    )
    
    memory.store(
        "Machine learning is a subset of AI that enables systems to learn from data.",
        {"category": "AI", "importance": "high", "timestamp": time.time() - 7200}  # 2 hours ago
    )
    
    memory.store(
        "Neural networks are computing systems inspired by biological neural networks.",
        {"category": "AI", "importance": "medium", "timestamp": time.time() - 86400}  # 1 day ago
    )
    
    memory.store(
        "Python is a high-level programming language known for its readability.",
        {"category": "programming", "importance": "medium", "timestamp": time.time() - 172800}  # 2 days ago
    )
    
    memory.store(
        "JavaScript is a scripting language used primarily for web development.",
        {"category": "programming", "importance": "medium", "timestamp": time.time() - 259200}  # 3 days ago
    )
    
    return memory, test_dir

def setup_episodic_memory():
    """Set up a test episodic memory with sample data"""
    episodic = EpisodicMemory()
    
    # Add some conversation episodes
    episodic.record_episode(
        episode_type="conversation",
        content={
            "user_input": "What is artificial intelligence?",
            "agent_response": "Artificial intelligence is the simulation of human intelligence by machines."
        },
        metadata={"timestamp": time.time() - 3600}  # 1 hour ago
    )
    
    episodic.record_episode(
        episode_type="conversation",
        content={
            "user_input": "How does machine learning work?",
            "agent_response": "Machine learning works by training algorithms on data to make predictions or decisions."
        },
        metadata={"timestamp": time.time() - 7200}  # 2 hours ago
    )
    
    episodic.record_episode(
        episode_type="action",
        content={
            "action": "search",
            "query": "programming languages"
        },
        metadata={"timestamp": time.time() - 86400}  # 1 day ago
    )
    
    return episodic

def test_conversation_memory():
    """Test the ConversationMemory class"""
    print("\n=== Testing ConversationMemory ===")
    
    # Create a vector store for the conversation memory
    vector_store = SimpleVectorDB()
    
    # Create conversation memory
    memory = ConversationMemory(vector_store=vector_store, max_history=5)
    print(f"Created conversation memory: {memory}")
    
    # Add some conversation turns
    print("\nAdding conversation turns...")
    memory.add_turn(
        user_input="Hello, I'm interested in learning about AI.",
        agent_response="That's great! AI is a fascinating field. What specific aspects are you interested in?"
    )
    
    memory.add_turn(
        user_input="I want to learn about machine learning.",
        agent_response="Machine learning is a subset of AI that focuses on algorithms that can learn from data."
    )
    
    memory.add_turn(
        user_input="What programming languages are used for machine learning?",
        agent_response="Python is the most popular language for machine learning, but R and Julia are also commonly used."
    )
    
    print(f"Added {len(memory)} conversation turns")
    
    # Test get_recent_turns
    print("\nTesting get_recent_turns...")
    recent_turns = memory.get_recent_turns(2)
    print(f"Got {len(recent_turns)} recent turns:")
    for i, turn in enumerate(recent_turns):
        print(f"{i+1}. User: {turn['user_input'][:30]}...")
        print(f"   Agent: {turn['agent_response'][:30]}...")
    
    # Test search_conversation
    print("\nTesting search_conversation...")
    query = "programming languages"
    results = memory.search_conversation(query, top_k=2)
    
    print(f"Search query: '{query}'")
    print(f"Found {len(results)} relevant turns:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   User: {result['turn']['user_input']}")
        print(f"   Agent: {result['turn']['agent_response']}")
    
    # Test get_conversation_summary
    print("\nTesting get_conversation_summary...")
    summary = memory.get_conversation_summary()
    print("Conversation summary:")
    print(summary)
    
    # Test focused summary
    print("\nTesting focused summary...")
    focused_summary = memory.get_conversation_summary(query="machine learning")
    print("Focused summary on 'machine learning':")
    print(focused_summary)
    
    print("ConversationMemory test completed.")

def test_hybrid_retrieval_system():
    """Test the HybridRetrievalSystem class"""
    print("\n=== Testing HybridRetrievalSystem ===")
    
    # Set up test memories
    vector_memory, test_dir = setup_test_memory()
    episodic_memory = setup_episodic_memory()
    
    # Create a simple long-term memory
    class SimpleLongTermMemory:
        def __init__(self):
            self.memory = {
                "what is ai": "Artificial Intelligence (AI) is the field of computer science focused on creating machines that can perform tasks that typically require human intelligence.",
                "best programming language": "The best programming language depends on your specific needs and the task at hand."
            }
        
        def retrieve(self, key):
            # Simple case-insensitive partial matching
            for k, v in self.memory.items():
                if key.lower() in k.lower() or k.lower() in key.lower():
                    return v
            return None
    
    long_term_memory = SimpleLongTermMemory()
    
    # Create hybrid retrieval system
    hybrid_system = HybridRetrievalSystem(
        vector_memory=vector_memory,
        episodic_memory=episodic_memory,
        long_term_memory=long_term_memory
    )
    
    print(f"Created hybrid retrieval system: {hybrid_system}")
    
    # Test retrieval with exact match in long-term memory
    print("\nTesting retrieval with exact match in long-term memory...")
    query = "What is AI?"
    results = hybrid_system.retrieve(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Source: {result.get('source', 'unknown')}")
        print(f"   Score: {result.get('score', 0):.4f}")
        print(f"   Text: {result.get('text', '')[:50]}...")
    
    # Test retrieval with conversation context
    print("\nTesting retrieval with conversation context...")
    conversation_context = [
        {"user_input": "I'm learning to code", "agent_response": "That's great!"},
        {"user_input": "Which language should I start with?", "agent_response": "Python is often recommended for beginners."}
    ]
    
    query = "programming recommendations"
    results = hybrid_system.retrieve(query, conversation_context=conversation_context, top_k=3)
    
    print(f"Query: '{query}' with conversation context")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Source: {result.get('source', 'unknown')}")
        print(f"   Score: {result.get('score', 0):.4f}")
        print(f"   Text: {result.get('text', '')[:50]}...")
    
    # Test retrieval with explanation
    print("\nTesting retrieval with explanation...")
    results = hybrid_system.retrieve_with_explanation(query, conversation_context=conversation_context, top_k=2)
    
    print(f"Query: '{query}' with explanation")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Source: {result.get('source', 'unknown')}")
        print(f"   Explanation: {result.get('explanation', '')}")
        print(f"   Text: {result.get('text', '')[:50]}...")
    
    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("HybridRetrievalSystem test completed.")

def test_context_aware_retrieval():
    """Test the ContextAwareRetrieval class"""
    print("\n=== Testing ContextAwareRetrieval ===")
    
    # Set up test memory
    vector_memory, test_dir = setup_test_memory()
    
    # Create conversation memory
    conversation_memory = ConversationMemory()
    
    # Add some conversation turns
    conversation_memory.add_turn(
        user_input="I want to learn programming.",
        agent_response="That's a great goal! There are many programming languages to choose from."
    )
    
    conversation_memory.add_turn(
        user_input="Which language is best for beginners?",
        agent_response="Python is often recommended for beginners due to its readability and simplicity."
    )
    
    # Create context-aware retrieval
    context_retrieval = ContextAwareRetrieval(
        memory_system=vector_memory,
        conversation_memory=conversation_memory
    )
    
    print(f"Created context-aware retrieval: {context_retrieval}")
    
    # Test retrieval with user profile
    print("\nTesting retrieval with user profile...")
    user_profile = {
        "interests": ["AI", "machine learning", "data science"],
        "expertise_level": "beginner"
    }
    
    query = "programming languages"
    results = context_retrieval.retrieve(query, user_profile=user_profile, top_k=2)
    
    print(f"Query: '{query}' with user profile")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.get('score', 0):.4f}")
        print(f"   Text: {result.get('text', '')[:50]}...")
        print(f"   Context info: {result.get('context_info', {})}")
    
    # Test retrieval with context explanation
    print("\nTesting retrieval with context explanation...")
    result_with_explanation = context_retrieval.retrieve_with_context_explanation(
        query, user_profile=user_profile, top_k=2
    )
    
    print(f"Query: '{query}' with context explanation")
    print("Context explanation:")
    for key, value in result_with_explanation['context_explanation'].items():
        if key != 'recent_conversation':
            print(f"   {key}: {value}")
    
    print(f"Found {len(result_with_explanation['results'])} results")
    
    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("ContextAwareRetrieval test completed.")

def test_relevance_scorer():
    """Test the RelevanceScorer class"""
    print("\n=== Testing RelevanceScorer ===")
    
    # Create a relevance scorer
    scorer = RelevanceScorer()
    print(f"Created relevance scorer: {scorer}")
    
    # Create some test items
    items = [
        {
            "id": "item1",
            "text": "Artificial intelligence is the simulation of human intelligence by machines.",
            "score": 0.85,
            "metadata": {"timestamp": time.time() - 3600, "category": "AI"}  # 1 hour ago
        },
        {
            "id": "item2",
            "text": "Python is a high-level programming language known for its readability.",
            "score": 0.75,
            "metadata": {"timestamp": time.time() - 86400, "category": "programming"}  # 1 day ago
        },
        {
            "id": "item3",
            "text": "Machine learning is a subset of AI that enables systems to learn from data.",
            "score": 0.65,
            "metadata": {"timestamp": time.time() - 7200, "category": "AI"}  # 2 hours ago
        }
    ]
    
    # Test scoring individual items
    print("\nTesting individual item scoring...")
    query = "AI and machine learning"
    user_profile = {"interests": ["AI", "data science"], "expertise_level": "intermediate"}
    conversation_context = [
        {"user_input": "Tell me about AI", "agent_response": "AI stands for Artificial Intelligence."},
        {"user_input": "How is it used?", "agent_response": "AI is used in many applications like recommendation systems."}
    ]
    
    for item in items:
        score = scorer.score_item(item, query, user_profile, conversation_context)
        print(f"Item: {item['id']}")
        print(f"Base score: {item['score']:.4f}")
        print(f"Final relevance score: {score:.4f}")
    
    # Test scoring and ranking results
    print("\nTesting scoring and ranking results...")
    ranked_results = scorer.score_and_rank_results(items, query, user_profile, conversation_context)
    
    print(f"Ranked {len(ranked_results)} results:")
    for i, result in enumerate(ranked_results):
        print(f"{i+1}. ID: {result['id']}")
        print(f"   Relevance score: {result['relevance_score']:.4f}")
        print(f"   Score breakdown: {result['score_breakdown']}")
    
    # Test with custom weights
    print("\nTesting with custom weights...")
    custom_scorer = RelevanceScorer({
        'base_similarity': 0.3,
        'recency': 0.4,
        'user_interest_match': 0.2,
        'conversation_relevance': 0.1
    })
    
    print(f"Custom scorer weights: {custom_scorer.weights}")
    
    ranked_results = custom_scorer.score_and_rank_results(items, query, user_profile, conversation_context)
    
    print(f"Ranked {len(ranked_results)} results with custom weights:")
    for i, result in enumerate(ranked_results):
        print(f"{i+1}. ID: {result['id']}")
        print(f"   Relevance score: {result['relevance_score']:.4f}")
    
    print("RelevanceScorer test completed.")

def run_all_tests():
    """Run all advanced retrieval tests"""
    print("=== Advanced Retrieval Tests ===")
    
    test_conversation_memory()
    test_hybrid_retrieval_system()
    test_context_aware_retrieval()
    test_relevance_scorer()
    
    print("\nAll advanced retrieval tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
