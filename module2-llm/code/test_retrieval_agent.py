"""
Test Script for Retrieval Agent with LLM Enhancement
-------------------------------------------------
This script demonstrates how to test the retrieval agent implementation
with LLM enhancement for better context understanding and retrieval.
"""

import os
import time
import json
from datetime import datetime

# Adjust the import path based on how you're running the script
try:
    # When running from the module2-llm/code directory
    from retrieval_agent import RetrievalAgent
    from vector_store import EnhancedVectorDB
except ImportError:
    # When running from the project root
    from module2_llm.code.retrieval_agent import RetrievalAgent
    from module2_llm.code.vector_store import EnhancedVectorDB


def test_basic_retrieval():
    """Test basic retrieval functionality"""
    print("\n=== Testing Basic Retrieval ===\n")
    
    # Create a retrieval agent
    agent = RetrievalAgent()
    
    # Add some knowledge
    print("Adding knowledge to the agent...")
    knowledge = [
        "Python is a high-level programming language known for its readability and simplicity.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.",
        "Vector databases store data as high-dimensional vectors and enable semantic search.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "Transformers are a type of neural network architecture used in NLP tasks.",
        "BERT is a transformer-based language model developed by Google.",
        "GPT (Generative Pre-trained Transformer) is a type of language model developed by OpenAI.",
        "Groq is a company that provides high-performance AI inference services.",
        "LangChain is a framework for developing applications powered by language models."
    ]
    
    agent.add_knowledge_batch(knowledge)
    print(f"Added {len(knowledge)} knowledge items")
    
    # Test basic retrieval
    print("\nTesting basic retrieval...")
    queries = [
        "What is Python?",
        "Explain machine learning",
        "How do vector databases work?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = agent.retrieve(query, top_k=2)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['text'][:100]}... (Similarity: {result['similarity']:.4f})")
    
    # Test retrieval with expansion
    print("\nTesting retrieval with query expansion...")
    query = "How do computers understand text?"
    print(f"Query: '{query}'")
    results = agent.retrieve_with_expansion(query, top_k=2)
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:100]}... (Similarity: {result['similarity']:.4f})")
        print(f"   Expanded query: {result['expanded_query']}")


def test_conversation_context():
    """Test retrieval with conversation context"""
    print("\n=== Testing Retrieval with Conversation Context ===\n")
    
    # Create a retrieval agent
    agent = RetrievalAgent()
    
    # Add some knowledge
    print("Adding knowledge to the agent...")
    knowledge = [
        "Python is a high-level programming language known for its readability and simplicity.",
        "Python is widely used in data science, machine learning, web development, and automation.",
        "Python was created by Guido van Rossum and first released in 1991.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.",
        "TensorFlow is an open-source machine learning framework developed by Google.",
        "PyTorch is an open-source machine learning framework developed by Facebook.",
        "Scikit-learn is a machine learning library for Python that provides simple and efficient tools for data analysis and modeling."
    ]
    
    agent.add_knowledge_batch(knowledge)
    print(f"Added {len(knowledge)} knowledge items")
    
    # Simulate a conversation
    print("\nSimulating a conversation...")
    
    # First question
    print("\nUser: What is Python?")
    response = agent.answer_question("What is Python?")
    print(f"Agent: {response}")
    
    # Follow-up question (should use conversation context)
    print("\nUser: What is it used for?")
    response = agent.answer_question("What is it used for?")
    print(f"Agent: {response}")
    
    # Another follow-up
    print("\nUser: Which libraries are popular for machine learning?")
    response = agent.answer_question("Which libraries are popular for machine learning?")
    print(f"Agent: {response}")
    
    # Test retrieval with context
    print("\nTesting retrieval with context...")
    results = agent.retrieve_with_context("What are the best tools?", top_k=2)
    print("Query: 'What are the best tools?'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:100]}... (Similarity: {result['similarity']:.4f})")
        print(f"   Context info: {result['context_info']}")


def test_retrieval_explanation():
    """Test retrieval explanation"""
    print("\n=== Testing Retrieval Explanation ===\n")
    
    # Create a retrieval agent
    agent = RetrievalAgent()
    
    # Add some knowledge
    print("Adding knowledge to the agent...")
    knowledge = [
        "Python is a high-level programming language known for its readability and simplicity.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.",
        "Vector databases store data as high-dimensional vectors and enable semantic search.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "Transformers are a type of neural network architecture used in NLP tasks.",
        "BERT is a transformer-based language model developed by Google.",
        "GPT (Generative Pre-trained Transformer) is a type of language model developed by OpenAI.",
        "Groq is a company that provides high-performance AI inference services.",
        "LangChain is a framework for developing applications powered by language models."
    ]
    
    agent.add_knowledge_batch(knowledge)
    print(f"Added {len(knowledge)} knowledge items")
    
    # Simulate a conversation to build context
    print("\nBuilding conversation context...")
    agent.answer_question("Tell me about AI")
    agent.answer_question("How do language models work?")
    
    # Test retrieval explanation
    print("\nTesting retrieval explanation...")
    query = "What are transformers in AI?"
    print(f"Query: '{query}'")
    
    explanation = agent.explain_retrieval(query)
    
    print("\nOriginal query:", explanation['original_query'])
    print("Enhanced query:", explanation['enhanced_query'])
    print("Expanded queries:")
    for q in explanation['expanded_queries']:
        print(f"- {q}")
    
    print("\nTop results:")
    for i, result in enumerate(explanation['top_results']):
        print(f"{i+1}. {result['text'][:50]}... (Similarity: {result['similarity']:.4f})")
    
    print("\nExplanation:")
    print(explanation['explanation'])


def test_key_information_extraction():
    """Test key information extraction"""
    print("\n=== Testing Key Information Extraction ===\n")
    
    # Create a retrieval agent
    agent = RetrievalAgent()
    
    # Test key information extraction
    print("Testing key information extraction...")
    texts = [
        "I'm looking for information about machine learning and its applications in healthcare.",
        "Can you tell me how Python is used in data science projects?",
        "I'm having trouble understanding how transformers work in natural language processing."
    ]
    
    for text in texts:
        print(f"\nText: '{text}'")
        processed = agent.process_user_input(text)
        print("Extracted key information:")
        print(json.dumps(processed['key_information'], indent=2))


if __name__ == "__main__":
    print("Testing Retrieval Agent with LLM Enhancement")
    print("=" * 50)
    print("Note: These tests require a valid Groq API key.")
    print("If you haven't set up your API key, please do so before running these tests.")
    print("=" * 50)
    
    try:
        # Run individual tests
        test_basic_retrieval()
        test_conversation_context()
        test_retrieval_explanation()
        test_key_information_extraction()
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Make sure your Groq API key is set up correctly.")
