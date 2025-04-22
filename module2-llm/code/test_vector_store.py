"""
Test Script for Vector Database with Real Embeddings
------------------------------------------------
This script demonstrates how to test the vector database implementation
with real embeddings from the Groq API.
"""

import os
import time
import json
import numpy as np

# Adjust the import path based on how you're running the script
try:
    # When running from the module2-llm/code directory
    from vector_store import SimpleVectorDB, EnhancedVectorDB
except ImportError:
    # When running from the project root
    from module2_llm.code.vector_store import SimpleVectorDB, EnhancedVectorDB


def test_simple_vector_db():
    """Test basic vector database functionality"""
    print("\n=== Testing Simple Vector Database ===\n")
    
    # Create a vector database
    db = SimpleVectorDB()
    
    # Add some items
    print("Adding items to the database...")
    items = [
        "Python is a high-level programming language known for its readability and simplicity.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.",
        "Vector databases store data as high-dimensional vectors and enable semantic search.",
        "Embeddings are numerical representations of text that capture semantic meaning."
    ]
    
    item_ids = []
    for item in items:
        item_id = db.add(item)
        item_ids.append(item_id)
        print(f"Added: {item[:50]}... (ID: {item_id})")
    
    # Test search functionality
    print("\nTesting search functionality...")
    queries = [
        "How do computers understand language?",
        "What is machine learning?",
        "Tell me about vector databases"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = db.search(query, top_k=2)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['text'][:50]}... (Similarity: {result['similarity']:.4f})")
    
    # Test update functionality
    print("\nTesting update functionality...")
    if item_ids:
        item_id = item_ids[0]
        original_item = db.get_item(item_id)
        print(f"Original item: {original_item['text'][:50]}...")
        
        updated_text = "Python is an interpreted, high-level programming language with dynamic semantics."
        db.update(item_id, text=updated_text)
        
        updated_item = db.get_item(item_id)
        print(f"Updated item: {updated_item['text'][:50]}...")
    
    # Test delete functionality
    print("\nTesting delete functionality...")
    if item_ids:
        item_id = item_ids[-1]
        print(f"Deleting item with ID: {item_id}")
        db.delete(item_id)
        print(f"Items remaining: {len(db.get_all())}")
    
    # Test save and load functionality
    print("\nTesting save and load functionality...")
    test_file = "test_vector_db.json"
    db.save(test_file)
    print(f"Database saved to {test_file}")
    
    loaded_db = SimpleVectorDB.load(test_file)
    print(f"Database loaded with {len(loaded_db.get_all())} items")
    
    # Clean up test file
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"Removed test file: {test_file}")


def test_batch_operations():
    """Test batch operations"""
    print("\n=== Testing Batch Operations ===\n")
    
    # Create a vector database
    db = SimpleVectorDB()
    
    # Test batch add
    print("Testing batch add...")
    texts = [
        "The sky is blue.",
        "Grass is green.",
        "The sun is yellow.",
        "Roses are red.",
        "Violets are blue."
    ]
    
    metadatas = [
        {"category": "nature", "confidence": 0.9},
        {"category": "nature", "confidence": 0.8},
        {"category": "astronomy", "confidence": 0.95},
        {"category": "flowers", "confidence": 0.85},
        {"category": "flowers", "confidence": 0.75}
    ]
    
    item_ids = db.add_batch(texts, metadatas)
    print(f"Added {len(item_ids)} items in batch")
    
    # Test filtering
    print("\nTesting search with filtering...")
    
    def nature_filter(item):
        return item.get('metadata', {}).get('category') == 'nature'
    
    results = db.search("natural elements", top_k=5, filter_func=nature_filter)
    print(f"Found {len(results)} nature-related results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text']} (Category: {result['metadata']['category']})")


def test_enhanced_vector_db():
    """Test enhanced vector database functionality"""
    print("\n=== Testing Enhanced Vector Database ===\n")
    
    # Create an enhanced vector database
    db = EnhancedVectorDB()
    
    # Add some items
    print("Adding items to the database...")
    items = [
        "Python is a high-level programming language known for its readability and simplicity.",
        "JavaScript is a programming language commonly used for web development.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning is a type of machine learning based on artificial neural networks.",
        "Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.",
        "Computer vision is a field of AI that enables computers to interpret and understand visual information.",
        "Vector databases store data as high-dimensional vectors and enable semantic search.",
        "Embeddings are numerical representations of text that capture semantic meaning.",
        "Transformers are a type of neural network architecture used in NLP tasks.",
        "BERT is a transformer-based language model developed by Google."
    ]
    
    for item in items:
        db.add(item)
    print(f"Added {len(items)} items")
    
    # Test query expansion
    print("\nTesting query expansion...")
    query = "How do AI systems understand text?"
    expanded_queries = db.expand_query(query)
    
    print(f"Original query: '{query}'")
    print("Expanded queries:")
    for i, expanded in enumerate(expanded_queries):
        print(f"{i+1}. {expanded}")
    
    # Test search with expansion
    print("\nTesting search with query expansion...")
    results = db.search_with_expansion(query, top_k=3)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:50]}... (Similarity: {result['similarity']:.4f})")
        print(f"   Expanded query: {result['expanded_query']}")
    
    # Test clustering
    try:
        print("\nTesting clustering...")
        clusters = db.cluster_items(num_clusters=3)
        
        print(f"Created {len(clusters)} clusters:")
        for cluster_id, cluster_items in clusters.items():
            print(f"Cluster {cluster_id}: {len(cluster_items)} items")
            for item in cluster_items[:2]:  # Show first 2 items in each cluster
                print(f"  - {item['text'][:50]}...")
        
        # Test cluster labeling
        print("\nTesting cluster labeling...")
        labels = db.get_cluster_labels(clusters)
        
        print("Cluster labels:")
        for cluster_id, label in labels.items():
            print(f"Cluster {cluster_id}: {label}")
    except Exception as e:
        print(f"Clustering tests skipped: {e}")


if __name__ == "__main__":
    print("Testing Vector Database with Real Embeddings")
    print("=" * 50)
    print("Note: These tests require a valid Groq API key.")
    print("If you haven't set up your API key, please do so before running these tests.")
    print("=" * 50)
    
    try:
        # Run individual tests
        test_simple_vector_db()
        test_batch_operations()
        test_enhanced_vector_db()
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Make sure your Groq API key is set up correctly.")
