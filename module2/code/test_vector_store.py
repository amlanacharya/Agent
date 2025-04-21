"""
Test Script for Vector Store
--------------------------
This script demonstrates and tests the vector database and retrieval memory
implementations in vector_store.py.
"""

import os
import shutil
import numpy as np
from vector_store import simple_embedding, SimpleVectorDB, RetrievalMemory

def test_simple_embedding():
    """Test the simple embedding function"""
    print("\n=== Testing Simple Embedding ===")
    
    # Test with simple text
    text1 = "This is a test sentence."
    vector1 = simple_embedding(text1)
    print(f"Embedding for '{text1}':")
    print(f"- Shape: {vector1.shape}")
    print(f"- Norm: {np.linalg.norm(vector1):.4f}")
    print(f"- First 5 values: {vector1[:5]}")
    
    # Test with similar text
    text2 = "This is a similar test sentence."
    vector2 = simple_embedding(text2)
    
    # Test with different text
    text3 = "Completely different content about artificial intelligence."
    vector3 = simple_embedding(text3)
    
    # Calculate similarities
    similarity_1_2 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    similarity_1_3 = np.dot(vector1, vector3) / (np.linalg.norm(vector1) * np.linalg.norm(vector3))
    
    print(f"\nSimilarity between similar texts: {similarity_1_2:.4f}")
    print(f"Similarity between different texts: {similarity_1_3:.4f}")
    
    print("Simple embedding test completed.")

def test_simple_vector_db():
    """Test the SimpleVectorDB class"""
    print("\n=== Testing Simple Vector DB ===")
    
    # Create a vector database
    db = SimpleVectorDB()
    print("Initial state:", db)
    
    # Add items
    print("\nAdding items...")
    db.add_item("doc1", "Artificial intelligence is the simulation of human intelligence by machines.")
    db.add_item("doc2", "Machine learning is a subset of AI that enables systems to learn from data.")
    db.add_item("doc3", "Neural networks are computing systems inspired by biological neural networks.")
    
    print("After adding items:", db)
    
    # Add items with metadata
    print("\nAdding items with metadata...")
    db.add_item("doc4", "Python is a programming language.", {"type": "language", "level": "beginner"})
    db.add_item("doc5", "JavaScript is used for web development.", {"type": "language", "level": "intermediate"})
    
    print("After adding more items:", db)
    
    # Test search with different metrics
    print("\nTesting search with different metrics...")
    
    query = "AI and machine learning"
    
    print(f"\nSearch query: '{query}'")
    
    print("\nCosine similarity results:")
    for item_id, text, score, metadata in db.search(query, top_k=2, metric="cosine"):
        print(f"- {item_id} (score: {score:.4f}): {text}")
        if metadata:
            print(f"  Metadata: {metadata}")
    
    print("\nEuclidean distance results:")
    for item_id, text, score, metadata in db.search(query, top_k=2, metric="euclidean"):
        print(f"- {item_id} (distance: {score:.4f}): {text}")
        if metadata:
            print(f"  Metadata: {metadata}")
    
    print("\nManhattan distance results:")
    for item_id, text, score, metadata in db.search(query, top_k=2, metric="manhattan"):
        print(f"- {item_id} (distance: {score:.4f}): {text}")
        if metadata:
            print(f"  Metadata: {metadata}")
    
    # Test update
    print("\nTesting item update...")
    
    print("Before update:")
    item = db.get_item("doc4")
    print(f"- {item[0]}: {item[1]}")
    print(f"  Metadata: {item[3]}")
    
    db.update_item("doc4", "Python is a high-level programming language.", {"level": "advanced"})
    
    print("After update:")
    item = db.get_item("doc4")
    print(f"- {item[0]}: {item[1]}")
    print(f"  Metadata: {item[3]}")
    
    # Test delete
    print("\nTesting item deletion...")
    print(f"Number of items before deletion: {len(db)}")
    db.delete_item("doc5")
    print(f"Number of items after deletion: {len(db)}")
    
    # Test save and load
    print("\nTesting save and load...")
    
    test_file = "test_vector_db.json"
    
    # Save the database
    db.save(test_file)
    print(f"Database saved to {test_file}")
    
    # Load the database
    loaded_db = SimpleVectorDB.load(test_file)
    print(f"Loaded database with {len(loaded_db)} items")
    
    # Verify the loaded database
    print("\nVerifying loaded database...")
    query = "AI and machine learning"
    
    print(f"Search query: '{query}'")
    print("Results from loaded database:")
    for item_id, text, score, metadata in loaded_db.search(query, top_k=2):
        print(f"- {item_id} (score: {score:.4f}): {text}")
        if metadata:
            print(f"  Metadata: {metadata}")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("Simple vector DB test completed.")

def test_retrieval_memory():
    """Test the RetrievalMemory class"""
    print("\n=== Testing Retrieval Memory ===")
    
    # Create a test directory
    test_dir = "test_retrieval_memory"
    
    # Remove the directory if it exists
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Create retrieval memory
    memory = RetrievalMemory(storage_dir=test_dir)
    print("Initial state:", memory)
    
    # Store information
    print("\nStoring information...")
    
    id1 = memory.store("Artificial intelligence is the simulation of human intelligence by machines.",
                      {"category": "AI", "importance": "high"})
    
    id2 = memory.store("Machine learning is a subset of AI that enables systems to learn from data.",
                      {"category": "AI", "importance": "high"})
    
    id3 = memory.store("Python is a high-level programming language known for its readability.",
                      {"category": "programming", "importance": "medium"})
    
    print(f"Stored 3 items with IDs: {id1}, {id2}, {id3}")
    print("After storing:", memory)
    
    # Retrieve information
    print("\nRetrieving information...")
    
    query = "AI and machine learning"
    print(f"Search query: '{query}'")
    
    results = memory.retrieve(query, top_k=2)
    print("\nTop 2 results:")
    for result in results:
        print(f"- {result['id']} (score: {result['score']:.4f}): {result['text']}")
        print(f"  Metadata: {result['metadata']}")
    
    # Test most similar
    print("\nTesting retrieve_most_similar...")
    
    query = "programming languages"
    print(f"Search query: '{query}'")
    
    result = memory.retrieve_most_similar(query)
    print("Most similar result:")
    print(f"- {result['id']} (score: {result['score']:.4f}): {result['text']}")
    print(f"  Metadata: {result['metadata']}")
    
    # Test update
    print("\nTesting update...")
    
    print(f"Updating item {id3}...")
    memory.update(id3, "Python is an interpreted, high-level programming language.", 
                 {"importance": "high"})
    
    # Verify update
    result = memory.retrieve_most_similar("Python programming")
    print("After update:")
    print(f"- {result['id']} (score: {result['score']:.4f}): {result['text']}")
    print(f"  Metadata: {result['metadata']}")
    
    # Test persistence
    print("\nTesting persistence...")
    
    # Create a new instance that should load from disk
    print("Creating new instance to test loading from disk...")
    new_memory = RetrievalMemory(storage_dir=test_dir)
    print(f"Loaded memory with {len(new_memory)} items")
    
    # Verify loaded data
    results = new_memory.retrieve("Python programming", top_k=1)
    print("Result from loaded memory:")
    print(f"- {results[0]['id']} (score: {results[0]['score']:.4f}): {results[0]['text']}")
    print(f"  Metadata: {results[0]['metadata']}")
    
    # Test delete
    print("\nTesting delete...")
    
    print(f"Deleting item {id2}...")
    memory.delete(id2)
    print(f"Number of items after deletion: {len(memory)}")
    
    # Test get_all
    print("\nTesting get_all...")
    
    all_items = memory.get_all()
    print(f"All items ({len(all_items)}):")
    for item in all_items:
        print(f"- {item['id']}: {item['text']}")
        print(f"  Metadata: {item['metadata']}")
    
    # Test clear
    print("\nTesting clear...")
    
    memory.clear()
    print(f"Number of items after clearing: {len(memory)}")
    
    # Clean up
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    print("Retrieval memory test completed.")

def run_all_tests():
    """Run all vector store tests"""
    print("=== Vector Store Tests ===")
    
    test_simple_embedding()
    test_simple_vector_db()
    test_retrieval_memory()
    
    print("\nAll vector store tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()
