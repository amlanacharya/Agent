"""
Vector Store Implementation
-------------------------
This file contains implementations of vector embeddings and a simple vector database
for semantic search and retrieval.
"""

import numpy as np
import json
import os
import time
from collections import Counter

def simple_embedding(text, dimensions=100):
    """
    Create a simple embedding vector for text.
    This is a very simplified version for demonstration purposes.
    In a real application, you would use a pre-trained model.
    
    Args:
        text (str): The text to embed
        dimensions (int): The number of dimensions for the vector
        
    Returns:
        np.ndarray: The embedding vector
    """
    # Normalize text
    text = text.lower()
    
    # Create a counter of words
    word_counts = Counter(text.split())
    
    # Use a simple hash function to map words to vector positions
    vector = np.zeros(dimensions)
    for word, count in word_counts.items():
        # Use hash of word to determine which dimensions to affect
        word_hash = abs(hash(word)) % dimensions
        # Use the count and a secondary hash to determine the value
        vector[word_hash] = count * (abs(hash(word + 'salt')) % 10 + 1) / 10
    
    # Normalize the vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector


class SimpleVectorDB:
    """
    A simple vector database for storing and retrieving text based on semantic similarity.
    
    This implementation stores vectors in memory and provides basic similarity search
    functionality. In a real application, you would use a specialized vector database
    like FAISS, Pinecone, or Chroma.
    """
    
    def __init__(self, embedding_function=None):
        """
        Initialize a simple vector database
        
        Args:
            embedding_function: Function to convert text to vectors
        """
        self.items = []  # Will store (id, text, vector, metadata) tuples
        self.embedding_function = embedding_function or simple_embedding
    
    def add_item(self, item_id, text, metadata=None):
        """
        Add an item to the database
        
        Args:
            item_id: Unique identifier for the item
            text: The text content to store
            metadata (dict, optional): Additional metadata for the item
            
        Returns:
            int: The index of the added item
        """
        vector = self.embedding_function(text)
        self.items.append((item_id, text, vector, metadata or {}))
        return len(self.items) - 1  # Return the index
    
    def add_items(self, items):
        """
        Add multiple items to the database
        
        Args:
            items: List of (item_id, text, metadata) tuples
            
        Returns:
            list: The indices of the added items
        """
        indices = []
        for item_id, text, metadata in items:
            index = self.add_item(item_id, text, metadata)
            indices.append(index)
        return indices
    
    def update_item(self, item_id, text=None, metadata=None):
        """
        Update an existing item
        
        Args:
            item_id: The ID of the item to update
            text (str, optional): New text content
            metadata (dict, optional): New or updated metadata
            
        Returns:
            bool: True if the item was updated, False if not found
        """
        for i, (id_, old_text, vector, old_metadata) in enumerate(self.items):
            if id_ == item_id:
                # Update text and vector if provided
                if text is not None:
                    new_vector = self.embedding_function(text)
                    new_text = text
                else:
                    new_vector = vector
                    new_text = old_text
                
                # Update metadata if provided
                if metadata is not None:
                    new_metadata = {**old_metadata, **metadata}
                else:
                    new_metadata = old_metadata
                
                # Replace the item
                self.items[i] = (item_id, new_text, new_vector, new_metadata)
                return True
        
        return False
    
    def get_item(self, item_id):
        """
        Get an item by ID
        
        Args:
            item_id: The ID of the item to retrieve
            
        Returns:
            tuple: (item_id, text, vector, metadata) or None if not found
        """
        for item in self.items:
            if item[0] == item_id:
                return item
        return None
    
    def delete_item(self, item_id):
        """
        Delete an item from the database
        
        Args:
            item_id: The ID of the item to delete
            
        Returns:
            bool: True if the item was deleted, False if not found
        """
        for i, item in enumerate(self.items):
            if item[0] == item_id:
                self.items.pop(i)
                return True
        return False
    
    def similarity(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Cosine similarity (between -1 and 1)
        """
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0  # Handle zero vectors
        
        return dot_product / (norm1 * norm2)
    
    def euclidean_distance(self, vector1, vector2):
        """
        Calculate Euclidean distance between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Euclidean distance (0 for identical vectors)
        """
        return np.linalg.norm(vector1 - vector2)
    
    def manhattan_distance(self, vector1, vector2):
        """
        Calculate Manhattan distance between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Manhattan distance (0 for identical vectors)
        """
        return np.sum(np.abs(vector1 - vector2))
    
    def search(self, query, top_k=3, metric="cosine"):
        """
        Search for items similar to the query
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            metric (str): Similarity metric to use ("cosine", "euclidean", "manhattan")
            
        Returns:
            list: Top k similar items with similarity scores
        """
        query_vector = self.embedding_function(query)
        
        # Calculate similarity for all items
        if metric == "cosine":
            # For cosine similarity, higher is better
            scores = [
                (item_id, text, self.similarity(query_vector, vector), metadata)
                for item_id, text, vector, metadata in self.items
            ]
            # Sort by similarity (highest first)
            scores.sort(key=lambda x: x[2], reverse=True)
        elif metric == "euclidean":
            # For distance metrics, lower is better
            scores = [
                (item_id, text, self.euclidean_distance(query_vector, vector), metadata)
                for item_id, text, vector, metadata in self.items
            ]
            # Sort by distance (lowest first)
            scores.sort(key=lambda x: x[2])
        elif metric == "manhattan":
            # For distance metrics, lower is better
            scores = [
                (item_id, text, self.manhattan_distance(query_vector, vector), metadata)
                for item_id, text, vector, metadata in self.items
            ]
            # Sort by distance (lowest first)
            scores.sort(key=lambda x: x[2])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Return top k results
        return scores[:top_k]
    
    def save(self, file_path):
        """
        Save the vector database to a file
        
        Args:
            file_path (str): Path to save the database
            
        Returns:
            bool: True if successful
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_items = [
            (id_, text, vector.tolist(), metadata)
            for id_, text, vector, metadata in self.items
        ]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_items, f)
        
        return True
    
    @classmethod
    def load(cls, file_path, embedding_function=None):
        """
        Load a vector database from a file
        
        Args:
            file_path (str): Path to load the database from
            embedding_function: Function to convert text to vectors
            
        Returns:
            SimpleVectorDB: The loaded database
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create a new instance
        db = cls(embedding_function)
        
        # Load from file
        with open(file_path, 'r') as f:
            serialized_items = json.load(f)
        
        # Convert lists back to numpy arrays
        db.items = [
            (id_, text, np.array(vector), metadata)
            for id_, text, vector, metadata in serialized_items
        ]
        
        return db
    
    def __len__(self):
        """Get the number of items in the database"""
        return len(self.items)
    
    def __str__(self):
        """String representation of the vector database"""
        return f"SimpleVectorDB(items={len(self.items)})"


class RetrievalMemory:
    """
    A retrieval-based memory system that uses vector embeddings for semantic search.
    
    This class combines a vector database with metadata to create a memory system
    that can store and retrieve information based on semantic similarity.
    """
    
    def __init__(self, storage_dir="retrieval_memory", embedding_function=None):
        """
        Initialize a retrieval-based memory system
        
        Args:
            storage_dir (str): Directory for persistent storage
            embedding_function: Function to convert text to vectors
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.db_path = os.path.join(storage_dir, "vector_db.json")
        
        # Initialize or load the vector database
        if os.path.exists(self.db_path):
            self.vector_db = SimpleVectorDB.load(self.db_path, embedding_function)
        else:
            self.vector_db = SimpleVectorDB(embedding_function)
        
        # Load or initialize the next ID counter
        self.counter_path = os.path.join(storage_dir, "counter.json")
        if os.path.exists(self.counter_path):
            with open(self.counter_path, 'r') as f:
                self.next_id = json.load(f)
        else:
            self.next_id = 0
    
    def _save_counter(self):
        """Save the ID counter to disk"""
        with open(self.counter_path, 'w') as f:
            json.dump(self.next_id, f)
    
    def store(self, text, metadata=None):
        """
        Store information in memory
        
        Args:
            text (str): The text to store
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: The ID of the stored item
        """
        # Generate a unique ID
        item_id = f"item_{self.next_id}"
        self.next_id += 1
        self._save_counter()
        
        # Add timestamp to metadata
        full_metadata = {
            "timestamp": time.time(),
            **(metadata or {})
        }
        
        # Store the item in the vector database
        self.vector_db.add_item(item_id, text, full_metadata)
        
        # Save the updated database
        self.vector_db.save(self.db_path)
        
        return item_id
    
    def retrieve(self, query, top_k=3, metric="cosine"):
        """
        Retrieve information similar to the query
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            metric (str): Similarity metric to use
            
        Returns:
            list: Top k similar items with similarity scores and metadata
        """
        results = self.vector_db.search(query, top_k, metric)
        
        # Format the results for easier consumption
        formatted_results = [
            {
                "id": item_id,
                "text": text,
                "score": score,
                "metadata": metadata
            }
            for item_id, text, score, metadata in results
        ]
        
        return formatted_results
    
    def retrieve_most_similar(self, query, metric="cosine"):
        """
        Retrieve the most similar item to the query
        
        Args:
            query (str): The search query
            metric (str): Similarity metric to use
            
        Returns:
            dict: Most similar item or None if no items
        """
        results = self.retrieve(query, top_k=1, metric=metric)
        return results[0] if results else None
    
    def update(self, item_id, text=None, metadata=None):
        """
        Update an existing memory item
        
        Args:
            item_id (str): The ID of the item to update
            text (str, optional): New text content
            metadata (dict, optional): New or updated metadata
            
        Returns:
            bool: True if the item was updated, False if not found
        """
        # Update the item
        success = self.vector_db.update_item(item_id, text, metadata)
        
        # Save the updated database if successful
        if success:
            self.vector_db.save(self.db_path)
        
        return success
    
    def delete(self, item_id):
        """
        Delete an item from memory
        
        Args:
            item_id (str): The ID of the item to delete
            
        Returns:
            bool: True if the item was deleted, False if not found
        """
        # Delete the item
        success = self.vector_db.delete_item(item_id)
        
        # Save the updated database if successful
        if success:
            self.vector_db.save(self.db_path)
        
        return success
    
    def get_all(self):
        """
        Get all items in memory
        
        Returns:
            list: All items with their metadata
        """
        return [
            {
                "id": item_id,
                "text": text,
                "metadata": metadata
            }
            for item_id, text, _, metadata in self.vector_db.items
        ]
    
    def clear(self):
        """Clear all memory"""
        self.vector_db = SimpleVectorDB(self.vector_db.embedding_function)
        self.vector_db.save(self.db_path)
        self.next_id = 0
        self._save_counter()
    
    def __len__(self):
        """Get the number of items in memory"""
        return len(self.vector_db)
    
    def __str__(self):
        """String representation of the retrieval memory"""
        return f"RetrievalMemory(items={len(self.vector_db)}, path='{self.storage_dir}')"


# Simple demonstration if run directly
if __name__ == "__main__":
    print("Vector Database Demo")
    print("-------------------")
    
    # Create a vector database
    db = SimpleVectorDB()
    
    # Add some items
    print("\nAdding items to the database...")
    db.add_item("doc1", "Artificial intelligence is the simulation of human intelligence by machines.")
    db.add_item("doc2", "Machine learning is a subset of AI that enables systems to learn from data.")
    db.add_item("doc3", "Neural networks are computing systems inspired by biological neural networks.")
    db.add_item("doc4", "Deep learning is a subset of machine learning using neural networks with many layers.")
    db.add_item("doc5", "Natural language processing helps computers understand and generate human language.")
    db.add_item("doc6", "Computer vision enables machines to interpret and make decisions based on visual data.")
    db.add_item("doc7", "Reinforcement learning is training algorithms using rewards and punishments.")
    db.add_item("doc8", "The Turing test measures a machine's ability to exhibit intelligent behavior.")
    
    print(f"Added {len(db)} items to the database.")
    
    # Search for similar items
    print("\nSearching for 'AI and machine learning'...")
    results = db.search("AI and machine learning", top_k=3)
    
    print("\nTop 3 results (cosine similarity):")
    for item_id, text, score, _ in results:
        print(f"- {item_id} (score: {score:.4f}): {text}")
    
    # Try different metrics
    print("\nSearching with Euclidean distance...")
    results = db.search("AI and machine learning", top_k=3, metric="euclidean")
    
    print("\nTop 3 results (Euclidean distance):")
    for item_id, text, score, _ in results:
        print(f"- {item_id} (distance: {score:.4f}): {text}")
    
    # Create a retrieval memory system
    print("\n\nRetrieval Memory Demo")
    print("--------------------")
    
    memory = RetrievalMemory(storage_dir="demo_retrieval_memory")
    
    # Store some information
    print("\nStoring information...")
    memory.store("Python is a high-level programming language known for its readability.", 
                {"category": "programming", "importance": "high"})
    memory.store("JavaScript is a scripting language used primarily for web development.", 
                {"category": "programming", "importance": "medium"})
    memory.store("Machine learning algorithms can improve with experience.", 
                {"category": "AI", "importance": "high"})
    memory.store("The Internet of Things connects everyday devices to the internet.", 
                {"category": "technology", "importance": "medium"})
    
    print(f"Stored {len(memory)} items in memory.")
    
    # Retrieve information
    print("\nRetrieving information about programming languages...")
    results = memory.retrieve("programming languages", top_k=2)
    
    print("\nTop 2 results:")
    for result in results:
        print(f"- {result['id']} (score: {result['score']:.4f}): {result['text']}")
        print(f"  Metadata: {result['metadata']}")
    
    # Retrieve most similar item
    print("\nMost similar item to 'artificial intelligence':")
    result = memory.retrieve_most_similar("artificial intelligence")
    if result:
        print(f"- {result['id']} (score: {result['score']:.4f}): {result['text']}")
        print(f"  Metadata: {result['metadata']}")
    
    # Clean up
    import shutil
    if os.path.exists("demo_retrieval_memory"):
        shutil.rmtree("demo_retrieval_memory")
