"""
Module 2: Memory Systems - Vector Database Exercises
---------------------------------------------------
This file contains solutions for the practice exercises from Lesson 2.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union


class PersistentVectorDB:
    """
    Exercise 1: A persistent vector database that can save and load vectors from disk.
    """
    
    def __init__(self, dimension: int = 128, storage_path: str = "vector_db_storage"):
        """
        Initialize the persistent vector database
        
        Args:
            dimension (int): Dimension of the vectors
            storage_path (str): Path to store the database files
        """
        self.dimension = dimension
        self.storage_path = storage_path
        self.items = {}  # id -> (vector, text, metadata)
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        # Load existing database if available
        self.load_database()
    
    def add_item(self, item_id: str, text: str, metadata: Dict[str, Any] = None, vector: Optional[List[float]] = None):
        """
        Add an item to the database
        
        Args:
            item_id (str): Unique identifier for the item
            text (str): Text content of the item
            metadata (dict, optional): Additional metadata for the item
            vector (list, optional): Pre-computed vector for the item
        """
        if vector is None:
            # In a real implementation, we would use an embedding model here
            # For this exercise, we'll create a simple random vector
            vector = self._create_simple_embedding(text)
        
        # Store the item
        self.items[item_id] = (vector, text, metadata or {})
        
        # Save the updated database
        self.save_database()
        
        return item_id
    
    def add_items_batch(self, items: List[Tuple[str, str, Dict[str, Any], Optional[List[float]]]]):
        """
        Add multiple items to the database in a batch
        
        Args:
            items (list): List of tuples (item_id, text, metadata, vector)
        
        Returns:
            list: List of added item IDs
        """
        added_ids = []
        
        for item_id, text, metadata, vector in items:
            if vector is None:
                vector = self._create_simple_embedding(text)
            
            self.items[item_id] = (vector, text, metadata or {})
            added_ids.append(item_id)
        
        # Save the updated database
        self.save_database()
        
        return added_ids
    
    def update_item(self, item_id: str, text: str = None, metadata: Dict[str, Any] = None, vector: List[float] = None):
        """
        Update an existing item in the database
        
        Args:
            item_id (str): ID of the item to update
            text (str, optional): New text content
            metadata (dict, optional): New metadata
            vector (list, optional): New vector
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if item_id not in self.items:
            return False
        
        current_vector, current_text, current_metadata = self.items[item_id]
        
        # Update the fields that were provided
        if text is not None:
            current_text = text
            # Recompute vector if not provided but text was updated
            if vector is None:
                current_vector = self._create_simple_embedding(text)
        
        if vector is not None:
            current_vector = vector
        
        if metadata is not None:
            # Merge the new metadata with existing metadata
            current_metadata.update(metadata)
        
        # Store the updated item
        self.items[item_id] = (current_vector, current_text, current_metadata)
        
        # Save the updated database
        self.save_database()
        
        return True
    
    def search(self, query: str, top_k: int = 5, metric: str = "cosine") -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """
        Search for items similar to the query
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            metric (str): Similarity metric to use ('cosine', 'euclidean', or 'manhattan')
            
        Returns:
            list: List of tuples (id, text, score, metadata)
        """
        if not self.items:
            return []
        
        # Create query vector
        query_vector = self._create_simple_embedding(query)
        
        # Calculate similarity scores
        scores = []
        for item_id, (vector, text, metadata) in self.items.items():
            if metric == "cosine":
                score = self._cosine_similarity(query_vector, vector)
            elif metric == "euclidean":
                score = self._euclidean_distance(query_vector, vector)
                # Convert distance to similarity score (1 / (1 + distance))
                score = 1 / (1 + score)
            elif metric == "manhattan":
                score = self._manhattan_distance(query_vector, vector)
                # Convert distance to similarity score (1 / (1 + distance))
                score = 1 / (1 + score)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append((item_id, text, score, metadata))
        
        # Sort by score (descending) and return top k
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]
    
    def recommend_metric(self, query: str, item_ids: List[str]) -> str:
        """
        Recommend the best similarity metric for a given query and set of items
        
        Args:
            query (str): The search query
            item_ids (list): List of item IDs to consider
            
        Returns:
            str: Recommended metric ('cosine', 'euclidean', or 'manhattan')
        """
        # Create query vector
        query_vector = self._create_simple_embedding(query)
        
        # Get vectors for the specified items
        item_vectors = []
        for item_id in item_ids:
            if item_id in self.items:
                vector, _, _ = self.items[item_id]
                item_vectors.append(vector)
        
        if not item_vectors:
            return "cosine"  # Default if no items found
        
        # Calculate average vector length
        avg_length = np.mean([np.linalg.norm(v) for v in item_vectors])
        
        # Calculate average sparsity (percentage of near-zero elements)
        sparsity = np.mean([np.sum(np.abs(v) < 0.01) / len(v) for v in item_vectors])
        
        # Calculate average dimensionality
        dimensionality = len(item_vectors[0])
        
        # Simple heuristic for metric recommendation:
        # - For high-dimensional sparse data, cosine is often better
        # - For dense low-dimensional data, euclidean often works well
        # - For data with outliers, manhattan can be more robust
        
        if sparsity > 0.5 and dimensionality > 50:
            return "cosine"
        elif sparsity < 0.2 and avg_length < 10:
            return "euclidean"
        else:
            return "manhattan"
    
    def save_database(self):
        """Save the database to disk"""
        # Save items dictionary
        with open(os.path.join(self.storage_path, "items.pkl"), "wb") as f:
            pickle.dump(self.items, f)
        
        # Save metadata in a more human-readable format
        metadata_dict = {
            item_id: metadata for item_id, (_, _, metadata) in self.items.items()
        }
        with open(os.path.join(self.storage_path, "metadata.json"), "w") as f:
            json.dump(metadata_dict, f, indent=2)
    
    def load_database(self):
        """Load the database from disk"""
        items_path = os.path.join(self.storage_path, "items.pkl")
        if os.path.exists(items_path):
            with open(items_path, "rb") as f:
                self.items = pickle.load(f)
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """
        Create a simple embedding for text (for demonstration purposes)
        
        In a real implementation, you would use a proper embedding model here.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list: Vector representation of the text
        """
        # This is a very simplistic embedding function for demonstration
        # It creates a deterministic but not very meaningful vector
        
        # Hash the text to get a seed
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        
        # Create a random vector
        vector = np.random.randn(self.dimension)
        
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1 (list): First vector
            vec2 (list): Second vector
            
        Returns:
            float: Cosine similarity (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate Euclidean distance between two vectors
        
        Args:
            vec1 (list): First vector
            vec2 (list): Second vector
            
        Returns:
            float: Euclidean distance
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        return np.linalg.norm(vec1 - vec2)
    
    def _manhattan_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate Manhattan distance between two vectors
        
        Args:
            vec1 (list): First vector
            vec2 (list): Second vector
            
        Returns:
            float: Manhattan distance
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        return np.sum(np.abs(vec1 - vec2))


class HybridRetrievalSystem:
    """
    Exercise 3: A hybrid retrieval system that combines keyword search with vector search
    """
    
    def __init__(self, vector_db):
        """
        Initialize the hybrid retrieval system
        
        Args:
            vector_db: Vector database for semantic search
        """
        self.vector_db = vector_db
        self.keyword_index = {}  # word -> list of item_ids
        
        # Build keyword index from existing items
        self._build_keyword_index()
    
    def add_item(self, item_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        Add an item to the retrieval system
        
        Args:
            item_id (str): Unique identifier for the item
            text (str): Text content of the item
            metadata (dict, optional): Additional metadata for the item
        """
        # Add to vector database
        self.vector_db.add_item(item_id, text, metadata)
        
        # Add to keyword index
        self._index_text(item_id, text)
    
    def search(self, query: str, top_k: int = 5, hybrid_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for items using a hybrid approach
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            hybrid_weight (float): Weight for vector search (0-1)
                                  0 = keyword only, 1 = vector only
            
        Returns:
            list: List of result dictionaries
        """
        # Get keyword search results
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Get vector search results
        vector_results = self.vector_db.search(query, top_k * 2)
        
        # Combine results
        combined_results = self._combine_results(
            keyword_results, vector_results, hybrid_weight, top_k
        )
        
        return combined_results
    
    def explain_result(self, result: Dict[str, Any], query: str) -> str:
        """
        Explain why a particular result was returned
        
        Args:
            result (dict): The result to explain
            query (str): The original query
            
        Returns:
            str: Explanation of why the result was returned
        """
        explanation = f"Result: {result['text'][:100]}...\n\n"
        
        # Explain vector similarity
        explanation += f"Semantic similarity score: {result['vector_score']:.2f}\n"
        
        # Explain keyword matches
        if result['keyword_matches']:
            explanation += f"Keyword matches: {', '.join(result['keyword_matches'])}\n"
            explanation += f"Keyword score: {result['keyword_score']:.2f}\n"
        else:
            explanation += "No direct keyword matches found.\n"
        
        # Explain combined score
        explanation += f"Combined score: {result['score']:.2f}\n"
        
        # Add metadata explanation if available
        if result['metadata']:
            explanation += "\nAdditional factors:\n"
            for key, value in result['metadata'].items():
                explanation += f"- {key}: {value}\n"
        
        return explanation
    
    def _build_keyword_index(self):
        """Build keyword index from existing items in the vector database"""
        for item_id, (_, text, _) in self.vector_db.items.items():
            self._index_text(item_id, text)
    
    def _index_text(self, item_id: str, text: str):
        """
        Index text for keyword search
        
        Args:
            item_id (str): ID of the item
            text (str): Text to index
        """
        # Tokenize and normalize text
        words = self._tokenize(text)
        
        # Add to index
        for word in words:
            if word not in self.keyword_index:
                self.keyword_index[word] = []
            
            if item_id not in self.keyword_index[word]:
                self.keyword_index[word].append(item_id)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of words
        """
        # Simple tokenization for demonstration
        # In a real system, you would use more sophisticated NLP
        text = text.lower()
        # Remove punctuation
        for char in '.,;:!?"\'()[]{}':
            text = text.replace(char, ' ')
        
        # Split into words and filter out stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about'}
        
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        
        return words
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float, List[str]]]:
        """
        Perform keyword search
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            
        Returns:
            list: List of tuples (item_id, score, matched_keywords)
        """
        # Tokenize query
        query_words = self._tokenize(query)
        
        # Find matching items
        matching_items = {}  # item_id -> (score, matched_keywords)
        
        for word in query_words:
            if word in self.keyword_index:
                for item_id in self.keyword_index[word]:
                    if item_id not in matching_items:
                        matching_items[item_id] = [0, []]
                    
                    # Increase score and add matched keyword
                    matching_items[item_id][0] += 1
                    matching_items[item_id][1].append(word)
        
        # Convert to list and normalize scores
        results = []
        max_score = len(query_words)
        
        for item_id, (score, matched_keywords) in matching_items.items():
            normalized_score = score / max_score if max_score > 0 else 0
            results.append((item_id, normalized_score, matched_keywords))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _combine_results(self, keyword_results, vector_results, hybrid_weight, top_k):
        """
        Combine keyword and vector search results
        
        Args:
            keyword_results (list): Results from keyword search
            vector_results (list): Results from vector search
            hybrid_weight (float): Weight for vector search (0-1)
            top_k (int): Number of results to return
            
        Returns:
            list: Combined and ranked results
        """
        # Create dictionaries for easier lookup
        keyword_dict = {item_id: (score, keywords) for item_id, score, keywords in keyword_results}
        vector_dict = {item_id: (score, text, metadata) for item_id, text, score, metadata in vector_results}
        
        # Combine all unique item IDs
        all_ids = set(keyword_dict.keys()) | set(vector_dict.keys())
        
        # Calculate combined scores
        combined_results = []
        
        for item_id in all_ids:
            # Get scores (default to 0 if not found)
            keyword_score, keywords = keyword_dict.get(item_id, (0, []))
            
            if item_id in vector_dict:
                vector_score, text, metadata = vector_dict[item_id]
            else:
                # If not in vector results, get info from vector DB
                if item_id in self.vector_db.items:
                    _, text, metadata = self.vector_db.items[item_id]
                    vector_score = 0
                else:
                    # This shouldn't happen, but just in case
                    continue
            
            # Calculate combined score
            combined_score = (1 - hybrid_weight) * keyword_score + hybrid_weight * vector_score
            
            # Create result object
            result = {
                'id': item_id,
                'text': text,
                'score': combined_score,
                'vector_score': vector_score,
                'keyword_score': keyword_score,
                'keyword_matches': keywords,
                'metadata': metadata
            }
            
            combined_results.append(result)
        
        # Sort by combined score and return top k
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:top_k]


# Example usage
if __name__ == "__main__":
    # Exercise 1: Persistent Vector Database
    print("Testing Persistent Vector Database...")
    db = PersistentVectorDB(dimension=64, storage_path="vector_db_test")
    
    # Add some items
    db.add_item("item1", "Machine learning is a subset of artificial intelligence")
    db.add_item("item2", "Natural language processing helps computers understand human language")
    db.add_item("item3", "Vector databases are optimized for similarity search")
    
    # Update an item
    db.update_item("item1", metadata={"category": "AI", "importance": "high"})
    
    # Batch addition
    batch_items = [
        ("item4", "Neural networks are inspired by the human brain", {"category": "AI"}, None),
        ("item5", "Embeddings represent text as vectors in a high-dimensional space", {"category": "NLP"}, None)
    ]
    db.add_items_batch(batch_items)
    
    # Search with different metrics
    print("\nCosine similarity search:")
    results = db.search("artificial intelligence", metric="cosine")
    for item_id, text, score, metadata in results:
        print(f"{item_id}: {text[:30]}... (Score: {score:.2f})")
    
    print("\nEuclidean distance search:")
    results = db.search("artificial intelligence", metric="euclidean")
    for item_id, text, score, metadata in results:
        print(f"{item_id}: {text[:30]}... (Score: {score:.2f})")
    
    print("\nManhattan distance search:")
    results = db.search("artificial intelligence", metric="manhattan")
    for item_id, text, score, metadata in results:
        print(f"{item_id}: {text[:30]}... (Score: {score:.2f})")
    
    # Recommend metric
    query = "language understanding"
    recommended_metric = db.recommend_metric(query, ["item1", "item2", "item3"])
    print(f"\nRecommended metric for '{query}': {recommended_metric}")
    
    # Exercise 3: Hybrid Retrieval System
    print("\n\nTesting Hybrid Retrieval System...")
    hybrid_system = HybridRetrievalSystem(db)
    
    # Add a new item
    hybrid_system.add_item("item6", "Language models can generate human-like text", {"category": "NLP"})
    
    # Search with different weights
    print("\nKeyword-focused search (weight=0.2):")
    results = hybrid_system.search("language understanding", hybrid_weight=0.2)
    for result in results:
        print(f"{result['id']}: {result['text'][:30]}... (Score: {result['score']:.2f})")
    
    print("\nBalanced search (weight=0.5):")
    results = hybrid_system.search("language understanding", hybrid_weight=0.5)
    for result in results:
        print(f"{result['id']}: {result['text'][:30]}... (Score: {result['score']:.2f})")
    
    print("\nVector-focused search (weight=0.8):")
    results = hybrid_system.search("language understanding", hybrid_weight=0.8)
    for result in results:
        print(f"{result['id']}: {result['text'][:30]}... (Score: {result['score']:.2f})")
    
    # Explain a result
    if results:
        print("\nExplanation for top result:")
        explanation = hybrid_system.explain_result(results[0], "language understanding")
        print(explanation)
