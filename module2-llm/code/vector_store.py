"""
Vector Database Implementation with Real Embeddings
------------------------------------------------
This file contains a simple vector database implementation that uses
real embeddings from the Groq API for semantic search.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import uuid

# Import our Groq client
try:
    # When running from the module2-llm/code directory
    from groq_client import GroqClient
except ImportError:
    # When running from the project root
    from module2_llm.code.groq_client import GroqClient


class SimpleVectorDB:
    """
    A simple vector database implementation with real embeddings from Groq API.
    """
    
    def __init__(self, embedding_function: Optional[Callable] = None):
        """
        Initialize the vector database
        
        Args:
            embedding_function (callable, optional): Function to convert text to vectors.
                                                    If None, uses Groq embeddings.
        """
        self.items = []
        self.groq_client = GroqClient()
        self.embedding_function = embedding_function or self._get_groq_embedding
    
    def _get_groq_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for text using the Groq API
        
        Args:
            text (str): Text to get embeddings for
            
        Returns:
            list: Embedding vector
        """
        try:
            embeddings = self.groq_client.get_embeddings(text)
            return embeddings[0]  # Return the first (and only) embedding
        except Exception as e:
            # Fallback to a simple hash-based embedding if Groq API fails
            print(f"Warning: Groq embedding failed, using fallback embedding. Error: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        Create a fallback embedding based on hash values
        
        Args:
            text (str): Text to create embedding for
            dim (int): Dimension of the embedding vector
            
        Returns:
            list: Embedding vector
        """
        # Create a simple hash-based embedding (not for production use)
        import hashlib
        
        # Initialize a vector of zeros
        vector = [0.0] * dim
        
        # Use words to influence different dimensions
        words = text.lower().split()
        for i, word in enumerate(words):
            # Hash the word
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            
            # Use the hash to set values in the vector
            for j in range(min(10, len(word))):
                idx = (hash_value + j) % dim
                vector[idx] = 0.1 * ((hash_value % 20) - 10) + vector[idx]
        
        # Normalize the vector
        magnitude = sum(x**2 for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
    
    def add(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add an item to the vector database
        
        Args:
            text (str): The text to add
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: ID of the added item
        """
        item_id = str(uuid.uuid4())
        
        # Get embedding for the text
        embedding = self.embedding_function(text)
        
        # Add the item
        self.items.append({
            'id': item_id,
            'text': text,
            'embedding': embedding,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
        
        return item_id
    
    def add_batch(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add multiple items to the vector database
        
        Args:
            texts (list): List of texts to add
            metadatas (list, optional): List of metadata dictionaries
            
        Returns:
            list: IDs of the added items
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Get embeddings for all texts
        try:
            # Try to get embeddings in batch
            all_embeddings = self.groq_client.get_embeddings(texts)
        except Exception:
            # Fall back to getting embeddings one by one
            all_embeddings = [self.embedding_function(text) for text in texts]
        
        # Add all items
        item_ids = []
        for text, embedding, metadata in zip(texts, all_embeddings, metadatas):
            item_id = str(uuid.uuid4())
            self.items.append({
                'id': item_id,
                'text': text,
                'embedding': embedding,
                'metadata': metadata,
                'timestamp': time.time()
            })
            item_ids.append(item_id)
        
        return item_ids
    
    def search(self, query: str, top_k: int = 5, filter_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Search the vector database for items similar to the query
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            filter_func (callable, optional): Function to filter results
            
        Returns:
            list: Top k most similar items
        """
        if not self.items:
            return []
        
        # Get embedding for the query
        query_embedding = self.embedding_function(query)
        
        # Calculate similarity scores
        results = []
        for item in self.items:
            # Apply filter if provided
            if filter_func and not filter_func(item):
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, item['embedding'])
            
            # Add to results
            results.append({
                'id': item['id'],
                'text': item['text'],
                'metadata': item['metadata'],
                'similarity': similarity
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1 (list): First vector
            vec2 (list): Second vector
            
        Returns:
            float: Cosine similarity (-1 to 1)
        """
        # Convert to numpy arrays for efficient calculation
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        if magnitude1 > 0 and magnitude2 > 0:
            return dot_product / (magnitude1 * magnitude2)
        else:
            return 0.0
    
    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an item by ID
        
        Args:
            item_id (str): ID of the item to get
            
        Returns:
            dict or None: The item, or None if not found
        """
        for item in self.items:
            if item['id'] == item_id:
                return item
        return None
    
    def update(self, item_id: str, text: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update an item in the vector database
        
        Args:
            item_id (str): ID of the item to update
            text (str, optional): New text for the item
            metadata (dict, optional): New or updated metadata
            
        Returns:
            bool: True if the item was updated, False otherwise
        """
        for i, item in enumerate(self.items):
            if item['id'] == item_id:
                # Update text and embedding if provided
                if text is not None:
                    self.items[i]['text'] = text
                    self.items[i]['embedding'] = self.embedding_function(text)
                
                # Update metadata if provided
                if metadata is not None:
                    if isinstance(metadata, dict):
                        # Merge with existing metadata
                        self.items[i]['metadata'].update(metadata)
                    else:
                        # Replace metadata
                        self.items[i]['metadata'] = metadata
                
                # Update timestamp
                self.items[i]['timestamp'] = time.time()
                
                return True
        
        return False
    
    def delete(self, item_id: str) -> bool:
        """
        Delete an item from the vector database
        
        Args:
            item_id (str): ID of the item to delete
            
        Returns:
            bool: True if the item was deleted, False otherwise
        """
        for i, item in enumerate(self.items):
            if item['id'] == item_id:
                del self.items[i]
                return True
        return False
    
    def save(self, file_path: str) -> None:
        """
        Save the vector database to a file
        
        Args:
            file_path (str): Path to save the database to
        """
        with open(file_path, 'w') as f:
            json.dump({
                'items': self.items,
                'timestamp': time.time()
            }, f)
    
    @classmethod
    def load(cls, file_path: str, embedding_function: Optional[Callable] = None) -> 'SimpleVectorDB':
        """
        Load a vector database from a file
        
        Args:
            file_path (str): Path to load the database from
            embedding_function (callable, optional): Function to convert text to vectors
            
        Returns:
            SimpleVectorDB: The loaded vector database
        """
        db = cls(embedding_function)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                db.items = data['items']
        
        return db
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all items in the vector database
        
        Returns:
            list: All items in the database
        """
        return [{
            'id': item['id'],
            'text': item['text'],
            'metadata': item['metadata'],
            'timestamp': item['timestamp']
        } for item in self.items]
    
    def clear(self) -> None:
        """Clear the vector database"""
        self.items = []


class EnhancedVectorDB(SimpleVectorDB):
    """
    Enhanced vector database with additional features like
    semantic clustering and query expansion.
    """
    
    def __init__(self, embedding_function: Optional[Callable] = None):
        """
        Initialize the enhanced vector database
        
        Args:
            embedding_function (callable, optional): Function to convert text to vectors
        """
        super().__init__(embedding_function)
        self.groq_client = GroqClient()
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query using LLM to improve search results
        
        Args:
            query (str): The original query
            
        Returns:
            list: List of expanded queries
        """
        prompt = f"""
        Generate 3 alternative phrasings of the following search query to improve search results.
        Make sure to preserve the original meaning but use different words and phrasings.
        
        Original query: "{query}"
        
        Return only the alternative phrasings, one per line, without numbering or additional text.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=150)
            expanded = self.groq_client.extract_text_from_response(response)
            
            # Parse the response
            expansions = [line.strip() for line in expanded.split('\n') if line.strip()]
            
            # Add the original query
            all_queries = [query] + expansions
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in all_queries:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique_queries.append(q)
            
            return unique_queries
        except Exception as e:
            # Return just the original query if expansion fails
            print(f"Query expansion failed: {e}")
            return [query]
    
    def search_with_expansion(self, query: str, top_k: int = 5, filter_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Search with query expansion for better recall
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            filter_func (callable, optional): Function to filter results
            
        Returns:
            list: Top k most similar items
        """
        # Expand the query
        expanded_queries = self.expand_query(query)
        
        # Search with each expanded query
        all_results = []
        for expanded_query in expanded_queries:
            results = self.search(expanded_query, top_k=top_k, filter_func=filter_func)
            for result in results:
                result['expanded_query'] = expanded_query
            all_results.extend(results)
        
        # Remove duplicates (same ID)
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Sort by similarity
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top k results
        return unique_results[:top_k]
    
    def cluster_items(self, num_clusters: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster items in the vector database
        
        Args:
            num_clusters (int): Number of clusters to create
            
        Returns:
            dict: Clusters of items
        """
        if not self.items or len(self.items) < num_clusters:
            return {0: self.items}
        
        try:
            # Import sklearn for clustering
            from sklearn.cluster import KMeans
            
            # Extract embeddings
            embeddings = np.array([item['embedding'] for item in self.items])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(num_clusters, len(self.items)), random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Group items by cluster
            clustered_items = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_items:
                    clustered_items[cluster_id] = []
                clustered_items[cluster_id].append(self.items[i])
            
            return clustered_items
        except Exception as e:
            # Return all items in a single cluster if clustering fails
            print(f"Clustering failed: {e}")
            return {0: self.items}
    
    def get_cluster_labels(self, clusters: Dict[int, List[Dict[str, Any]]]) -> Dict[int, str]:
        """
        Generate labels for clusters using LLM
        
        Args:
            clusters (dict): Clusters of items
            
        Returns:
            dict: Labels for each cluster
        """
        labels = {}
        
        for cluster_id, items in clusters.items():
            # Get text from items
            texts = [item['text'] for item in items[:5]]  # Use up to 5 items per cluster
            
            # Create prompt
            prompt = f"""
            Generate a short, descriptive label (3-5 words) for a cluster of documents with the following content:
            
            {texts}
            
            Return only the label, without quotes or additional text.
            """
            
            try:
                response = self.groq_client.generate_text(prompt, max_tokens=50)
                label = self.groq_client.extract_text_from_response(response).strip()
                labels[cluster_id] = label
            except Exception:
                # Use a default label if LLM fails
                labels[cluster_id] = f"Cluster {cluster_id}"
        
        return labels


# Example usage
if __name__ == "__main__":
    # Create a vector database
    db = SimpleVectorDB()
    
    # Add some items
    db.add("Python is a high-level programming language known for its readability and simplicity.")
    db.add("Machine learning is a subset of artificial intelligence that enables systems to learn from data.")
    db.add("Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.")
    db.add("Vector databases store data as high-dimensional vectors and enable semantic search.")
    db.add("Embeddings are numerical representations of text that capture semantic meaning.")
    
    # Search for similar items
    query = "How do computers understand language?"
    results = db.search(query, top_k=2)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text']} (Similarity: {result['similarity']:.4f})")
    
    # Create an enhanced vector database
    enhanced_db = EnhancedVectorDB()
    
    # Add the same items
    enhanced_db.add("Python is a high-level programming language known for its readability and simplicity.")
    enhanced_db.add("Machine learning is a subset of artificial intelligence that enables systems to learn from data.")
    enhanced_db.add("Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.")
    enhanced_db.add("Vector databases store data as high-dimensional vectors and enable semantic search.")
    enhanced_db.add("Embeddings are numerical representations of text that capture semantic meaning.")
    
    # Search with query expansion
    expanded_results = enhanced_db.search_with_expansion(query, top_k=2)
    
    print("\nSearch with query expansion:")
    print(f"Query: '{query}'")
    print(f"Found {len(expanded_results)} results:")
    for i, result in enumerate(expanded_results):
        print(f"{i+1}. {result['text']} (Similarity: {result['similarity']:.4f})")
        print(f"   Expanded query: {result['expanded_query']}")
