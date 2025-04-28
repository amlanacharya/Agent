"""
Embedding Pipelines
------------------
This module provides embedding generation pipelines for different content types.
It supports multiple embedding models, preprocessing, batching, and caching.
"""

import os
import json
import time
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence_transformers not available. SentenceTransformerEmbeddings will use fallback method.")

# Try to import requests for API calls
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False
    logger.warning("requests not available. API-based embeddings will use fallback method.")

# -----------------------------------------------------------------------------
# Base Embedding Classes
# -----------------------------------------------------------------------------

class BaseEmbeddings(ABC):
    """Abstract base class for all embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer representing the embedding dimension
        """
        # Default implementation - override in subclasses if known in advance
        sample_text = "This is a sample text to determine embedding dimension."
        sample_embedding = self.embed_text(sample_text)
        return len(sample_embedding)
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: The vector to normalize
            
        Returns:
            Normalized vector
        """
        # Convert to numpy array for efficient operations
        np_vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(np_vector)
        
        # Avoid division by zero
        if norm > 0:
            normalized = np_vector / norm
            return normalized.tolist()
        return vector


class HashEmbeddings(BaseEmbeddings):
    """
    Simple hash-based embeddings for fallback when other methods are unavailable.
    Not for production use - provides deterministic but not semantic embeddings.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the hash embeddings.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate a hash-based embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Initialize a vector of zeros
        vector = [0.0] * self.dimension
        
        # Use words to influence different dimensions
        words = text.lower().split()
        for i, word in enumerate(words):
            # Hash the word
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            
            # Use the hash to set values in the vector
            for j in range(min(10, len(word))):
                idx = (hash_value + j) % self.dimension
                vector[idx] = 0.1 * ((hash_value % 20) - 10) + vector[idx]
        
        # Normalize the vector
        return self.normalize_vector(vector)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate hash-based embeddings for multiple documents.
        
        Args:
            documents: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.embed_text(doc) for doc in documents]
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer representing the embedding dimension
        """
        return self.dimension


# -----------------------------------------------------------------------------
# Embedding Model Implementations
# -----------------------------------------------------------------------------

class SentenceTransformerEmbeddings(BaseEmbeddings):
    """
    Embeddings using the sentence-transformers library.
    Provides high-quality semantic embeddings for text.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer embeddings.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.fallback = HashEmbeddings()
        
        # Try to load the model
        if HAVE_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self._dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {e}")
                self.model = None
        else:
            self.model = None
            logger.warning("SentenceTransformer not available, using fallback embeddings")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if self.model is not None:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Error generating embedding with SentenceTransformer: {e}")
                return self.fallback.embed_text(text)
        else:
            return self.fallback.embed_text(text)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.model is not None:
            try:
                embeddings = self.model.encode(documents, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Error generating embeddings with SentenceTransformer: {e}")
                return self.fallback.embed_documents(documents)
        else:
            return self.fallback.embed_documents(documents)
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer representing the embedding dimension
        """
        if hasattr(self, '_dimension'):
            return self._dimension
        return super().get_dimension()


class OpenAIEmbeddings(BaseEmbeddings):
    """
    Embeddings using OpenAI-compatible API (including Groq).
    Provides high-quality embeddings through API calls.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-ada-002",
        api_base: str = "https://api.openai.com/v1",
        api_type: str = "openai"
    ):
        """
        Initialize the OpenAI embeddings.
        
        Args:
            api_key: API key for OpenAI or Groq
            model_name: Name of the embedding model to use
            api_base: Base URL for API calls
            api_type: Type of API ('openai' or 'groq')
        """
        self.model_name = model_name
        self.api_base = api_base
        self.api_type = api_type
        
        # Set API key from args or environment
        if api_key is None:
            if api_type == "groq":
                self.api_key = os.environ.get("GROQ_API_KEY")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            self.api_key = api_key
        
        # Create fallback for when API is unavailable
        self.fallback = HashEmbeddings()
        
        # Check if we can make API calls
        self.can_use_api = HAVE_REQUESTS and self.api_key is not None
        if not self.can_use_api:
            logger.warning(f"API-based embeddings unavailable. Using fallback method.")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using API.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.can_use_api:
            return self.fallback.embed_text(text)
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "input": text
            }
            
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                logger.warning(f"API error: {response.status_code} - {response.text}")
                return self.fallback.embed_text(text)
                
        except Exception as e:
            logger.warning(f"Error calling embedding API: {e}")
            return self.fallback.embed_text(text)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents using API.
        
        Args:
            documents: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.can_use_api:
            return self.fallback.embed_documents(documents)
        
        # For small batches, we can send them all at once
        if len(documents) <= 20:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "model": self.model_name,
                    "input": documents
                }
                
                response = requests.post(
                    f"{self.api_base}/embeddings",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Sort by index to ensure correct order
                    embeddings = sorted(result["data"], key=lambda x: x["index"])
                    return [item["embedding"] for item in embeddings]
                else:
                    logger.warning(f"API error: {response.status_code} - {response.text}")
                    return self.fallback.embed_documents(documents)
                    
            except Exception as e:
                logger.warning(f"Error calling embedding API: {e}")
                return self.fallback.embed_documents(documents)
        
        # For larger batches, process in chunks
        else:
            all_embeddings = []
            batch_size = 20  # API typically limits batch size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_embeddings = self.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings


# -----------------------------------------------------------------------------
# Embedding Pipeline
# -----------------------------------------------------------------------------

class EmbeddingPipeline:
    """
    A robust pipeline for generating and managing embeddings.
    Includes preprocessing, batching, caching, and optimization.
    """
    
    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddings] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        use_preprocessing: bool = True
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            embedding_model: The embedding model to use
            batch_size: Number of texts to embed in each batch
            cache_dir: Directory to cache embeddings (optional)
            use_preprocessing: Whether to apply preprocessing to texts
        """
        # Set embedding model (default to SentenceTransformer if available)
        if embedding_model is None:
            self.embedding_model = SentenceTransformerEmbeddings()
        else:
            self.embedding_model = embedding_model
        
        self.batch_size = batch_size
        self.use_preprocessing = use_preprocessing
        
        # Set up caching
        self.cache_dir = cache_dir
        self.cache = {}  # In-memory cache
        
        # Create cache directory if specified
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not self.use_preprocessing:
            return text
        
        # Basic preprocessing steps
        processed = text.strip()
        
        # Remove extra whitespace
        processed = " ".join(processed.split())
        
        return processed
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text.
        
        Args:
            text: The text to generate a key for
            
        Returns:
            Cache key string
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Try to get embedding from cache.
        
        Args:
            text: The text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text)
        
        # Check in-memory cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check disk cache if enabled
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading from cache: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """
        Save embedding to cache.
        
        Args:
            text: The text that was embedded
            embedding: The embedding vector
        """
        cache_key = self._get_cache_key(text)
        
        # Save to in-memory cache
        self.cache[cache_key] = embedding
        
        # Save to disk cache if enabled
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                with open(cache_file, 'w') as f:
                    json.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Error saving to cache: {e}")
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector
        """
        # Preprocess the text
        processed_text = self.preprocess(text)
        
        # Try to get from cache
        if use_cache:
            cached_embedding = self._get_from_cache(processed_text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        embedding = self.embedding_model.embed_text(processed_text)
        
        # Save to cache
        if use_cache:
            self._save_to_cache(processed_text, embedding)
        
        return embedding
    
    def embed_documents(
        self, 
        documents: List[str], 
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of texts to embed
            use_cache: Whether to use caching
            show_progress: Whether to show progress information
            
        Returns:
            List of embedding vectors
        """
        # Preprocess all documents
        processed_documents = [self.preprocess(doc) for doc in documents]
        
        # Initialize results
        embeddings = []
        documents_to_embed = []
        document_indices = []
        
        # Check cache for each document
        if use_cache:
            for i, doc in enumerate(processed_documents):
                cached_embedding = self._get_from_cache(doc)
                if cached_embedding is not None:
                    # Add placeholder to maintain order
                    embeddings.append((i, cached_embedding))
                else:
                    documents_to_embed.append(doc)
                    document_indices.append(i)
        else:
            documents_to_embed = processed_documents
            document_indices = list(range(len(processed_documents)))
        
        # If we have documents to embed
        if documents_to_embed:
            if show_progress:
                logger.info(f"Embedding {len(documents_to_embed)} documents in batches of {self.batch_size}")
            
            # Process in batches
            for i in range(0, len(documents_to_embed), self.batch_size):
                batch = documents_to_embed[i:i+self.batch_size]
                batch_indices = document_indices[i:i+self.batch_size]
                
                if show_progress:
                    logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(documents_to_embed)-1)//self.batch_size + 1}")
                
                # Generate embeddings for batch
                batch_embeddings = self.embedding_model.embed_documents(batch)
                
                # Save to cache and add to results
                for j, (idx, embedding) in enumerate(zip(batch_indices, batch_embeddings)):
                    if use_cache:
                        self._save_to_cache(batch[j], embedding)
                    embeddings.append((idx, embedding))
        
        # Sort by original index and extract just the embeddings
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        # Clear in-memory cache
        self.cache = {}
        
        # Clear disk cache if enabled
        if self.cache_dir is not None:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.json'):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        logger.warning(f"Error removing cache file: {e}")


# -----------------------------------------------------------------------------
# Evaluation Utilities
# -----------------------------------------------------------------------------

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    # Convert to numpy arrays
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)


def evaluate_embedding_model(
    model: BaseEmbeddings,
    test_pairs: List[Dict[str, Any]],
    similarity_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate an embedding model on test pairs.
    
    Args:
        model: The embedding model to evaluate
        test_pairs: List of dictionaries with 'text1', 'text2', and 'expected_similar' keys
        similarity_threshold: Threshold for considering two texts similar
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "total_pairs": len(test_pairs),
        "correct_predictions": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "average_similarity": 0,
        "execution_time": 0
    }
    
    start_time = time.time()
    
    similarities = []
    for pair in test_pairs:
        text1 = pair["text1"]
        text2 = pair["text2"]
        expected_similar = pair["expected_similar"]
        
        # Generate embeddings
        emb1 = model.embed_text(text1)
        emb2 = model.embed_text(text2)
        
        # Calculate similarity
        similarity = cosine_similarity(emb1, emb2)
        similarities.append(similarity)
        
        # Check prediction
        predicted_similar = similarity >= similarity_threshold
        
        if predicted_similar == expected_similar:
            results["correct_predictions"] += 1
        elif predicted_similar and not expected_similar:
            results["false_positives"] += 1
        elif not predicted_similar and expected_similar:
            results["false_negatives"] += 1
    
    # Calculate metrics
    results["execution_time"] = time.time() - start_time
    results["average_similarity"] = sum(similarities) / len(similarities) if similarities else 0
    results["accuracy"] = results["correct_predictions"] / results["total_pairs"] if results["total_pairs"] > 0 else 0
    
    return results


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_embedding_model(model_type: str = "sentence_transformer", **kwargs) -> BaseEmbeddings:
    """
    Get an embedding model based on type.
    
    Args:
        model_type: Type of embedding model ('sentence_transformer', 'openai', 'hash')
        **kwargs: Additional arguments for the specific model
        
    Returns:
        An embedding model instance
    """
    if model_type == "sentence_transformer":
        return SentenceTransformerEmbeddings(**kwargs)
    elif model_type == "openai":
        return OpenAIEmbeddings(**kwargs)
    elif model_type == "hash":
        return HashEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model type: {model_type}")


def example_usage():
    """Example usage of the embedding pipeline."""
    # Create a pipeline with default settings
    pipeline = EmbeddingPipeline()
    
    # Example texts
    texts = [
        "Embeddings are numerical representations of text.",
        "Vector representations capture semantic meaning.",
        "Machine learning models use embeddings for NLP tasks."
    ]
    
    # Generate embeddings
    embeddings = pipeline.embed_documents(texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Calculate similarities
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    sim_2_3 = cosine_similarity(embeddings[1], embeddings[2])
    
    print(f"Similarity between text 1 and 2: {sim_1_2:.4f}")
    print(f"Similarity between text 1 and 3: {sim_1_3:.4f}")
    print(f"Similarity between text 2 and 3: {sim_2_3:.4f}")


if __name__ == "__main__":
    example_usage()
