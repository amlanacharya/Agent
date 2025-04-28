"""
Lesson 3 Exercises: Embedding Selection & Generation
-------------------------------------------------
This module contains exercises for working with embeddings and embedding pipelines.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the code directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
from embedding_pipelines import (
    BaseEmbeddings,
    HashEmbeddings,
    SentenceTransformerEmbeddings,
    OpenAIEmbeddings,
    EmbeddingPipeline,
    cosine_similarity
)


# -----------------------------------------------------------------------------
# Exercise 1: Implement a Custom Embedding Model
# -----------------------------------------------------------------------------

class KeywordEmbeddings(BaseEmbeddings):
    """
    Exercise 1: Implement a custom embedding model based on keyword matching.

    This model creates embeddings by checking for the presence of keywords
    in the text and setting specific dimensions accordingly.
    """

    def __init__(self, dimension: int = 100, keywords: Optional[List[str]] = None):
        """
        Initialize the keyword embeddings.

        Args:
            dimension: Dimension of the embedding vectors
            keywords: List of keywords to use (if None, will use default keywords)
        """
        self.dimension = dimension

        # Use provided keywords or default ones
        if keywords is None:
            # Default keywords for different topics
            self.keywords = [
                "machine", "learning", "artificial", "intelligence", "neural",
                "network", "deep", "model", "data", "training", "algorithm",
                "python", "code", "programming", "software", "development",
                "computer", "science", "technology", "engineering", "math",
                "natural", "language", "processing", "nlp", "text", "speech",
                "vision", "image", "recognition", "classification", "detection",
                "web", "internet", "cloud", "server", "database", "api",
                "security", "privacy", "encryption", "blockchain", "crypto",
                "mobile", "app", "device", "hardware", "system", "platform",
                "user", "interface", "experience", "design", "product"
            ]
        else:
            self.keywords = keywords

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # 1. Initialize a vector of zeros with self.dimension
        vector = [0.0] * self.dimension

        # 2. Process the input text (lowercase, remove punctuation)
        import re
        processed_text = re.sub(r'[^\w\s]', '', text.lower())

        # 3. For each keyword in self.keywords:
        for keyword in self.keywords:
            # Check if the keyword is in the text
            if keyword in processed_text:
                # Use the hash of the keyword to determine which dimension to set
                import hashlib
                hash_value = int(hashlib.md5(keyword.encode()).hexdigest(), 16)
                dimension_index = hash_value % self.dimension

                # Set a specific dimension to 1.0
                vector[dimension_index] = 1.0

        # 4. Normalize the vector using self.normalize_vector()
        # 5. Return the normalized vector
        return self.normalize_vector(vector)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

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
# Exercise 2: Implement a Domain-Specific Embedding Adapter
# -----------------------------------------------------------------------------

class DomainAdapter:
    """
    Exercise 2: Implement a domain adapter for embeddings.

    This class adapts general embeddings to be more effective for a specific domain
    by applying domain-specific transformations.
    """

    def __init__(
        self,
        base_model: BaseEmbeddings,
        domain_keywords: List[str],
        boost_factor: float = 1.5
    ):
        """
        Initialize the domain adapter.

        Args:
            base_model: The base embedding model to adapt
            domain_keywords: List of keywords relevant to the domain
            boost_factor: Factor to boost dimensions related to domain keywords
        """
        self.base_model = base_model
        self.domain_keywords = domain_keywords
        self.boost_factor = boost_factor

        # Pre-compute keyword embeddings
        self.keyword_embeddings = [self.base_model.embed_text(kw) for kw in domain_keywords]

    def adapt_embedding(self, embedding: List[float]) -> List[float]:
        """
        Adapt an embedding to the specific domain.

        Args:
            embedding: The original embedding vector

        Returns:
            Domain-adapted embedding vector
        """
        # 1. Start with the original embedding
        adapted_embedding = np.array(embedding, dtype=np.float32)

        # 2. For each keyword embedding:
        for keyword_embedding in self.keyword_embeddings:
            # Calculate similarity with the original embedding
            similarity = cosine_similarity(embedding, keyword_embedding)

            # If similarity is above a threshold, boost the embedding in that direction
            if similarity > 0.3:  # Threshold for considering relevant
                # Convert keyword embedding to numpy array
                keyword_vector = np.array(keyword_embedding, dtype=np.float32)

                # Boost the embedding in the direction of the keyword
                # The higher the similarity, the stronger the boost
                boost = similarity * self.boost_factor
                adapted_embedding = adapted_embedding + (keyword_vector * boost)

        # 3. Normalize the resulting vector
        norm = np.linalg.norm(adapted_embedding)
        if norm > 0:
            adapted_embedding = adapted_embedding / norm

        # 4. Return the adapted embedding
        return adapted_embedding.tolist()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a domain-adapted embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Domain-adapted embedding vector
        """
        # Get the base embedding
        base_embedding = self.base_model.embed_text(text)

        # Adapt the embedding to the domain
        return self.adapt_embedding(base_embedding)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate domain-adapted embeddings for multiple documents.

        Args:
            documents: List of texts to embed

        Returns:
            List of domain-adapted embedding vectors
        """
        # Get base embeddings
        base_embeddings = self.base_model.embed_documents(documents)

        # Adapt each embedding
        return [self.adapt_embedding(emb) for emb in base_embeddings]


# -----------------------------------------------------------------------------
# Exercise 3: Implement an Advanced Caching System
# -----------------------------------------------------------------------------

class AdvancedEmbeddingCache:
    """
    Exercise 3: Implement an advanced caching system for embeddings.

    This cache system includes:
    - Time-based expiration
    - LRU (Least Recently Used) eviction policy
    - Similarity-based retrieval
    """

    def __init__(
        self,
        max_size: int = 1000,
        expiration_seconds: int = 86400,  # 24 hours
        similarity_threshold: float = 0.95
    ):
        """
        Initialize the advanced embedding cache.

        Args:
            max_size: Maximum number of items to store in cache
            expiration_seconds: Time in seconds after which cache items expire
            similarity_threshold: Threshold for similarity-based retrieval
        """
        self.max_size = max_size
        self.expiration_seconds = expiration_seconds
        self.similarity_threshold = similarity_threshold

        # Cache storage: {key: (embedding, timestamp, access_count)}
        self.cache = {}

        # Access tracking
        self.access_order = []

    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text.

        Args:
            text: The text to generate a key for

        Returns:
            Cache key string
        """
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str, embedding_model: BaseEmbeddings) -> Optional[List[float]]:
        """
        Get an embedding from cache, with similarity-based retrieval.

        Args:
            text: The text to get embedding for
            embedding_model: Model to use if cache miss

        Returns:
            Cached embedding or None if not found
        """
        import time
        current_time = time.time()

        # 1. Generate cache key for the text
        cache_key = self._get_cache_key(text)

        # 2. Check if the key exists in the cache
        if cache_key in self.cache:
            embedding, timestamp, access_count = self.cache[cache_key]

            # 3. If found, check if it's expired
            if current_time - timestamp <= self.expiration_seconds:
                # 4. If not expired, update access info and return the embedding
                # Update access count and timestamp
                self.cache[cache_key] = (embedding, timestamp, access_count + 1)

                # Update access order (move to end of list)
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                self.access_order.append(cache_key)

                return embedding

        # 5. If not found by exact match, try similarity-based retrieval
        # Generate embedding for the query text
        query_embedding = embedding_model.embed_text(text)

        # Check similarity with cached embeddings
        best_match = None
        best_similarity = 0

        for key, (cached_embedding, timestamp, _) in self.cache.items():
            # Skip expired items
            if current_time - timestamp > self.expiration_seconds:
                continue

            # Calculate similarity
            similarity = cosine_similarity(query_embedding, cached_embedding)

            # If similarity is above threshold and better than current best
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_embedding

        # If a similar embedding is found, return it
        if best_match is not None:
            return best_match

        # 6. Return None if no suitable embedding is found
        return None

    def put(self, text: str, embedding: List[float]) -> None:
        """
        Store an embedding in the cache.

        Args:
            text: The text that was embedded
            embedding: The embedding vector
        """
        import time

        # 1. Generate cache key for the text
        cache_key = self._get_cache_key(text)

        # 2. Store the embedding with current timestamp and initial access count
        current_time = time.time()
        self.cache[cache_key] = (embedding, current_time, 1)

        # 3. Update access order
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

        # 4. If cache exceeds max_size, evict least recently used items
        while len(self.cache) > self.max_size:
            # Get the oldest item
            oldest_key = self.access_order[0]

            # Remove it from cache and access order
            del self.cache[oldest_key]
            self.access_order.pop(0)

    def clear_expired(self) -> int:
        """
        Clear expired items from the cache.

        Returns:
            Number of items cleared
        """
        import time

        # 1. Get current timestamp
        current_time = time.time()

        # 2. Identify and remove expired items
        expired_keys = []
        for key, (_, timestamp, _) in self.cache.items():
            if current_time - timestamp > self.expiration_seconds:
                expired_keys.append(key)

        # Remove expired items
        for key in expired_keys:
            del self.cache[key]

        # 3. Update access order
        self.access_order = [key for key in self.access_order if key not in expired_keys]

        # 4. Return count of removed items
        return len(expired_keys)


# -----------------------------------------------------------------------------
# Exercise 4: Implement a Batching Mechanism with Progress Tracking
# -----------------------------------------------------------------------------

class BatchProcessor:
    """
    Exercise 4: Implement a batching mechanism with progress tracking.

    This class processes large collections of documents in batches,
    with progress tracking and error handling.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddings,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Initialize the batch processor.

        Args:
            embedding_model: The embedding model to use
            batch_size: Number of documents to process in each batch
            show_progress: Whether to show progress information
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.show_progress = show_progress

        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.total_time = 0

    def process_batch(self, batch: List[str]) -> List[Optional[List[float]]]:
        """
        Process a single batch of documents.

        Args:
            batch: List of texts to process

        Returns:
            List of embeddings (or None for failed items)
        """
        results = [None] * len(batch)

        # 1. Try to embed the entire batch
        try:
            batch_embeddings = self.embedding_model.embed_documents(batch)
            return batch_embeddings
        except Exception as e:
            # 2. If batch embedding fails, fall back to processing one by one
            for i, text in enumerate(batch):
                try:
                    results[i] = self.embedding_model.embed_text(text)
                except Exception:
                    # If individual embedding fails, leave as None
                    results[i] = None

        # 3. Return list of embeddings (with None for any that failed)
        return results

    def process_documents(self, documents: List[str]) -> List[Optional[List[float]]]:
        """
        Process a collection of documents in batches.

        Args:
            documents: List of texts to process

        Returns:
            List of embeddings (or None for failed items)
        """
        import time

        # 1. Initialize results list and timing
        results = []
        start_time = time.time()

        # 2. Process documents in batches of self.batch_size
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(documents), self.batch_size):
            # Get the current batch
            batch = documents[batch_idx:batch_idx + self.batch_size]

            # 3. For each batch:
            #    - Process the batch
            batch_start_time = time.time()
            batch_results = self.process_batch(batch)
            batch_time = time.time() - batch_start_time

            #    - Update statistics
            self.total_processed += len(batch)
            self.total_errors += batch_results.count(None)
            self.total_time += batch_time

            #    - Show progress if enabled
            if self.show_progress:
                current_batch = batch_idx // self.batch_size + 1
                success_rate = (len(batch) - batch_results.count(None)) / len(batch) * 100
                print(f"Batch {current_batch}/{total_batches}: "
                      f"Processed {len(batch)} documents in {batch_time:.2f}s "
                      f"({success_rate:.1f}% success)")

            # Add batch results to overall results
            results.extend(batch_results)

        # 4. Return all results
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "total_time": self.total_time,
            "average_time_per_doc": self.total_time / max(1, self.total_processed),
            "success_rate": (self.total_processed - self.total_errors) / max(1, self.total_processed)
        }


# -----------------------------------------------------------------------------
# Exercise 5: Implement Dimensionality Reduction for Embeddings
# -----------------------------------------------------------------------------

class EmbeddingReducer:
    """
    Exercise 5: Implement dimensionality reduction for embeddings.

    This class reduces the dimensionality of embeddings while
    preserving semantic similarity as much as possible.
    """

    def __init__(
        self,
        original_dim: int,
        target_dim: int,
        method: str = "random_projection"
    ):
        """
        Initialize the embedding reducer.

        Args:
            original_dim: Original embedding dimension
            target_dim: Target dimension after reduction
            method: Reduction method ('random_projection', 'pca', or 'svd')
        """
        self.original_dim = original_dim
        self.target_dim = target_dim
        self.method = method

        # Initialize the projection matrix
        if method == "random_projection":
            # Random projection matrix
            self.projection = self._create_random_projection()
        else:
            # For PCA and SVD, we'll need to fit on data
            self.projection = None
            self.is_fitted = False

    def _create_random_projection(self) -> np.ndarray:
        """
        Create a random projection matrix.

        Returns:
            Random projection matrix
        """
        # 1. Create a random matrix of shape (original_dim, target_dim)
        projection = np.random.randn(self.original_dim, self.target_dim)

        # 2. Normalize the columns to preserve distances approximately
        # This helps maintain the relative distances between points after projection
        for col in range(projection.shape[1]):
            # Get the column
            column = projection[:, col]

            # Normalize the column
            norm = np.linalg.norm(column)
            if norm > 0:
                projection[:, col] = column / norm

        # 3. Return the projection matrix
        return projection

    def fit(self, embeddings: List[List[float]]) -> None:
        """
        Fit the reducer on a set of embeddings (for PCA and SVD).

        Args:
            embeddings: List of embedding vectors
        """
        # 1. Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # 2. If method is 'pca' or 'svd', compute the projection
        if self.method == "pca":
            # Perform PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.target_dim)
            pca.fit(embeddings_array)
            self.projection = pca.components_.T  # Transpose to get shape (original_dim, target_dim)

        elif self.method == "svd":
            # Perform SVD
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=self.target_dim)
            svd.fit(embeddings_array)
            self.projection = svd.components_.T  # Transpose to get shape (original_dim, target_dim)

        # For random projection, we already created the projection matrix in __init__

        # 3. Set is_fitted to True
        self.is_fitted = True

    def reduce(self, embedding: List[float]) -> List[float]:
        """
        Reduce the dimensionality of a single embedding.

        Args:
            embedding: The embedding vector to reduce

        Returns:
            Reduced embedding vector
        """
        # Check if the model is fitted
        if not self.is_fitted and self.method != "random_projection":
            raise ValueError("Reducer must be fitted before reducing embeddings")

        # 1. Convert embedding to numpy array
        embedding_array = np.array(embedding, dtype=np.float32)

        # Ensure correct shape (should be a row vector)
        if embedding_array.ndim == 1:
            embedding_array = embedding_array.reshape(1, -1)

        # 2. Apply the projection
        if self.method == "pca" or self.method == "svd":
            # For PCA and SVD, we use matrix multiplication with the projection
            reduced = np.dot(embedding_array, self.projection)
        else:  # random_projection
            # For random projection, we use matrix multiplication with the projection
            reduced = np.dot(embedding_array, self.projection)

        # 3. Return the reduced embedding as a list
        return reduced.flatten().tolist()

    def reduce_batch(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Reduce the dimensionality of multiple embeddings.

        Args:
            embeddings: List of embedding vectors to reduce

        Returns:
            List of reduced embedding vectors
        """
        return [self.reduce(emb) for emb in embeddings]


# -----------------------------------------------------------------------------
# Helper Functions for Testing
# -----------------------------------------------------------------------------

def test_exercise1():
    """Test the KeywordEmbeddings implementation."""
    print("\n=== Testing Exercise 1: KeywordEmbeddings ===\n")

    # Create the model
    model = KeywordEmbeddings(dimension=50)

    # Test texts
    texts = [
        "Machine learning models are trained on data.",
        "Python is a popular programming language for AI.",
        "The weather is sunny today, perfect for a picnic."
    ]

    # Generate embeddings
    embeddings = model.embed_documents(texts)

    # Print info
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        print(f"Text {i+1}: '{text}'")
        print(f"  Embedding dimension: {len(embedding)}")
        print(f"  Norm: {np.linalg.norm(embedding):.4f}")

    # Calculate similarities
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    sim_2_3 = cosine_similarity(embeddings[1], embeddings[2])

    print("\nSimilarities:")
    print(f"  Text 1 & 2: {sim_1_2:.4f}")
    print(f"  Text 1 & 3: {sim_1_3:.4f}")
    print(f"  Text 2 & 3: {sim_2_3:.4f}")

    # Check if the model is working correctly
    if sim_1_2 > sim_1_3:
        print("\n✅ Model correctly identifies that texts 1 and 2 are more similar (both about tech).")
    else:
        print("\n❌ Model does not correctly identify text similarities.")


def test_exercise2():
    """Test the DomainAdapter implementation."""
    print("\n=== Testing Exercise 2: DomainAdapter ===\n")

    # Create base model
    base_model = SentenceTransformerEmbeddings()

    # Create domain adapter for medical domain
    medical_keywords = [
        "patient", "doctor", "hospital", "treatment", "disease",
        "diagnosis", "medicine", "symptom", "healthcare", "medical"
    ]

    medical_adapter = DomainAdapter(base_model, medical_keywords)

    # Test texts
    texts = [
        "The patient was diagnosed with pneumonia.",
        "The doctor prescribed antibiotics for the infection.",
        "The company released a new smartphone today."
    ]

    try:
        # Generate regular embeddings
        regular_embeddings = base_model.embed_documents(texts)

        # Generate domain-adapted embeddings
        adapted_embeddings = [medical_adapter.embed_text(text) for text in texts]

        # Calculate similarities with regular embeddings
        reg_sim_1_2 = cosine_similarity(regular_embeddings[0], regular_embeddings[1])
        reg_sim_1_3 = cosine_similarity(regular_embeddings[0], regular_embeddings[2])

        # Calculate similarities with adapted embeddings
        adp_sim_1_2 = cosine_similarity(adapted_embeddings[0], adapted_embeddings[1])
        adp_sim_1_3 = cosine_similarity(adapted_embeddings[0], adapted_embeddings[2])

        print("Regular Embeddings Similarities:")
        print(f"  Medical texts (1 & 2): {reg_sim_1_2:.4f}")
        print(f"  Medical & Non-medical (1 & 3): {reg_sim_1_3:.4f}")
        print(f"  Difference: {reg_sim_1_2 - reg_sim_1_3:.4f}")

        print("\nDomain-Adapted Embeddings Similarities:")
        print(f"  Medical texts (1 & 2): {adp_sim_1_2:.4f}")
        print(f"  Medical & Non-medical (1 & 3): {adp_sim_1_3:.4f}")
        print(f"  Difference: {adp_sim_1_2 - adp_sim_1_3:.4f}")

        # Check if domain adaptation improved the distinction
        if (adp_sim_1_2 - adp_sim_1_3) > (reg_sim_1_2 - reg_sim_1_3):
            print("\n✅ Domain adaptation successfully increased the distinction between domain and non-domain texts.")
        else:
            print("\n❌ Domain adaptation did not improve the distinction between domain and non-domain texts.")

    except Exception as e:
        print(f"Error testing domain adapter: {e}")


def test_exercise3():
    """Test the AdvancedEmbeddingCache implementation."""
    print("\n=== Testing Exercise 3: AdvancedEmbeddingCache ===\n")

    # Create embedding model
    model = HashEmbeddings()

    # Create cache
    cache = AdvancedEmbeddingCache(max_size=5, expiration_seconds=10)

    # Test texts
    texts = [
        "This is the first test text.",
        "This is the second test text.",
        "This is the third test text.",
        "This is the fourth test text.",
        "This is the fifth test text.",
        "This is the sixth test text.",  # Should cause eviction
        "This is a very similar text to the first one.",  # Should match by similarity
    ]

    # Add items to cache
    print("Adding items to cache...")
    for text in texts[:5]:
        embedding = model.embed_text(text)
        cache.put(text, embedding)
        print(f"  Added: '{text}'")

    # Test exact retrieval
    print("\nTesting exact retrieval...")
    for text in texts[:5]:
        embedding = cache.get(text, model)
        if embedding is not None:
            print(f"  ✅ Found: '{text}'")
        else:
            print(f"  ❌ Not found: '{text}'")

    # Test eviction
    print("\nTesting cache eviction...")
    embedding = model.embed_text(texts[5])
    cache.put(texts[5], embedding)
    print(f"  Added: '{texts[5]}'")

    # Check if oldest item was evicted
    embedding = cache.get(texts[0], model)
    if embedding is None:
        print(f"  ✅ Evicted: '{texts[0]}'")
    else:
        print(f"  ❌ Not evicted: '{texts[0]}'")

    # Test similarity-based retrieval
    print("\nTesting similarity-based retrieval...")
    embedding = cache.get(texts[6], model)
    if embedding is not None:
        print(f"  ✅ Found similar: '{texts[6]}'")
    else:
        print(f"  ❌ Not found similar: '{texts[6]}'")

    # Test expiration
    print("\nTesting cache expiration...")
    import time
    print("  Waiting for items to expire...")
    time.sleep(11)  # Wait for expiration

    cleared = cache.clear_expired()
    print(f"  Cleared {cleared} expired items")

    # Check if items are expired
    for text in texts[1:6]:
        embedding = cache.get(text, model)
        if embedding is None:
            print(f"  ✅ Expired: '{text}'")
        else:
            print(f"  ❌ Not expired: '{text}'")


def test_exercise4():
    """Test the BatchProcessor implementation."""
    print("\n=== Testing Exercise 4: BatchProcessor ===\n")

    # Create embedding model
    model = HashEmbeddings()

    # Create batch processor
    processor = BatchProcessor(model, batch_size=3, show_progress=True)

    # Generate test documents
    documents = [f"This is test document {i+1}." for i in range(10)]

    # Process documents
    print("Processing documents in batches...")
    embeddings = processor.process_documents(documents)

    # Check results
    success_count = sum(1 for emb in embeddings if emb is not None)
    print(f"\nSuccessfully processed {success_count}/{len(documents)} documents")

    # Print statistics
    stats = processor.get_statistics()
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_exercise5():
    """Test the EmbeddingReducer implementation."""
    print("\n=== Testing Exercise 5: EmbeddingReducer ===\n")

    # Create embedding model
    model = HashEmbeddings(dimension=100)

    # Create reducer
    reducer = EmbeddingReducer(original_dim=100, target_dim=20)

    # Test texts
    texts = [
        "Machine learning models are trained on data.",
        "Neural networks are a type of machine learning model.",
        "The weather is sunny today, perfect for a picnic."
    ]

    # Generate original embeddings
    original_embeddings = model.embed_documents(texts)

    # Fit and reduce
    reducer.fit(original_embeddings)
    reduced_embeddings = reducer.reduce_batch(original_embeddings)

    # Print dimensions
    print(f"Original dimension: {len(original_embeddings[0])}")
    print(f"Reduced dimension: {len(reduced_embeddings[0])}")
    print(f"Reduction ratio: {len(reduced_embeddings[0]) / len(original_embeddings[0]):.2f}")

    # Calculate similarities with original embeddings
    orig_sim_1_2 = cosine_similarity(original_embeddings[0], original_embeddings[1])
    orig_sim_1_3 = cosine_similarity(original_embeddings[0], original_embeddings[2])

    # Calculate similarities with reduced embeddings
    red_sim_1_2 = cosine_similarity(reduced_embeddings[0], reduced_embeddings[1])
    red_sim_1_3 = cosine_similarity(reduced_embeddings[0], reduced_embeddings[2])

    print("\nOriginal Embeddings Similarities:")
    print(f"  Similar texts (1 & 2): {orig_sim_1_2:.4f}")
    print(f"  Different texts (1 & 3): {orig_sim_1_3:.4f}")

    print("\nReduced Embeddings Similarities:")
    print(f"  Similar texts (1 & 2): {red_sim_1_2:.4f}")
    print(f"  Different texts (1 & 3): {red_sim_1_3:.4f}")

    # Check if reduction preserved relative similarities
    if (red_sim_1_2 > red_sim_1_3) == (orig_sim_1_2 > orig_sim_1_3):
        print("\n✅ Dimensionality reduction preserved relative similarities.")
    else:
        print("\n❌ Dimensionality reduction did not preserve relative similarities.")


if __name__ == "__main__":
    # Uncomment to test individual exercises
    # test_exercise1()
    # test_exercise2()
    # test_exercise3()
    # test_exercise4()
    # test_exercise5()

    # Or run all tests
    print("Running all exercise tests...")
    test_exercise1()
    test_exercise2()
    test_exercise3()
    test_exercise4()
    test_exercise5()

    print("\nAll tests completed.")
