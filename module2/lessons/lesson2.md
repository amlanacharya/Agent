# 🚀 Module 2: Memory Systems - Lesson 2: Vector Databases 🔍

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧮 Understand **vector embeddings** and their role in AI memory
- 🔢 Learn how to **convert text to vectors** using embeddings
- 📊 Implement a **simple vector database** for semantic search
- 🔍 Master **similarity search** techniques for finding relevant information
- 🧩 Build a **retrieval system** that can find semantically similar content

---

## 📚 Introduction to Vector Databases

Traditional memory systems store information as key-value pairs, making it easy to retrieve exact matches but difficult to find semantically similar content. Vector databases solve this problem by:

1. **Converting text to numerical vectors** (embeddings)
2. **Storing these vectors** in an efficient data structure
3. **Finding similar vectors** using distance metrics
4. **Retrieving the original content** associated with those vectors

This approach enables **semantic search** - finding information based on meaning rather than exact keyword matches.

---

## 🧮 Understanding Vector Embeddings

![Embeddings](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Vector embeddings are numerical representations of text that capture semantic meaning. Words or phrases with similar meanings have similar vector representations.

### How Embeddings Work

1. **Text Input**: Start with a word, sentence, or document
2. **Embedding Model**: Process through a pre-trained model (like Word2Vec, GloVe, or modern transformer models)
3. **Vector Output**: Generate a fixed-length vector (typically 100-1536 dimensions)

### Example

The sentences "I love dogs" and "I adore canines" would have similar vector representations because they have similar meanings, despite using different words.

### Simple Visualization

Imagine a 3D space where:
- "Dog" is at coordinates [0.2, 0.5, 0.1]
- "Cat" is at [0.3, 0.4, 0.2] (relatively close to "Dog")
- "Automobile" is at [0.9, 0.1, 0.7] (far from both "Dog" and "Cat")

In reality, these vectors have many more dimensions, but the principle is the same - semantic similarity is represented by proximity in vector space.

---

## 🔢 Converting Text to Vectors

![Conversion](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

In a production environment, you would use a pre-trained embedding model from libraries like:
- OpenAI's text-embedding models
- Hugging Face's sentence-transformers
- TensorFlow Hub's Universal Sentence Encoder

For our learning purposes, we'll implement a simplified embedding function that simulates this process:

```python
import numpy as np
from collections import Counter

def simple_embedding(text, dimensions=100):
    """
    Create a simple embedding vector for text.
    This is a very simplified version for demonstration purposes.

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
        word_hash = hash(word) % dimensions
        # Use the count and a secondary hash to determine the value
        vector[word_hash] = count * (hash(word + 'salt') % 10 + 1) / 10

    # Normalize the vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector
```

In a real implementation, you would use a more sophisticated model:

```python
# Example using OpenAI's embedding API
import openai

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding from OpenAI API"""
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response["data"][0]["embedding"]

# Example using sentence-transformers
from sentence_transformers import SentenceTransformer

def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    """Get embedding using sentence-transformers"""
    model = SentenceTransformer(model_name)
    return model.encode(text)
```

---

## 📊 Implementing a Simple Vector Database

![Database](https://media.giphy.com/media/3o7btNDyBs5dKdhTqM/giphy.gif)

Now, let's implement a simple vector database that can:
1. Store text and its vector representation
2. Find similar vectors using cosine similarity
3. Retrieve the most similar items for a query

```python
class SimpleVectorDB:
    def __init__(self, embedding_function=None):
        """
        Initialize a simple vector database

        Args:
            embedding_function: Function to convert text to vectors
        """
        self.items = []  # Will store (id, text, vector) tuples
        self.embedding_function = embedding_function or simple_embedding

    def add_item(self, item_id, text):
        """
        Add an item to the database

        Args:
            item_id: Unique identifier for the item
            text: The text content to store
        """
        vector = self.embedding_function(text)
        self.items.append((item_id, text, vector))
        return len(self.items) - 1  # Return the index

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

    def search(self, query, top_k=3):
        """
        Search for items similar to the query

        Args:
            query (str): The search query
            top_k (int): Number of results to return

        Returns:
            list: Top k similar items with similarity scores
        """
        query_vector = self.embedding_function(query)

        # Calculate similarity for all items
        similarities = [
            (item_id, text, self.similarity(query_vector, vector))
            for item_id, text, vector in self.items
        ]

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[2], reverse=True)

        # Return top k results
        return similarities[:top_k]
```

---

## 🔍 Similarity Search Techniques

![Search](https://media.giphy.com/media/3o7TKT6gL5B7Lzq3re/giphy.gif)

Our simple implementation uses **cosine similarity**, but there are several common distance metrics for vector search:

### 1. Cosine Similarity
- Measures the cosine of the angle between vectors
- Range: -1 (opposite) to 1 (identical)
- Good for: Text similarity where magnitude doesn't matter

### 2. Euclidean Distance
- Measures the straight-line distance between vectors
- Range: 0 (identical) to ∞
- Good for: When absolute distances matter

### 3. Dot Product
- Measures the product of vector magnitudes and cosine similarity
- Range: -∞ to ∞
- Good for: When both direction and magnitude matter

### 4. Manhattan Distance
- Measures the sum of absolute differences between vector components
- Range: 0 (identical) to ∞
- Good for: Grid-like spaces or when diagonal movement is not allowed

### Optimization Techniques

For large vector databases, searching through all vectors becomes inefficient. Advanced techniques include:

- **Approximate Nearest Neighbors (ANN)**: Sacrifices some accuracy for speed
- **Locality-Sensitive Hashing (LSH)**: Groups similar vectors into the same "buckets"
- **Hierarchical Navigable Small World (HNSW)**: Creates a graph structure for efficient navigation
- **Inverted File Index (IVF)**: Divides the vector space into clusters

These techniques are implemented in libraries like FAISS, Annoy, and NMSLIB.

---

## 🧩 Building a Retrieval System

![Retrieval](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Let's combine our vector database with our memory system from Lesson 1 to create a retrieval-based memory system:

```python
class RetrievalMemory:
    def __init__(self, embedding_function=None):
        """
        Initialize a retrieval-based memory system

        Args:
            embedding_function: Function to convert text to vectors
        """
        self.vector_db = SimpleVectorDB(embedding_function)
        self.next_id = 0

    def store(self, text, metadata=None):
        """
        Store information in memory

        Args:
            text (str): The text to store
            metadata (dict, optional): Additional metadata

        Returns:
            int: The ID of the stored item
        """
        item_id = f"item_{self.next_id}"
        self.next_id += 1

        # Store the item in the vector database
        self.vector_db.add_item(item_id, text)

        return item_id

    def retrieve(self, query, top_k=3):
        """
        Retrieve information similar to the query

        Args:
            query (str): The search query
            top_k (int): Number of results to return

        Returns:
            list: Top k similar items with similarity scores
        """
        return self.vector_db.search(query, top_k)

    def retrieve_most_similar(self, query):
        """
        Retrieve the most similar item to the query

        Args:
            query (str): The search query

        Returns:
            tuple: (item_id, text, similarity) or None if no items
        """
        results = self.retrieve(query, top_k=1)
        return results[0] if results else None
```

---

## 💪 Practice Exercises

1. **Implement a Persistent Vector Database**:
   - Extend the SimpleVectorDB to save and load vectors from disk
   - Add functionality to update existing items
   - Implement batch addition of multiple items

2. **Experiment with Different Similarity Metrics**:
   - Implement Euclidean distance and Manhattan distance
   - Compare the results of different metrics on the same dataset
   - Create a function that recommends the best metric for a given use case

3. **Build a Hybrid Retrieval System**:
   - Combine keyword search with vector search
   - Implement a scoring system that considers both exact matches and semantic similarity
   - Create a method to explain why a particular result was returned

---

## 🔍 Key Concepts to Remember

1. **Vector Embeddings**: Numerical representations that capture semantic meaning
2. **Similarity Metrics**: Different ways to measure distance between vectors
3. **Vector Databases**: Specialized storage for efficient similarity search
4. **Retrieval Systems**: Combine embeddings and search to find relevant information
5. **Optimization Techniques**: Methods to make vector search scalable

---

## 🚀 Next Steps

In the next lesson, we'll explore:
- Advanced retrieval patterns for contextual memory
- Combining different memory types for more effective retrieval
- Implementing conversation memory with semantic search
- Building a more sophisticated knowledge retrieval system

---

## 📚 Resources

- [Pinecone Vector Database Guide](https://www.pinecone.io/learn/vector-database/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)

---

## 🎯 Mini-Project Progress: Knowledge Base Assistant

![Knowledge Base](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

In this lesson, we've learned about vector databases and similarity search, which are essential components for our Knowledge Base Assistant. We can now:
- Convert text to vector embeddings
- Store these vectors in a database
- Find semantically similar content
- Retrieve information based on meaning rather than exact matches

In the next lesson, we'll explore how to use these capabilities to build a more sophisticated retrieval system for our Knowledge Base Assistant.

---

> 💡 **Note on LLM Integration**: This lesson uses simulated embedding functions for demonstration purposes. In a real implementation, you would use a dedicated embedding model from providers like OpenAI, Hugging Face, or open-source alternatives. For LLM integration, see the Module 2-LLM version.

---

Happy coding! 🚀
