# 🧠 Module 2 LLM Version: Memory Systems - Lesson 2 🔍

![Vector Database](https://media.giphy.com/media/l0HlHFRbmaZtBRhXG/giphy.gif)

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧮 Understand **vector embeddings** and how they're generated with real LLMs
- 🔢 Learn how to **convert text to vectors** using the Groq API
- 📊 Implement a **vector database** with real embeddings for semantic search
- 🔍 Master **similarity search** techniques for finding relevant information
- 🧩 Build a **retrieval system** with LLM-powered query expansion

---

## 📚 Introduction to Vector Databases with Real Embeddings

![Vector Space](https://media.giphy.com/media/26ufoAcuAQKjkVZHW/giphy.gif)

Vector databases are a fundamental component of modern AI memory systems. They allow us to store and retrieve information based on semantic meaning rather than exact keyword matches. In this lesson, we'll explore how to implement a vector database with real embeddings from the Groq API.

> 💡 **Note on LLM Integration**: Unlike the standard Module 2 which uses simulated embeddings, this version uses real embeddings from the Groq API. This allows us to perform true semantic search based on the meaning of text rather than just keywords.

### What Are Vector Embeddings?

Vector embeddings are numerical representations of text (or other data) that capture semantic meaning. They convert words, sentences, or documents into high-dimensional vectors where:

- Similar texts have vectors that are close together in the vector space
- Dissimilar texts have vectors that are far apart
- The relationships between concepts are preserved in the vector space

For example, in a good embedding space:
- "dog" and "puppy" would be close together
- "computer" and "laptop" would be close together
- "dog" and "computer" would be far apart

### How LLMs Generate Embeddings

Large Language Models like those available through the Groq API can generate high-quality embeddings by:

1. Processing the input text through multiple layers of neural networks
2. Capturing semantic relationships between words and concepts
3. Encoding these relationships into fixed-length vectors
4. Ensuring similar concepts have similar vector representations

These embeddings are much more powerful than traditional techniques like TF-IDF or bag-of-words because they capture deep semantic meaning rather than just word frequency.

---

## 🔢 Converting Text to Vectors with the Groq API

![Conversion](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

Let's look at how to convert text to vectors using the Groq API:

```python
def get_embeddings(self, texts, model=None):
    """
    Get embeddings for text using the Groq API
    
    Args:
        texts (str or list): Text or list of texts to get embeddings for
        model (str, optional): The embedding model to use
        
    Returns:
        list: List of embedding vectors
    """
    model = model or self.default_embedding_model
    
    # Convert single text to list
    if isinstance(texts, str):
        texts = [texts]
    
    payload = {
        "model": model,
        "input": texts
    }
    
    response = requests.post(
        f"{self.base_url}/embeddings",
        headers=self.headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Error getting embeddings: {response.text}")
    
    result = response.json()
    return [item["embedding"] for item in result["data"]]
```

This function sends text to the Groq API and receives back embedding vectors that represent the semantic meaning of the text. These vectors typically have hundreds of dimensions, allowing them to capture complex relationships between concepts.

### Handling API Failures

When working with external APIs, it's important to handle potential failures gracefully. Our implementation includes a fallback mechanism:

```python
def _get_groq_embedding(self, text):
    """Get embeddings for text using the Groq API"""
    try:
        embeddings = self.groq_client.get_embeddings(text)
        return embeddings[0]  # Return the first (and only) embedding
    except Exception as e:
        # Fallback to a simple hash-based embedding if Groq API fails
        print(f"Warning: Groq embedding failed, using fallback embedding. Error: {e}")
        return self._fallback_embedding(text)
```

This ensures that our system can continue to function even if the API is temporarily unavailable, though with reduced semantic understanding.

---

## 📊 Implementing a Vector Database with Real Embeddings

![Database](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

Now let's look at how to implement a vector database that uses these real embeddings:

```python
class SimpleVectorDB:
    """A simple vector database implementation with real embeddings from Groq API."""
    
    def __init__(self, embedding_function=None):
        """Initialize the vector database"""
        self.items = []
        self.groq_client = GroqClient()
        self.embedding_function = embedding_function or self._get_groq_embedding
    
    def add(self, text, metadata=None):
        """Add an item to the vector database"""
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
```

This implementation stores items with their text, metadata, and most importantly, their embedding vectors. The embeddings are generated using the Groq API, ensuring high-quality semantic representations.

### Batch Operations for Efficiency

For efficiency, we can also add items in batch:

```python
def add_batch(self, texts, metadatas=None):
    """Add multiple items to the vector database"""
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
```

This allows us to make a single API call to get embeddings for multiple texts, which is much more efficient than making separate calls for each text.

---

## 🔍 Similarity Search with Real Embeddings

![Search](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

The power of vector databases comes from their ability to find semantically similar content. Let's look at how to implement similarity search:

```python
def search(self, query, top_k=5, filter_func=None):
    """Search the vector database for items similar to the query"""
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
```

This function:
1. Converts the query to an embedding vector using the Groq API
2. Calculates the similarity between the query vector and each item vector
3. Sorts the results by similarity
4. Returns the top k most similar items

### Cosine Similarity

The most common similarity metric for embeddings is cosine similarity:

```python
def _cosine_similarity(self, vec1, vec2):
    """Calculate cosine similarity between two vectors"""
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
```

Cosine similarity measures the cosine of the angle between two vectors, resulting in a value between -1 and 1, where:
- 1 means the vectors are identical
- 0 means the vectors are orthogonal (unrelated)
- -1 means the vectors are exactly opposite

For embeddings, higher cosine similarity indicates that the texts are more semantically similar.

---

## 🧩 LLM-Enhanced Retrieval with Query Expansion

![Enhanced Retrieval](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

We can further enhance our vector database with LLM-powered query expansion:

```python
class EnhancedVectorDB(SimpleVectorDB):
    """Enhanced vector database with additional features like query expansion."""
    
    def expand_query(self, query):
        """Expand a query using LLM to improve search results"""
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
```

Query expansion uses the LLM to generate alternative phrasings of the original query, which can help overcome vocabulary mismatches and improve recall. For example, if the user searches for "heart attack symptoms," the LLM might expand this to include "signs of myocardial infarction" and "cardiac arrest warning signs."

### Searching with Query Expansion

We can then use these expanded queries to improve our search results:

```python
def search_with_expansion(self, query, top_k=5, filter_func=None):
    """Search with query expansion for better recall"""
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
```

This approach:
1. Generates multiple variations of the original query
2. Searches with each variation
3. Combines and deduplicates the results
4. Returns the top k most similar items

This can significantly improve recall, especially for queries that might use different terminology than what's in the database.

---

## 🧪 Advanced Features: Clustering and Labeling

![Clustering](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

Our enhanced vector database also includes advanced features like clustering and automatic labeling:

```python
def cluster_items(self, num_clusters=5):
    """Cluster items in the vector database"""
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
```

This function uses K-means clustering to group similar items together based on their embedding vectors. This can be useful for organizing large collections of documents or for discovering themes in a corpus.

### Automatic Cluster Labeling with LLM

We can also use the LLM to automatically generate labels for each cluster:

```python
def get_cluster_labels(self, clusters):
    """Generate labels for clusters using LLM"""
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
```

This function uses the LLM to analyze the content of each cluster and generate a descriptive label, making it easier to understand the themes in your data.

---

## 💪 Practice Exercises

![Practice](https://media.giphy.com/media/3oKIPrc2ngFZ6BTyww/giphy.gif)

1. **Implement a Hybrid Search System**:
   - Combine vector search with keyword search for better results
   - Use the LLM to determine when to use each approach
   - Implement a scoring system that weights both approaches

2. **Create a Document Chunking System**:
   - Implement a system that breaks documents into smaller chunks
   - Use the LLM to create semantically meaningful chunks
   - Store and retrieve chunks with appropriate context

3. **Build a Metadata-Enhanced Search**:
   - Extend the vector database to support rich metadata
   - Implement filters based on metadata attributes
   - Use the LLM to extract metadata from text automatically

---

## 🎯 Mini-Project Progress: Knowledge Base Assistant with Groq

![Knowledge Base](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

In this lesson, we've learned how to implement a vector database with real embeddings from the Groq API. This is a crucial component of our Knowledge Base Assistant:

- **Vector Storage**: Will store knowledge as embedding vectors
- **Semantic Search**: Will find relevant information based on meaning
- **Query Expansion**: Will improve recall by considering alternative phrasings
- **Clustering**: Will help organize and understand the knowledge base

In the next lesson, we'll explore retrieval patterns with LLM enhancement, which will enable our Knowledge Base Assistant to find the most relevant information for user queries.

---

## 📚 Resources

- [Groq API Documentation](https://console.groq.com/docs/quickstart)
- [Understanding Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Cosine Similarity Explained](https://www.machinelearningplus.com/nlp/cosine-similarity/)
