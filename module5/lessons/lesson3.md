# 🔄 Module 5: Advanced RAG Systems - Lesson 3: Reranking and Result Optimization 🚀

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧠 Understand why initial retrieval results often need optimization
- 🔍 Implement cross-encoder rerankers for improved relevance scoring
- 🔄 Build reciprocal rank fusion systems to combine multiple result sets
- 📊 Create maximal marginal relevance reranking for diversity
- 📝 Develop source attribution mechanisms for transparency
- 🔗 Integrate these techniques using LCEL

---

## 🧠 The Reranking Challenge in RAG Systems

<img src="https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif" width="50%" height="50%"/>

### Why Initial Retrieval Results Need Optimization

Initial retrieval results, even from advanced retrieval systems, often have limitations:

1. **Embedding Limitations**: Vector similarity doesn't always correlate with true relevance
2. **Lack of Diversity**: Top results may be too similar to each other
3. **Missing Context**: Relevance depends on the specific question, not just general similarity
4. **Transparency Issues**: Users need to understand why results were selected
5. **Ranking Inconsistency**: Different retrieval methods may produce conflicting rankings

### The Reranking Landscape

To address these limitations, we can implement several reranking strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| Cross-Encoder Reranking | Uses transformer models to score query-document pairs | High-precision requirements |
| Reciprocal Rank Fusion | Combines rankings from multiple retrievers | Ensemble retrieval systems |
| Maximal Marginal Relevance | Balances relevance with diversity | Comprehensive coverage |
| Source Attribution | Tracks and explains document sources | Transparency and trust |
| Contextual Reranking | Considers broader context for scoring | Complex, nuanced queries |

---

## 🔍 Cross-Encoder Reranking: Beyond Vector Similarity

Cross-encoder rerankers provide more accurate relevance scoring than initial retrieval:

### How Cross-Encoder Reranking Works

1. **Initial Retrieval**: Get candidate documents using fast retrieval methods
2. **Pair Creation**: Form query-document pairs for each candidate
3. **Cross-Encoder Scoring**: Score each pair using a transformer model
4. **Reranking**: Sort documents by their cross-encoder scores

### Bi-Encoders vs. Cross-Encoders

| Aspect | Bi-Encoders (Initial Retrieval) | Cross-Encoders (Reranking) |
|--------|----------------------------------|----------------------------|
| Architecture | Encode query and documents separately | Process query and document together |
| Speed | Fast (vector comparison) | Slower (full attention mechanism) |
| Accuracy | Lower (no direct comparison) | Higher (direct comparison) |
| Scalability | High (pre-compute embeddings) | Low (compute at query time) |
| Use Case | Initial retrieval of candidates | Reranking a small set of candidates |

### Implementing Cross-Encoder Reranking with LCEL

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from sentence_transformers import CrossEncoder

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Create cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Create reranker function
def rerank_with_cross_encoder(query_and_docs):
    query = query_and_docs["query"]
    docs = query_and_docs["docs"]
    
    # Create query-document pairs
    pairs = [[query, doc.page_content] for doc in docs]
    
    # Score pairs with cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Create scored documents
    scored_docs = list(zip(docs, scores))
    
    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return reranked documents
    return [doc for doc, _ in scored_docs[:10]]

# Create LCEL chain
reranking_chain = (
    {
        "query": RunnablePassthrough(),
        "docs": base_retriever
    }
    | RunnableLambda(rerank_with_cross_encoder)
)
```

---

## 🔄 Reciprocal Rank Fusion: Combining Multiple Rankings

Reciprocal Rank Fusion (RRF) combines rankings from multiple retrievers:

### How Reciprocal Rank Fusion Works

1. **Multiple Retrievers**: Get rankings from different retrieval methods
2. **RRF Score Calculation**: Calculate RRF score for each document
3. **Fusion**: Combine scores and sort documents

The RRF score for a document is calculated as:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where:
- $R$ is the set of all rankings
- $\text{rank}_r(d)$ is the rank of document $d$ in ranking $r$
- $k$ is a constant (typically 60) that prevents very high scores for top-ranked documents

### Implementing Reciprocal Rank Fusion with LCEL

```python
from langchain.retrievers import EnsembleRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Create multiple retrievers
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
keyword_retriever = BM25Retriever.from_documents(documents, k=10)
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)

# Create RRF function
def reciprocal_rank_fusion(query_and_retrievers):
    query = query_and_retrievers["query"]
    retrievers = query_and_retrievers["retrievers"]
    k = 60  # RRF constant
    
    # Get results from each retriever
    all_results = {}
    for retriever_name, retriever in retrievers.items():
        results = retriever.get_relevant_documents(query)
        for rank, doc in enumerate(results):
            doc_id = hash(doc.page_content)  # Use content hash as ID
            if doc_id not in all_results:
                all_results[doc_id] = {"doc": doc, "score": 0}
            # Add RRF score: 1 / (k + rank)
            all_results[doc_id]["score"] += 1 / (k + rank)
    
    # Sort by RRF score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    # Return reranked documents
    return [item["doc"] for item in sorted_results[:10]]

# Create LCEL chain
rrf_chain = (
    {
        "query": RunnablePassthrough(),
        "retrievers": lambda _: {
            "semantic": semantic_retriever,
            "keyword": keyword_retriever,
            "hybrid": hybrid_retriever
        }
    }
    | RunnableLambda(reciprocal_rank_fusion)
)
```

---

## 📊 Maximal Marginal Relevance: Balancing Relevance with Diversity

Maximal Marginal Relevance (MMR) reranking balances relevance with diversity:

### How MMR Works

1. **Initial Retrieval**: Get candidate documents using standard methods
2. **Iterative Selection**: Select documents one by one, considering both:
   - Relevance to the query
   - Diversity compared to already selected documents
3. **Balance Parameter**: Adjust the balance between relevance and diversity

The MMR score for a document is calculated as:

$$\text{MMR} = \lambda \cdot \text{sim}(d, q) - (1 - \lambda) \cdot \max_{d_j \in S} \text{sim}(d, d_j)$$

Where:
- $\text{sim}(d, q)$ is the similarity between document $d$ and query $q$
- $\text{sim}(d, d_j)$ is the similarity between document $d$ and already selected document $d_j$
- $S$ is the set of already selected documents
- $\lambda$ is a parameter between 0 and 1 that controls the trade-off

### Implementing MMR with LCEL

```python
from langchain.retrievers import EnsembleRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import numpy as np

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Create MMR function
def mmr_reranking(query_and_docs):
    query = query_and_docs["query"]
    docs = query_and_docs["docs"]
    embedding_model = query_and_docs["embedding_model"]
    lambda_param = 0.7  # Balance between relevance and diversity
    
    # Get embeddings
    query_embedding = embedding_model.embed_query(query)
    doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in docs]
    
    # Calculate relevance scores (similarity to query)
    relevance_scores = [np.dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    
    # Initialize selected documents
    selected_indices = []
    remaining_indices = list(range(len(docs)))
    
    # Select documents iteratively
    while len(selected_indices) < min(10, len(docs)):
        # Calculate MMR scores
        mmr_scores = []
        for i in remaining_indices:
            # Relevance component
            relevance = relevance_scores[i]
            
            # Diversity component
            if not selected_indices:
                diversity = 0
            else:
                # Maximum similarity to any already selected document
                similarities = [np.dot(doc_embeddings[i], doc_embeddings[j]) 
                               for j in selected_indices]
                diversity = max(similarities)
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((i, mmr_score))
        
        # Select document with highest MMR score
        selected_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(selected_idx)
        remaining_indices.remove(selected_idx)
    
    # Return reranked documents
    return [docs[i] for i in selected_indices]

# Create LCEL chain
mmr_chain = (
    {
        "query": RunnablePassthrough(),
        "docs": base_retriever,
        "embedding_model": lambda _: embedding_model
    }
    | RunnableLambda(mmr_reranking)
)
```

---

## 📝 Source Attribution: Transparency in RAG Systems

Source attribution tracks and explains document sources:

### Why Source Attribution Matters

1. **Transparency**: Users understand where information comes from
2. **Trust**: Attribution builds trust in the system's responses
3. **Verification**: Users can verify information at the source
4. **Compliance**: Many domains require proper citation
5. **Debugging**: Helps identify issues in the retrieval process

### Implementing Source Attribution with LCEL

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create source attribution function
def add_source_attribution(docs):
    attributed_docs = []
    for doc in docs:
        # Extract source information
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        author = doc.metadata.get("author", "")
        date = doc.metadata.get("date", "")
        
        # Create attribution string
        attribution = f"Source: {source}"
        if page:
            attribution += f", Page: {page}"
        if author:
            attribution += f", Author: {author}"
        if date:
            attribution += f", Date: {date}"
        
        # Add attribution to document
        doc.metadata["attribution"] = attribution
        attributed_docs.append(doc)
    
    return attributed_docs

# Create prompt template
prompt_template = """Answer the following question based on the provided context.
Include source attribution for each piece of information in your answer.

Question: {question}

Context:
{context}

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create LLM
llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")

# Create LCEL chain
attribution_chain = (
    {
        "question": RunnablePassthrough(),
        "context": base_retriever | RunnableLambda(add_source_attribution)
    }
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "context": "\n\n".join([
            f"{doc.page_content}\n{doc.metadata['attribution']}" 
            for doc in x["context"]
        ])
    })
    | prompt
    | llm
)
```

---

## 🔗 Combining Reranking Strategies with LCEL

One of the strengths of LCEL is the ability to combine multiple reranking strategies:

### Creating a Multi-Strategy Reranker

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch

# Define a function to route queries to different rerankers
def route_query(query):
    if "diverse" in query.lower():
        return "mmr"
    elif "accurate" in query.lower():
        return "cross_encoder"
    elif "multiple sources" in query.lower():
        return "rrf"
    else:
        return "default"

# Create a dictionary of rerankers
rerankers = {
    "cross_encoder": lambda x: rerank_with_cross_encoder(x),
    "rrf": lambda x: reciprocal_rank_fusion(x),
    "mmr": lambda x: mmr_reranking(x),
    "default": lambda x: rerank_with_cross_encoder(x)  # Default to cross-encoder
}

# Create a routing chain
router_chain = RunnableBranch(
    (lambda x: route_query(x["query"]) == "mmr", rerankers["mmr"]),
    (lambda x: route_query(x["query"]) == "cross_encoder", rerankers["cross_encoder"]),
    (lambda x: route_query(x["query"]) == "rrf", rerankers["rrf"]),
    rerankers["default"]  # Default
)

# Create the final reranking chain
reranking_chain = (
    {
        "query": RunnablePassthrough(),
        "docs": base_retriever,
        "embedding_model": lambda _: embedding_model,
        "retrievers": lambda _: {
            "semantic": semantic_retriever,
            "keyword": keyword_retriever,
            "hybrid": hybrid_retriever
        }
    }
    | router_chain
)
```

---

## 💪 Practice Exercises

1. **Implement a Cross-Encoder Reranker**: Create a cross-encoder reranker using a pre-trained model from Hugging Face.

2. **Build a Reciprocal Rank Fusion System**: Implement a reciprocal rank fusion system that combines results from multiple retrievers.

3. **Create a Maximal Marginal Relevance Reranker**: Build a maximal marginal relevance reranker that balances relevance with diversity.

4. **Develop a Source Attribution System**: Implement a source attribution system that tracks and explains document sources.

5. **Combine Multiple Reranking Strategies**: Create an advanced reranking system that combines multiple strategies based on query type.

6. **Implement Reranking with LCEL**: Build a complete reranking system using LCEL for improved readability and composability.

---

## 🔍 Key Takeaways

1. **Beyond Initial Retrieval**: Reranking is essential for optimizing initial retrieval results.

2. **Cross-Encoder Advantage**: Cross-encoders provide more accurate relevance scoring than bi-encoders.

3. **Ensemble Methods**: Reciprocal rank fusion combines strengths of multiple retrieval methods.

4. **Diversity Matters**: Maximal marginal relevance balances relevance with diversity.

5. **Transparency**: Source attribution builds trust and enables verification.

6. **LCEL Integration**: All these strategies can be elegantly implemented and combined using LCEL.

---

## 📚 Resources

- [LangChain Rerankers Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/)
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Maximal Marginal Relevance Paper](https://dl.acm.org/doi/10.1145/290941.291025)
- [LCEL Reranking Patterns](https://python.langchain.com/docs/expression_language/cookbook/retrieval)

---

## 🚀 Next Steps

In the next lesson, we'll explore self-querying and adaptive RAG techniques that can dynamically adjust retrieval strategies based on query type and content.
