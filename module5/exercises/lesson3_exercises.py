"""
Exercises for Module 5 - Lesson 3: Reranking and Result Optimization

This module contains exercises for implementing various reranking strategies
to optimize retrieval results, including:
- Cross-encoder reranking
- Reciprocal rank fusion
- Maximal marginal relevance
- Source attribution
- Combined reranking strategies
"""

from typing import List, Dict, Any
import numpy as np
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch

# Check if LangChain is available
try:
    from langchain.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Check if sentence-transformers is available
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Exercise 1: Implement a cross-encoder reranker
def exercise1_cross_encoder_reranker(
    query: str,
    documents: List[Document],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5
) -> List[Document]:
    """
    Exercise 1: Implement a cross-encoder reranker that scores query-document pairs.

    Args:
        query: User query
        documents: List of documents to rerank
        model_name: Name of the cross-encoder model
        top_k: Number of documents to return

    Returns:
        Reranked list of documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required for this exercise. Install with 'pip install sentence-transformers'")

    # TODO: Implement cross-encoder reranking
    # 1. Create a cross-encoder model
    # 2. Create query-document pairs
    # 3. Score pairs with the cross-encoder
    # 4. Sort documents by score
    # 5. Return top_k documents

    # Create cross-encoder model
    cross_encoder = CrossEncoder(model_name)

    # Create query-document pairs
    pairs = [[query, doc.page_content] for doc in documents]

    # Score pairs with cross-encoder
    scores = cross_encoder.predict(pairs)

    # Create scored documents
    scored_docs = list(zip(documents, scores))

    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Add scores to metadata
    for doc, score in scored_docs[:top_k]:
        doc.metadata["cross_encoder_score"] = float(score)

    # Return reranked documents
    return [doc for doc, _ in scored_docs[:top_k]]


# Exercise 2: Implement reciprocal rank fusion
def exercise2_reciprocal_rank_fusion(
    query: str,
    retrievers: Dict[str, BaseRetriever],
    k: int = 60,
    top_k: int = 5
) -> List[Document]:
    """
    Exercise 2: Implement reciprocal rank fusion to combine results from multiple retrievers.

    Args:
        query: User query
        retrievers: Dictionary of retrievers to combine
        k: RRF constant
        top_k: Number of documents to return

    Returns:
        Combined list of documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # TODO: Implement reciprocal rank fusion
    # 1. Get results from each retriever
    # 2. Calculate RRF score for each document
    # 3. Combine scores and sort documents
    # 4. Return top_k documents

    # Get results from each retriever
    all_results = {}
    for retriever_name, retriever in retrievers.items():
        results = retriever.get_relevant_documents(query)
        for rank, doc in enumerate(results):
            # Use content hash as ID
            doc_id = hash(doc.page_content)

            if doc_id not in all_results:
                all_results[doc_id] = {"doc": doc, "score": 0, "sources": []}

            # Add RRF score: 1 / (k + rank)
            rrf_score = 1 / (k + rank)
            all_results[doc_id]["score"] += rrf_score
            all_results[doc_id]["sources"].append(retriever_name)

    # Sort by RRF score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Add scores and sources to metadata
    for item in sorted_results[:top_k]:
        item["doc"].metadata["rrf_score"] = item["score"]
        item["doc"].metadata["retrieval_sources"] = item["sources"]

    # Return reranked documents
    return [item["doc"] for item in sorted_results[:top_k]]


# Exercise 3: Implement maximal marginal relevance
def exercise3_maximal_marginal_relevance(
    query: str,
    documents: List[Document],
    embedding_model: Any,
    lambda_param: float = 0.7,
    top_k: int = 5
) -> List[Document]:
    """
    Exercise 3: Implement maximal marginal relevance reranking for diversity.

    Args:
        query: User query
        documents: List of documents to rerank
        embedding_model: Model to generate embeddings
        lambda_param: Balance between relevance and diversity (0-1)
        top_k: Number of documents to return

    Returns:
        Reranked list of documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # TODO: Implement maximal marginal relevance
    # 1. Get embeddings for query and documents
    # 2. Calculate relevance scores
    # 3. Iteratively select documents based on MMR
    # 4. Return selected documents

    if not documents:
        return []

    # Get embeddings
    query_embedding = embedding_model.embed_query(query)
    doc_embeddings = [
        embedding_model.embed_query(doc.page_content)
        for doc in documents
    ]

    # Calculate relevance scores (similarity to query)
    relevance_scores = [
        np.dot(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    # Initialize selected documents
    selected_indices = []
    remaining_indices = list(range(len(documents)))

    # Select documents iteratively
    while len(selected_indices) < min(top_k, len(documents)):
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
                similarities = [
                    np.dot(doc_embeddings[i], doc_embeddings[j])
                    for j in selected_indices
                ]
                diversity = max(similarities)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((i, mmr_score))

        # Select document with highest MMR score
        selected_idx, score = max(mmr_scores, key=lambda x: x[1])

        # Add MMR score to metadata
        documents[selected_idx].metadata["mmr_score"] = float(score)
        documents[selected_idx].metadata["mmr_rank"] = len(selected_indices) + 1

        selected_indices.append(selected_idx)
        remaining_indices.remove(selected_idx)

    # Return reranked documents
    return [documents[i] for i in selected_indices]


# Exercise 4: Implement source attribution
def exercise4_source_attribution(
    documents: List[Document],
    include_metadata_fields: List[str] = None
) -> List[Document]:
    """
    Exercise 4: Implement a source attribution system that tracks document sources.

    Args:
        documents: List of documents to add attribution to
        include_metadata_fields: List of metadata fields to include in attribution

    Returns:
        Documents with attribution added
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # TODO: Implement source attribution
    # 1. Extract source information from metadata
    # 2. Create attribution string
    # 3. Add attribution to document metadata
    # 4. Return documents with attribution

    # Default metadata fields to include
    include_metadata_fields = include_metadata_fields or ["source", "page", "author", "date"]

    attributed_docs = []
    for doc in documents:
        # Extract metadata fields
        metadata_values = {}
        for field in include_metadata_fields:
            metadata_values[field] = doc.metadata.get(field, "")

        # Create attribution string
        attribution_parts = []
        for field, value in metadata_values.items():
            if value:
                attribution_parts.append(f"{field.capitalize()}: {value}")

        attribution = ", ".join(attribution_parts)

        # Add attribution to metadata
        doc.metadata["attribution"] = attribution
        attributed_docs.append(doc)

    return attributed_docs


# Exercise 5: Implement a combined reranking system
def exercise5_combined_reranking(
    query: str,
    documents: List[Document],
    embedding_model: Any,
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> List[Document]:
    """
    Exercise 5: Implement a combined reranking system that selects the appropriate strategy.

    Args:
        query: User query
        documents: List of documents to rerank
        embedding_model: Model to generate embeddings
        cross_encoder_model_name: Name of the cross-encoder model

    Returns:
        Reranked list of documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required for this exercise. Install with 'pip install sentence-transformers'")

    # TODO: Implement combined reranking
    # 1. Determine which reranking strategy to use based on the query
    # 2. Apply the appropriate reranking strategy
    # 3. Return reranked documents

    # Determine which reranking strategy to use
    if "diverse" in query.lower() or "variety" in query.lower():
        # Use MMR for queries asking for diverse results
        reranked_docs = exercise3_maximal_marginal_relevance(
            query, documents, embedding_model, lambda_param=0.7, top_k=5
        )
    elif "accurate" in query.lower() or "precise" in query.lower():
        # Use cross-encoder for queries asking for accurate results
        reranked_docs = exercise1_cross_encoder_reranker(
            query, documents, model_name=cross_encoder_model_name, top_k=5
        )
    else:
        # Default to cross-encoder
        reranked_docs = exercise1_cross_encoder_reranker(
            query, documents, model_name=cross_encoder_model_name, top_k=5
        )

    # Add source attribution
    reranked_docs = exercise4_source_attribution(reranked_docs)

    return reranked_docs


# Exercise 6: Implement reranking with LCEL
def exercise6_lcel_reranking(
    documents: List[Document],
    embedding_model: Any,
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> Any:
    """
    Exercise 6: Implement a complete reranking system using LCEL.

    Args:
        documents: List of documents to index
        embedding_model: Model to generate embeddings
        cross_encoder_model_name: Name of the cross-encoder model

    Returns:
        LCEL chain for reranking
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required for this exercise. Install with 'pip install sentence-transformers'")

    # TODO: Implement LCEL reranking chain
    # 1. Create a base retriever
    # 2. Create reranking functions
    # 3. Create a router function
    # 4. Create an LCEL chain that routes queries to the appropriate reranker
    # 5. Return the LCEL chain

    # Create a simple vector store for testing
    from langchain.vectorstores import FAISS
    vectorstore = FAISS.from_documents(documents, embedding_model)

    # Create base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Create cross-encoder model
    cross_encoder = CrossEncoder(cross_encoder_model_name)

    # Create reranking functions
    def cross_encoder_rerank(query_and_docs):
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

        # Add scores to metadata
        for doc, score in scored_docs[:5]:
            doc.metadata["cross_encoder_score"] = float(score)

        # Return reranked documents
        return [doc for doc, _ in scored_docs[:5]]

    def mmr_rerank(query_and_docs):
        query = query_and_docs["query"]
        docs = query_and_docs["docs"]

        return exercise3_maximal_marginal_relevance(
            query, docs, embedding_model, lambda_param=0.7, top_k=5
        )

    def add_attribution(docs):
        return exercise4_source_attribution(docs)

    # Create router function
    def route_query(query):
        if "diverse" in query.lower() or "variety" in query.lower():
            return "mmr"
        else:
            return "cross_encoder"

    # Create LCEL chain
    retrieval_chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"query": x, "docs": base_retriever.get_relevant_documents(x)})
    )

    reranking_chain = (
        retrieval_chain
        | RunnableBranch(
            (lambda x: route_query(x["query"]) == "mmr", RunnableLambda(mmr_rerank)),
            RunnableLambda(cross_encoder_rerank)
        )
        | RunnableLambda(add_attribution)
    )

    return reranking_chain


# Example usage
if __name__ == "__main__":
    print("Lesson 3 Exercises: Reranking and Result Optimization")

    # Sample documents
    documents = [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs with external knowledge.",
            metadata={"source": "RAG Paper", "author": "Meta AI", "date": "2023-01-01"}
        ),
        Document(
            page_content="Vector databases store embeddings for efficient similarity search.",
            metadata={"source": "Vector DB Guide", "author": "Database Experts", "date": "2023-02-15"}
        ),
        Document(
            page_content="Cross-encoders provide more accurate relevance scoring than bi-encoders.",
            metadata={"source": "Reranking Paper", "author": "NLP Researchers", "date": "2023-03-10"}
        ),
        Document(
            page_content="Maximal Marginal Relevance (MMR) balances relevance with diversity in search results.",
            metadata={"source": "Search Algorithms", "author": "IR Experts", "date": "2023-04-20"}
        ),
        Document(
            page_content="Reciprocal Rank Fusion combines rankings from multiple retrieval systems.",
            metadata={"source": "Fusion Techniques", "author": "Search Engineers", "date": "2023-05-05"}
        )
    ]

    print("These exercises require additional dependencies:")
    print("- langchain: pip install langchain")
    print("- sentence-transformers: pip install sentence-transformers")
    print("- numpy: pip install numpy")
    print("- faiss-cpu: pip install faiss-cpu")

    print("\nComplete the exercises by implementing the TODO sections in each function.")
