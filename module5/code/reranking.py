"""
Reranking and Result Optimization for RAG Systems

This module implements various reranking strategies that optimize initial
retrieval results, including:
- Cross-encoder reranking
- Reciprocal rank fusion
- Maximal marginal relevance
- Source attribution

All implementations use LangChain Expression Language (LCEL) for improved
readability and composability.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np

# Updated LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_groq import ChatGroq

# Check if sentence-transformers is available
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class BaseReranker:
    """Base class for rerankers."""

    def __init__(self, name: str = "base_reranker"):
        """Initialize the reranker.

        Args:
            name: Name of the reranker
        """
        self.name = name

    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents based on the query.

        Args:
            query: User query
            documents: List of documents to rerank
            **kwargs: Additional arguments

        Returns:
            Reranked list of documents
        """
        # Base implementation just returns the original documents
        return documents

    def as_lcel_chain(self):
        """Return the reranker as an LCEL chain."""
        return (
            {"query": RunnablePassthrough(), "documents": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.rerank(x["query"], x["documents"]))
        )


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        name: str = "cross_encoder_reranker"
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            top_k: Number of documents to return
            name: Name of the reranker
        """
        super().__init__(name)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with 'pip install sentence-transformers'."
            )

        self.model_name = model_name
        self.top_k = top_k
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents using the cross-encoder model.

        Args:
            query: User query
            documents: List of documents to rerank
            **kwargs: Additional arguments

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Score pairs with cross-encoder
        scores = self.cross_encoder.predict(pairs)

        # Create scored documents
        scored_docs = list(zip(documents, scores))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Add scores to metadata
        for doc, score in scored_docs[:self.top_k]:
            doc.metadata["cross_encoder_score"] = float(score)

        # Return reranked documents
        return [doc for doc, _ in scored_docs[:self.top_k]]

    def as_lcel_chain(self):
        """Return the cross-encoder reranker as an LCEL chain."""
        return (
            {"query": RunnablePassthrough(), "documents": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.rerank(x["query"], x["documents"]))
        )


class ReciprocalRankFusion(BaseReranker):
    """Reranker using reciprocal rank fusion."""

    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        k: int = 60,
        top_k: int = 10,
        name: str = "reciprocal_rank_fusion"
    ):
        """Initialize the reciprocal rank fusion reranker.

        Args:
            retrievers: Dictionary of retrievers to combine
            k: RRF constant
            top_k: Number of documents to return
            name: Name of the reranker
        """
        super().__init__(name)
        self.retrievers = retrievers
        self.k = k
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document] = None, **kwargs) -> List[Document]:
        """Rerank documents using reciprocal rank fusion.

        Args:
            query: User query
            documents: Optional list of documents (not used in this implementation)
            **kwargs: Additional arguments

        Returns:
            Reranked list of documents
        """
        # Get results from each retriever
        all_results = {}
        for retriever_name, retriever in self.retrievers.items():
            results = retriever.get_relevant_documents(query)
            for rank, doc in enumerate(results):
                # Use content hash as ID
                doc_id = hash(doc.page_content)

                if doc_id not in all_results:
                    all_results[doc_id] = {"doc": doc, "score": 0, "sources": []}

                # Add RRF score: 1 / (k + rank)
                rrf_score = 1 / (self.k + rank)
                all_results[doc_id]["score"] += rrf_score
                all_results[doc_id]["sources"].append(retriever_name)

        # Sort by RRF score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # Add scores and sources to metadata
        for item in sorted_results[:self.top_k]:
            item["doc"].metadata["rrf_score"] = item["score"]
            item["doc"].metadata["retrieval_sources"] = item["sources"]

        # Return reranked documents
        return [item["doc"] for item in sorted_results[:self.top_k]]

    def as_lcel_chain(self):
        """Return the RRF reranker as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.rerank(x["query"]))
        )


class MaximalMarginalRelevance(BaseReranker):
    """Reranker using maximal marginal relevance."""

    def __init__(
        self,
        embedding_model: Embeddings,
        lambda_param: float = 0.7,
        top_k: int = 10,
        name: str = "maximal_marginal_relevance"
    ):
        """Initialize the MMR reranker.

        Args:
            embedding_model: Model to generate embeddings
            lambda_param: Balance between relevance and diversity (0-1)
            top_k: Number of documents to return
            name: Name of the reranker
        """
        super().__init__(name)
        self.embedding_model = embedding_model
        self.lambda_param = lambda_param
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents using maximal marginal relevance.

        Args:
            query: User query
            documents: List of documents to rerank
            **kwargs: Additional arguments

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        # Get embeddings
        query_embedding = self.embedding_model.embed_query(query)
        doc_embeddings = [
            self.embedding_model.embed_query(doc.page_content)
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
        while len(selected_indices) < min(self.top_k, len(documents)):
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
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * diversity
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

    def as_lcel_chain(self):
        """Return the MMR reranker as an LCEL chain."""
        return (
            {"query": RunnablePassthrough(), "documents": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.rerank(x["query"], x["documents"]))
        )


class SourceAttributionSystem:
    """System for adding source attribution to documents."""

    def __init__(
        self,
        include_metadata_fields: List[str] = None,
        attribution_format: str = "Source: {source}, Page: {page}, Author: {author}, Date: {date}"
    ):
        """Initialize the source attribution system.

        Args:
            include_metadata_fields: List of metadata fields to include in attribution
            attribution_format: Format string for attribution
        """
        self.include_metadata_fields = include_metadata_fields or ["source", "page", "author", "date"]
        self.attribution_format = attribution_format

    def add_attribution(self, documents: List[Document]) -> List[Document]:
        """Add source attribution to documents.

        Args:
            documents: List of documents to add attribution to

        Returns:
            Documents with attribution added
        """
        attributed_docs = []
        for doc in documents:
            # Extract metadata fields
            metadata_values = {}
            for field in self.include_metadata_fields:
                metadata_values[field] = doc.metadata.get(field, "")

            # Create attribution string
            try:
                attribution = self.attribution_format.format(**metadata_values)
                # Remove empty fields
                attribution = ", ".join([part for part in attribution.split(", ") if ":" not in part or part.split(":")[1].strip()])
            except KeyError:
                # Fallback if format string contains fields not in metadata_values
                attribution = ", ".join([f"{field}: {value}" for field, value in metadata_values.items() if value])

            # Add attribution to metadata
            doc.metadata["attribution"] = attribution
            attributed_docs.append(doc)

        return attributed_docs

    def as_lcel_chain(self):
        """Return the source attribution system as an LCEL chain."""
        return RunnableLambda(self.add_attribution)


class CombinedReranker(BaseReranker):
    """Reranker that combines multiple reranking strategies."""

    def __init__(
        self,
        rerankers: Dict[str, BaseReranker],
        router_function: Callable[[str], str] = None,
        default_reranker: str = None,
        name: str = "combined_reranker"
    ):
        """Initialize the combined reranker.

        Args:
            rerankers: Dictionary of rerankers
            router_function: Function that routes queries to rerankers
            default_reranker: Name of the default reranker
            name: Name of the reranker
        """
        super().__init__(name)
        self.rerankers = rerankers
        self.router_function = router_function or self._default_router
        self.default_reranker = default_reranker or next(iter(rerankers.keys()))

    def _default_router(self, query: str) -> str:
        """Default router function.

        Args:
            query: User query

        Returns:
            Name of the reranker to use
        """
        if "diverse" in query.lower():
            return "mmr"
        elif "accurate" in query.lower():
            return "cross_encoder"
        elif "multiple sources" in query.lower() or "combine" in query.lower():
            return "rrf"
        else:
            return self.default_reranker

    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents using the appropriate strategy.

        Args:
            query: User query
            documents: List of documents to rerank
            **kwargs: Additional arguments

        Returns:
            Reranked list of documents
        """
        # Determine which reranker to use
        reranker_name = self.router_function(query)

        # Get the reranker
        reranker = self.rerankers.get(reranker_name, self.rerankers[self.default_reranker])

        # Rerank documents
        return reranker.rerank(query, documents, **kwargs)

    def as_lcel_chain(self):
        """Return the combined reranker as an LCEL chain."""
        # Create a branch for each reranker
        branches = []
        for name, reranker in self.rerankers.items():
            branches.append(
                (lambda x, name=name: self.router_function(x["query"]) == name,
                 RunnableLambda(lambda x, r=reranker: r.rerank(x["query"], x["documents"])))
            )

        # Add default branch
        default_reranker = self.rerankers[self.default_reranker]
        branches.append(
            RunnableLambda(lambda x: default_reranker.rerank(x["query"], x["documents"]))
        )

        # Create the branch chain
        branch_chain = RunnableBranch(*branches)

        # Create the final chain
        return (
            {"query": RunnablePassthrough(), "documents": RunnablePassthrough()}
            | branch_chain
        )


# Example usage
if __name__ == "__main__":
    print("Reranking and Result Optimization for RAG Systems")

    # This is just an example and won't run without the necessary dependencies
    # and document data. See the lesson3_exercises.py file for working examples.
