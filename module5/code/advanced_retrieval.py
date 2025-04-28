"""
Advanced Retrieval Strategies for RAG Systems

This module implements various advanced retrieval strategies that go beyond
basic vector search, including:
- Hybrid search (semantic + keyword)
- Multi-index retrieval
- Parent document retrieval
- Contextual compression

All implementations use LangChain Expression Language (LCEL) for improved
readability and composability.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np

# Updated LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.storage import InMemoryStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

try:
    from langchain_community.retrievers import ParentDocumentRetriever, MergerRetriever
    PARENT_RETRIEVER_AVAILABLE = True
except ImportError:
    PARENT_RETRIEVER_AVAILABLE = False


class HybridRetriever(BaseRetriever):
    """
    A retriever that combines semantic search with keyword search.

    Attributes:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        keyword_weight: Weight for keyword search (0-1)
        top_k: Number of results to return
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model: Embeddings,
        keyword_weight: float = 0.3,
        top_k: int = 5
    ):
        """Initialize the hybrid retriever."""
        super().__init__()
        self.documents = documents
        self.embedding_model = embedding_model
        self.keyword_weight = keyword_weight
        self.top_k = top_k

        # Create vector store
        self.vectorstore = FAISS.from_documents(documents, embedding_model)
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Create keyword retriever
        self.keyword_retriever = BM25Retriever.from_documents(documents)
        self.keyword_retriever.k = top_k

        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.keyword_retriever, self.vector_retriever],
            weights=[keyword_weight, 1 - keyword_weight]
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid search."""
        return self.ensemble_retriever.get_relevant_documents(query)

    def as_lcel_chain(self):
        """Return the retriever as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.get_relevant_documents(x["query"]))
        )


class MultiIndexRetriever(BaseRetriever):
    """
    A retriever that uses multiple specialized indices for different content types.

    Attributes:
        index_map: Dictionary mapping content types to retrievers
        default_retriever: Default retriever to use if no content type matches
        content_type_field: Metadata field to use for content type
    """

    def __init__(
        self,
        index_map: Dict[str, BaseRetriever],
        default_retriever: Optional[BaseRetriever] = None,
        content_type_field: str = "type"
    ):
        """Initialize the multi-index retriever."""
        super().__init__()
        self.index_map = index_map
        self.default_retriever = default_retriever
        self.content_type_field = content_type_field

        # Create merger retriever if available
        if PARENT_RETRIEVER_AVAILABLE:
            self.merger_retriever = MergerRetriever(
                retrievers=list(index_map.values())
            )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using multiple indices."""
        # Extract content type from query if possible
        content_type = self._extract_content_type(query)

        if content_type and content_type in self.index_map:
            # Use specialized retriever
            return self.index_map[content_type].get_relevant_documents(query)
        elif self.default_retriever:
            # Use default retriever
            return self.default_retriever.get_relevant_documents(query)
        else:
            # Use merger retriever
            if PARENT_RETRIEVER_AVAILABLE:
                return self.merger_retriever.get_relevant_documents(query)
            else:
                # Fallback: combine results from all retrievers
                all_docs = []
                for retriever in self.index_map.values():
                    all_docs.extend(retriever.get_relevant_documents(query))
                return all_docs[:5]  # Return top 5 results

    def _extract_content_type(self, query: str) -> Optional[str]:
        """Extract content type from query."""
        # Simple keyword-based extraction
        # In a real system, you might use an LLM to extract this
        query_lower = query.lower()
        for content_type in self.index_map.keys():
            if content_type.lower() in query_lower:
                return content_type
        return None

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding_models: Dict[str, Embeddings],
        content_type_field: str = "type",
        default_embedding_model: Optional[Embeddings] = None
    ) -> "MultiIndexRetriever":
        """Create a multi-index retriever from documents."""
        # Group documents by content type
        grouped_docs = {}
        for doc in documents:
            content_type = doc.metadata.get(content_type_field, "default")
            if content_type not in grouped_docs:
                grouped_docs[content_type] = []
            grouped_docs[content_type].append(doc)

        # Create retrievers for each content type
        retrievers = {}
        for content_type, docs in grouped_docs.items():
            if content_type in embedding_models:
                embedding_model = embedding_models[content_type]
            elif default_embedding_model:
                embedding_model = default_embedding_model
            else:
                continue

            vectorstore = FAISS.from_documents(docs, embedding_model)
            retrievers[content_type] = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Create default retriever if needed
        default_retriever = None
        if default_embedding_model and "default" in grouped_docs:
            vectorstore = FAISS.from_documents(grouped_docs["default"], default_embedding_model)
            default_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        return cls(retrievers, default_retriever, content_type_field)

    def as_lcel_chain(self):
        """Return the retriever as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.get_relevant_documents(x["query"]))
        )


class ParentDocRetriever:
    """
    A retriever that maintains document context by retrieving parent documents.

    Attributes:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        child_chunk_size: Size of child chunks for searching
        parent_chunk_size: Size of parent chunks for retrieval
        child_chunk_overlap: Overlap between child chunks
        parent_chunk_overlap: Overlap between parent chunks
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model: Embeddings,
        child_chunk_size: int = 500,
        parent_chunk_size: int = 2000,
        child_chunk_overlap: int = 50,
        parent_chunk_overlap: int = 200
    ):
        """Initialize the parent document retriever."""
        self.documents = documents
        self.embedding_model = embedding_model

        # Create text splitters
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap
        )

        # Create vector store
        self.vectorstore = FAISS.from_documents([], embedding_model)

        # Create document store
        self.doc_store = InMemoryStore()

        # Create parent document retriever if available
        if PARENT_RETRIEVER_AVAILABLE:
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.doc_store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )

            # Add documents
            self.retriever.add_documents(documents)
        else:
            # Fallback implementation if ParentDocumentRetriever is not available
            self._create_fallback_retriever(documents)

    def _create_fallback_retriever(self, documents: List[Document]):
        """Create a fallback implementation of parent document retrieval."""
        # Create parent and child documents
        self.parent_docs = []
        self.child_docs = []
        self.child_to_parent_map = {}

        for i, doc in enumerate(documents):
            # Create parent chunks
            parent_chunks = self.parent_splitter.split_documents([doc])
            self.parent_docs.extend(parent_chunks)

            # Create child chunks for each parent
            for j, parent in enumerate(parent_chunks):
                parent_id = f"parent_{i}_{j}"
                parent.metadata["parent_id"] = parent_id

                child_chunks = self.child_splitter.split_documents([parent])
                for k, child in enumerate(child_chunks):
                    child_id = f"child_{i}_{j}_{k}"
                    child.metadata["child_id"] = child_id
                    child.metadata["parent_id"] = parent_id
                    self.child_to_parent_map[child_id] = parent_id

                self.child_docs.extend(child_chunks)

        # Add child documents to vector store
        self.vectorstore = FAISS.from_documents(self.child_docs, self.embedding_model)

    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """Get relevant parent documents for a query."""
        if PARENT_RETRIEVER_AVAILABLE:
            return self.retriever.get_relevant_documents(query)
        else:
            # Fallback implementation
            # 1. Retrieve relevant child documents
            child_docs = self.vectorstore.similarity_search(query, k=top_k)

            # 2. Map to parent documents
            parent_ids = set()
            for doc in child_docs:
                parent_id = doc.metadata.get("parent_id")
                if parent_id:
                    parent_ids.add(parent_id)

            # 3. Return parent documents
            parent_docs = []
            for doc in self.parent_docs:
                if doc.metadata.get("parent_id") in parent_ids:
                    parent_docs.append(doc)

            return parent_docs[:top_k]

    def as_lcel_chain(self):
        """Return the retriever as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.get_relevant_documents(x["query"]))
        )


class ContextualCompressionRetrieverWrapper:
    """
    A retriever that filters irrelevant content from retrieved chunks.

    Attributes:
        base_retriever: Base retriever to use
        llm: Language model for compression
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: Optional[Any] = None
    ):
        """Initialize the contextual compression retriever."""
        self.base_retriever = base_retriever

        # Create LLM if not provided
        if llm is None:
            try:
                self.llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")
            except:
                # Fallback to a simple compressor
                self.compressor = self._create_simple_compressor()
                self.retriever = self._create_simple_compression_retriever()
                return
        else:
            self.llm = llm

        # Create document compressor
        self.compressor = LLMChainExtractor.from_llm(self.llm)

        # Create compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )

    def _create_simple_compressor(self):
        """Create a simple compressor that extracts relevant sentences."""
        class SimpleCompressor:
            def compress_documents(self, documents, query):
                compressed_docs = []
                for doc in documents:
                    # Simple relevance filtering based on sentence similarity
                    sentences = doc.page_content.split(". ")
                    relevant_sentences = []

                    for sentence in sentences:
                        # Simple keyword matching
                        if any(keyword in sentence.lower() for keyword in query.lower().split()):
                            relevant_sentences.append(sentence)

                    if relevant_sentences:
                        compressed_content = ". ".join(relevant_sentences) + "."
                        compressed_doc = Document(
                            page_content=compressed_content,
                            metadata=doc.metadata
                        )
                        compressed_docs.append(compressed_doc)
                    else:
                        # If no relevant sentences, keep the original
                        compressed_docs.append(doc)

                return compressed_docs

        return SimpleCompressor()

    def _create_simple_compression_retriever(self):
        """Create a simple compression retriever."""
        class SimpleCompressionRetriever:
            def __init__(self, base_retriever, compressor):
                self.base_retriever = base_retriever
                self.compressor = compressor

            def get_relevant_documents(self, query):
                # Get documents from base retriever
                docs = self.base_retriever.get_relevant_documents(query)

                # Compress documents
                compressed_docs = self.compressor.compress_documents(docs, query)

                return compressed_docs

        return SimpleCompressionRetriever(self.base_retriever, self.compressor)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant compressed documents for a query."""
        return self.retriever.get_relevant_documents(query)

    def as_lcel_chain(self):
        """Return the retriever as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.get_relevant_documents(x["query"]))
        )


class AdvancedRetrievalRouter:
    """
    A router that selects the appropriate retrieval strategy based on the query.

    Attributes:
        retrievers: Dictionary mapping retrieval strategies to retrievers
        router_function: Function to route queries to retrievers
    """

    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        router_function: Optional[Callable[[str], str]] = None
    ):
        """Initialize the advanced retrieval router."""
        self.retrievers = retrievers
        self.router_function = router_function or self._default_router

    def _default_router(self, query: str) -> str:
        """Default router function based on query keywords."""
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in ["exact", "keyword", "specific"]):
            return "hybrid"
        elif any(keyword in query_lower for keyword in ["context", "full", "document"]):
            return "parent"
        elif any(keyword in query_lower for keyword in ["compress", "relevant", "extract"]):
            return "compression"
        elif any(keyword in query_lower for keyword in ["technical", "general", "specialized"]):
            return "multi_index"
        else:
            return "default"

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using the appropriate retrieval strategy."""
        strategy = self.router_function(query)

        if strategy in self.retrievers:
            return self.retrievers[strategy].get_relevant_documents(query)
        elif "default" in self.retrievers:
            return self.retrievers["default"].get_relevant_documents(query)
        else:
            # Fallback to the first retriever
            return list(self.retrievers.values())[0].get_relevant_documents(query)

    def as_lcel_chain(self):
        """Return the router as an LCEL chain."""
        router_chain = (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.router_function(x["query"]))
        )

        retrieval_chain = (
            {"query": RunnablePassthrough(), "strategy": router_chain}
            | RunnableLambda(lambda x: self.retrievers.get(
                x["strategy"],
                self.retrievers.get("default", list(self.retrievers.values())[0])
            ).get_relevant_documents(x["query"]))
        )

        return retrieval_chain


# Example usage
def create_advanced_retrieval_system(
    documents: List[Document],
    embedding_model: Embeddings,
    llm: Optional[Any] = None
) -> AdvancedRetrievalRouter:
    """
    Create a complete advanced retrieval system with multiple strategies.

    Args:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        llm: Optional language model for compression

    Returns:
        An advanced retrieval router
    """
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        documents=documents,
        embedding_model=embedding_model,
        keyword_weight=0.3
    )

    # Create parent document retriever
    parent_retriever = ParentDocRetriever(
        documents=documents,
        embedding_model=embedding_model
    )

    # Create compression retriever
    compression_retriever = ContextualCompressionRetrieverWrapper(
        base_retriever=hybrid_retriever,
        llm=llm
    )

    # Create multi-index retriever
    # Group documents by type
    technical_docs = [doc for doc in documents if doc.metadata.get("type") == "technical"]
    general_docs = [doc for doc in documents if doc.metadata.get("type") == "general"]

    if technical_docs and general_docs:
        technical_vectorstore = FAISS.from_documents(technical_docs, embedding_model)
        general_vectorstore = FAISS.from_documents(general_docs, embedding_model)

        technical_retriever = technical_vectorstore.as_retriever(search_kwargs={"k": 3})
        general_retriever = general_vectorstore.as_retriever(search_kwargs={"k": 3})

        multi_index_retriever = MultiIndexRetriever(
            index_map={
                "technical": technical_retriever,
                "general": general_retriever
            }
        )
    else:
        # Fallback to hybrid retriever if document types are not available
        multi_index_retriever = hybrid_retriever

    # Create advanced retrieval router
    router = AdvancedRetrievalRouter(
        retrievers={
            "hybrid": hybrid_retriever,
            "parent": parent_retriever,
            "compression": compression_retriever,
            "multi_index": multi_index_retriever,
            "default": hybrid_retriever
        }
    )

    return router
