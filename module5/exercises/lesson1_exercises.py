"""
Module 5 - Lesson 1 Exercises: Advanced Retrieval Strategies

This module contains exercises for implementing advanced retrieval strategies
including hybrid search, multi-index retrieval, parent document retrieval,
and contextual compression.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Check if LangChain is available
try:
    from langchain.retrievers import BM25Retriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Check if Groq is available
try:
    from langchain.chat_models import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class SimpleLLMClient:
    """Simple LLM client for when Groq is not available."""
    
    def __init__(self):
        pass
    
    def invoke(self, prompt):
        return {"content": f"This is a simulated response to: {prompt}"}


# Exercise 1: Implement a hybrid search system
def exercise1_hybrid_search(documents: List[Document], embedding_model: Any) -> BaseRetriever:
    """
    Exercise 1: Implement a hybrid search system that combines semantic and keyword search.
    
    Args:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        
    Returns:
        A hybrid retriever that combines semantic and keyword search
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")
    
    # TODO: Implement hybrid search
    # 1. Create a vector store for semantic search
    # 2. Create a keyword retriever (BM25)
    # 3. Combine the retrievers with appropriate weights
    # 4. Return the hybrid retriever
    
    # Placeholder implementation
    class SimpleHybridRetriever(BaseRetriever):
        def __init__(self, documents, embedding_model):
            self.documents = documents
            self.embedding_model = embedding_model
        
        def _get_relevant_documents(self, query):
            # Simple keyword matching
            results = []
            for doc in self.documents:
                # Count keyword matches
                keywords = query.lower().split()
                content = doc.page_content.lower()
                
                # Simple score based on keyword frequency
                score = sum(content.count(keyword) for keyword in keywords)
                
                if score > 0:
                    results.append((doc, score))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5 documents
            return [doc for doc, _ in results[:5]]
    
    return SimpleHybridRetriever(documents, embedding_model)


# Exercise 2: Create a multi-index retriever
def exercise2_multi_index_retriever(documents: List[Document], embedding_model: Any) -> BaseRetriever:
    """
    Exercise 2: Create a multi-index retriever that uses different indices for different document types.
    
    Args:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        
    Returns:
        A multi-index retriever
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")
    
    # TODO: Implement multi-index retrieval
    # 1. Group documents by type
    # 2. Create separate vector stores for each document type
    # 3. Create a retriever that selects the appropriate index based on the query
    # 4. Return the multi-index retriever
    
    # Placeholder implementation
    class SimpleMultiIndexRetriever(BaseRetriever):
        def __init__(self, documents, embedding_model):
            self.documents = documents
            self.embedding_model = embedding_model
            
            # Group documents by type
            self.document_groups = {}
            for doc in documents:
                doc_type = doc.metadata.get("type", "default")
                if doc_type not in self.document_groups:
                    self.document_groups[doc_type] = []
                self.document_groups[doc_type].append(doc)
        
        def _get_relevant_documents(self, query):
            # Extract document type from query
            query_lower = query.lower()
            selected_type = None
            
            for doc_type in self.document_groups.keys():
                if doc_type.lower() in query_lower:
                    selected_type = doc_type
                    break
            
            # If no type is specified, use all documents
            if selected_type is None or selected_type not in self.document_groups:
                docs_to_search = self.documents
            else:
                docs_to_search = self.document_groups[selected_type]
            
            # Simple keyword matching
            results = []
            for doc in docs_to_search:
                # Count keyword matches
                keywords = query.lower().split()
                content = doc.page_content.lower()
                
                # Simple score based on keyword frequency
                score = sum(content.count(keyword) for keyword in keywords)
                
                if score > 0:
                    results.append((doc, score))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5 documents
            return [doc for doc, _ in results[:5]]
    
    return SimpleMultiIndexRetriever(documents, embedding_model)


# Exercise 3: Build a parent document retrieval system
def exercise3_parent_document_retriever(documents: List[Document], embedding_model: Any) -> BaseRetriever:
    """
    Exercise 3: Build a parent document retrieval system that maintains document context.
    
    Args:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        
    Returns:
        A parent document retriever
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")
    
    # TODO: Implement parent document retrieval
    # 1. Create a text splitter for child chunks
    # 2. Create a text splitter for parent chunks
    # 3. Create a vector store for child chunks
    # 4. Create a document store for parent chunks
    # 5. Create a parent document retriever
    # 6. Return the parent document retriever
    
    # Placeholder implementation
    class SimpleParentDocumentRetriever(BaseRetriever):
        def __init__(self, documents, embedding_model):
            self.documents = documents
            self.embedding_model = embedding_model
            
            # Create parent and child documents
            self.parent_docs = []
            self.child_docs = []
            self.child_to_parent_map = {}
            
            for i, doc in enumerate(documents):
                # Create parent chunks (simplified)
                parent_id = f"parent_{i}"
                parent_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "parent_id": parent_id}
                )
                self.parent_docs.append(parent_doc)
                
                # Create child chunks (simplified)
                # In a real implementation, use a text splitter
                sentences = doc.page_content.split(". ")
                for j, sentence in enumerate(sentences):
                    if sentence:
                        child_id = f"child_{i}_{j}"
                        child_doc = Document(
                            page_content=sentence,
                            metadata={
                                **doc.metadata,
                                "child_id": child_id,
                                "parent_id": parent_id
                            }
                        )
                        self.child_docs.append(child_doc)
                        self.child_to_parent_map[child_id] = parent_id
        
        def _get_relevant_documents(self, query):
            # Simple keyword matching on child documents
            child_results = []
            for doc in self.child_docs:
                # Count keyword matches
                keywords = query.lower().split()
                content = doc.page_content.lower()
                
                # Simple score based on keyword frequency
                score = sum(content.count(keyword) for keyword in keywords)
                
                if score > 0:
                    child_results.append((doc, score))
            
            # Sort by score
            child_results.sort(key=lambda x: x[1], reverse=True)
            
            # Get parent IDs from top child documents
            parent_ids = set()
            for doc, _ in child_results[:5]:
                parent_id = doc.metadata.get("parent_id")
                if parent_id:
                    parent_ids.add(parent_id)
            
            # Return parent documents
            return [doc for doc in self.parent_docs if doc.metadata.get("parent_id") in parent_ids]
    
    return SimpleParentDocumentRetriever(documents, embedding_model)


# Exercise 4: Develop a contextual compression system
def exercise4_contextual_compression(base_retriever: BaseRetriever, llm: Optional[Any] = None) -> BaseRetriever:
    """
    Exercise 4: Develop a contextual compression system that filters irrelevant content.
    
    Args:
        base_retriever: Base retriever to use
        llm: Language model for compression (optional)
        
    Returns:
        A contextual compression retriever
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")
    
    # Initialize LLM client if not provided
    if llm is None:
        if GROQ_AVAILABLE:
            llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")
        else:
            llm = SimpleLLMClient()
    
    # TODO: Implement contextual compression
    # 1. Create a document compressor using the LLM
    # 2. Create a contextual compression retriever
    # 3. Return the compression retriever
    
    # Placeholder implementation
    class SimpleCompressionRetriever(BaseRetriever):
        def __init__(self, base_retriever, llm):
            self.base_retriever = base_retriever
            self.llm = llm
        
        def _get_relevant_documents(self, query):
            # Get documents from base retriever
            docs = self.base_retriever.get_relevant_documents(query)
            
            # Simple compression: extract sentences containing query keywords
            compressed_docs = []
            for doc in docs:
                # Split into sentences
                sentences = doc.page_content.split(". ")
                
                # Filter relevant sentences
                keywords = query.lower().split()
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        relevant_sentences.append(sentence)
                
                # Create compressed document
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
    
    return SimpleCompressionRetriever(base_retriever, llm)


# Exercise 5: Combine multiple retrieval strategies
def exercise5_combined_retrieval_system(
    documents: List[Document],
    embedding_model: Any,
    llm: Optional[Any] = None
) -> BaseRetriever:
    """
    Exercise 5: Create an advanced retrieval system that combines multiple strategies.
    
    Args:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        llm: Language model for compression (optional)
        
    Returns:
        A combined retrieval system
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")
    
    # Initialize LLM client if not provided
    if llm is None:
        if GROQ_AVAILABLE:
            llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")
        else:
            llm = SimpleLLMClient()
    
    # TODO: Implement combined retrieval system
    # 1. Create retrievers using the previous exercises
    # 2. Create a router that selects the appropriate retriever based on the query
    # 3. Return the combined retrieval system
    
    # Create retrievers
    hybrid_retriever = exercise1_hybrid_search(documents, embedding_model)
    multi_index_retriever = exercise2_multi_index_retriever(documents, embedding_model)
    parent_retriever = exercise3_parent_document_retriever(documents, embedding_model)
    compression_retriever = exercise4_contextual_compression(hybrid_retriever, llm)
    
    # Placeholder implementation
    class CombinedRetrievalSystem(BaseRetriever):
        def __init__(self, retrievers, llm):
            self.retrievers = retrievers
            self.llm = llm
        
        def _route_query(self, query):
            """Route query to appropriate retriever."""
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
        
        def _get_relevant_documents(self, query):
            # Route query to appropriate retriever
            retriever_type = self._route_query(query)
            
            if retriever_type in self.retrievers:
                return self.retrievers[retriever_type].get_relevant_documents(query)
            else:
                return self.retrievers["default"].get_relevant_documents(query)
    
    # Create combined retrieval system
    retrievers = {
        "hybrid": hybrid_retriever,
        "multi_index": multi_index_retriever,
        "parent": parent_retriever,
        "compression": compression_retriever,
        "default": hybrid_retriever
    }
    
    return CombinedRetrievalSystem(retrievers, llm)


# Exercise 6: Implement an LCEL retrieval chain
def exercise6_lcel_retrieval_chain(
    documents: List[Document],
    embedding_model: Any,
    llm: Optional[Any] = None
):
    """
    Exercise 6: Implement an LCEL retrieval chain that combines multiple strategies.
    
    Args:
        documents: List of documents to search
        embedding_model: Model to generate embeddings
        llm: Language model for compression (optional)
        
    Returns:
        An LCEL retrieval chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")
    
    # Initialize LLM client if not provided
    if llm is None:
        if GROQ_AVAILABLE:
            llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")
        else:
            llm = SimpleLLMClient()
    
    # TODO: Implement LCEL retrieval chain
    # 1. Create retrievers using the previous exercises
    # 2. Create a router function that selects the appropriate retriever
    # 3. Create an LCEL chain that routes queries to the appropriate retriever
    # 4. Return the LCEL chain
    
    # Create retrievers
    hybrid_retriever = exercise1_hybrid_search(documents, embedding_model)
    multi_index_retriever = exercise2_multi_index_retriever(documents, embedding_model)
    parent_retriever = exercise3_parent_document_retriever(documents, embedding_model)
    compression_retriever = exercise4_contextual_compression(hybrid_retriever, llm)
    
    # Create retriever dictionary
    retrievers = {
        "hybrid": hybrid_retriever,
        "multi_index": multi_index_retriever,
        "parent": parent_retriever,
        "compression": compression_retriever,
        "default": hybrid_retriever
    }
    
    # Create router function
    def route_query(query):
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
    
    # Create LCEL chain
    router_chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(lambda x: route_query(x["query"]))
    )
    
    retrieval_chain = (
        {"query": RunnablePassthrough(), "retriever_type": router_chain}
        | RunnableLambda(lambda x: retrievers[x["retriever_type"]].get_relevant_documents(x["query"]))
    )
    
    return retrieval_chain
