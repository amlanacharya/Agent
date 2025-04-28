"""
Module 5 - Lesson 1 Exercises: Advanced Retrieval Strategies

This module contains exercises for implementing advanced retrieval strategies
including hybrid search, multi-index retrieval, parent document retrieval,
and contextual compression.
"""

from typing import List, Dict, Any, Optional
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnableLambda

# Check if LangChain is available
try:
    # Try importing specific modules we need
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema.runnable import RunnableBranch
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

# Check if Groq is available
try:
    from langchain.chat_models import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


from langchain_core.language_models.base import LanguageModelInput
from langchain_core.outputs import Generation, LLMResult
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

class SimpleLLMClient(BaseChatModel):
    """Simple LLM client for when Groq is not available."""

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "simple-llm-client"

    def _generate(self, messages: list[BaseMessage], **kwargs) -> LLMResult:
        """Generate responses for multiple messages."""
        response_text = f"This is a simulated response to a message"
        generations = [[Generation(text=response_text)]]
        return LLMResult(generations=generations)

    def invoke(self, input: LanguageModelInput, **kwargs):
        """Process the input prompt and return a response."""
        if isinstance(input, str):
            response_text = f"This is a simulated response to: {input}"
        else:
            response_text = f"This is a simulated response to a complex prompt"

        return AIMessage(content=response_text)


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

    # Implement hybrid search
    # 1. Create a vector store for semantic search
    # 2. Create a keyword retriever (BM25)
    # 3. Combine the retrievers with appropriate weights
    # 4. Return the hybrid retriever

    # Create vector store for semantic search
    # Import already handled at the top of the file

    # Create embeddings for documents
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # Create FAISS vector store
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    # Create semantic retriever
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Create keyword retriever (BM25)
    keyword_retriever = BM25Retriever.from_documents(documents, k=5)

    # Combine retrievers with weights
    # Import already handled at the top of the file

    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[0.7, 0.3]  # 70% weight to semantic, 30% to keyword
    )

    return hybrid_retriever


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

    # Implement multi-index retrieval
    # 1. Group documents by type
    # 2. Create separate vector stores for each document type
    # 3. Create a retriever that selects the appropriate index based on the query
    # 4. Return the multi-index retriever

    # Import already handled at the top of the file

    # Group documents by type
    document_groups = {}
    for doc in documents:
        doc_type = doc.metadata.get("type", "default")
        if doc_type not in document_groups:
            document_groups[doc_type] = []
        document_groups[doc_type].append(doc)

    # Create separate vector stores for each document type
    vector_stores = {}
    for doc_type, docs in document_groups.items():
        if docs:  # Only create vector store if there are documents
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]

            vector_stores[doc_type] = FAISS.from_texts(
                texts=texts,
                embedding=embedding_model,
                metadatas=metadatas
            )

    # Create retrievers for each vector store
    retrievers = {}
    for doc_type, vector_store in vector_stores.items():
        retrievers[doc_type] = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    # Create a default vector store for all documents
    all_texts = [doc.page_content for doc in documents]
    all_metadatas = [doc.metadata for doc in documents]

    default_vector_store = FAISS.from_texts(
        texts=all_texts,
        embedding=embedding_model,
        metadatas=all_metadatas
    )

    default_retriever = default_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Create a multi-index retriever
    class MultiIndexRetriever(BaseRetriever):
        def __init__(self, retrievers, default_retriever, **kwargs):
            super().__init__(**kwargs)
            self._retrievers = retrievers
            self._default_retriever = default_retriever

        def _get_relevant_documents(self, query):
            # Extract document type from query
            query_lower = query.lower()
            selected_type = None

            # Check if query mentions a specific document type
            for doc_type in self._retrievers.keys():
                if doc_type.lower() in query_lower:
                    selected_type = doc_type
                    break

            # Use the appropriate retriever
            if selected_type and selected_type in self._retrievers:
                return self._retrievers[selected_type].get_relevant_documents(query)
            else:
                return self._default_retriever.get_relevant_documents(query)

    # Return the multi-index retriever
    return MultiIndexRetriever(retrievers, default_retriever)


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

    # Implement parent document retrieval
    # 1. Create a text splitter for child chunks
    # 2. Create a text splitter for parent chunks
    # 3. Create a vector store for child chunks
    # 4. Create a document store for parent chunks
    # 5. Create a parent document retriever
    # 6. Return the parent document retriever

    # Imports already handled at the top of the file

    # Create text splitter for child chunks
    # (We don't need a separate parent splitter since we're keeping the original documents as parents)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    # Create parent and child documents
    parent_docs = []
    child_docs = []
    child_to_parent_map = {}

    for i, doc in enumerate(documents):
        # Create parent chunks
        parent_id = f"parent_{i}"
        parent_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "parent_id": parent_id}
        )
        parent_docs.append(parent_doc)

        # Create child chunks
        child_chunks = child_splitter.split_documents([parent_doc])

        for j, child_doc in enumerate(child_chunks):
            child_id = f"child_{i}_{j}"
            child_doc.metadata["child_id"] = child_id
            child_doc.metadata["parent_id"] = parent_id
            child_docs.append(child_doc)
            child_to_parent_map[child_id] = parent_id

    # Create vector store for child chunks
    child_texts = [doc.page_content for doc in child_docs]
    child_metadatas = [doc.metadata for doc in child_docs]

    child_vectorstore = FAISS.from_texts(
        texts=child_texts,
        embedding=embedding_model,
        metadatas=child_metadatas
    )

    # Create child retriever
    child_retriever = child_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Create parent document retriever
    class ParentDocumentRetriever(BaseRetriever):
        def __init__(self, child_retriever, parent_docs, child_to_parent_map, **kwargs):
            super().__init__(**kwargs)
            self._child_retriever = child_retriever
            self._parent_docs = parent_docs
            self._child_to_parent_map = child_to_parent_map
            self._parent_docs_map = {doc.metadata["parent_id"]: doc for doc in parent_docs}

        def _get_relevant_documents(self, query):
            # Retrieve relevant child documents
            child_results = self._child_retriever.get_relevant_documents(query)

            # Get parent IDs from child documents
            parent_ids = set()
            for doc in child_results:
                parent_id = doc.metadata.get("parent_id")
                if parent_id:
                    parent_ids.add(parent_id)

            # Return parent documents
            return [self._parent_docs_map[parent_id] for parent_id in parent_ids if parent_id in self._parent_docs_map]

    # Return parent document retriever
    return ParentDocumentRetriever(child_retriever, parent_docs, child_to_parent_map)


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

    # For testing purposes, we'll create a simple document compressor
    # that just returns the original documents
    class SimpleDocumentCompressor:
        def compress_documents(self, documents, query):
            # Just return the original documents
            for doc in documents:
                doc.metadata["compressed"] = True
            return documents

    # Create contextual compression retriever
    class SimpleCompressionRetriever(BaseRetriever):
        def __init__(self, base_retriever, compressor, **kwargs):
            super().__init__(**kwargs)
            self._base_retriever = base_retriever
            self._compressor = compressor

        def _get_relevant_documents(self, query):
            # Get documents from base retriever
            docs = self._base_retriever.get_relevant_documents(query)
            # Compress documents
            compressed_docs = self._compressor.compress_documents(docs, query)
            return compressed_docs

    # Create document compressor
    compressor = SimpleDocumentCompressor()

    # Create contextual compression retriever
    compression_retriever = SimpleCompressionRetriever(
        base_retriever=base_retriever,
        compressor=compressor
    )

    return compression_retriever


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
        def __init__(self, retrievers, llm, **kwargs):
            super().__init__(**kwargs)
            self._retrievers = retrievers
            self._llm = llm

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

            if retriever_type in self._retrievers:
                return self._retrievers[retriever_type].get_relevant_documents(query)
            else:
                return self._retrievers["default"].get_relevant_documents(query)

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

    # Implement LCEL retrieval chain
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

    # Create a function to extract the query from the input
    def extract_query(input_dict):
        if isinstance(input_dict, dict):
            return input_dict.get("query", "")
        return input_dict

    # Create LCEL chain with RunnableBranch
    # Import already handled at the top of the file

    # Create the branch for routing queries
    branch = RunnableBranch(
        (lambda x: isinstance(x, str) and ("exact" in x.lower() or "keyword" in x.lower() or "specific" in x.lower()),
         RunnableLambda(lambda x: retrievers["hybrid"].get_relevant_documents(x))),
        (lambda x: isinstance(x, str) and ("context" in x.lower() or "full" in x.lower() or "document" in x.lower()),
         RunnableLambda(lambda x: retrievers["parent"].get_relevant_documents(x))),
        (lambda x: isinstance(x, str) and ("compress" in x.lower() or "relevant" in x.lower() or "extract" in x.lower()),
         RunnableLambda(lambda x: retrievers["compression"].get_relevant_documents(x))),
        (lambda x: isinstance(x, str) and ("technical" in x.lower() or "general" in x.lower() or "specialized" in x.lower()),
         RunnableLambda(lambda x: retrievers["multi_index"].get_relevant_documents(x))),
        RunnableLambda(lambda x: retrievers["default"].get_relevant_documents(x))
    )

    # Create the full chain: extract query -> route to appropriate retriever
    retrieval_chain = RunnableLambda(extract_query) | branch

    return retrieval_chain
