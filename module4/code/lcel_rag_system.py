"""
LCEL RAG System Implementation
-----------------------------
This module implements a Retrieval-Augmented Generation (RAG) system
using LangChain Expression Language (LCEL) for more readable and maintainable chains.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import langchain
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with 'pip install langchain-core' for LCEL functionality.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with 'pip install faiss-cpu' for better vector search.")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with 'pip install chromadb' for persistent vector storage.")

# Try to import the GroqClient for LLM integration
try:
    # Try module3 path first
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    from module3.code.groq_client import GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    try:
        # Try module2-llm path
        from module2_llm.code.groq_client import GroqClient
        GROQ_AVAILABLE = True
    except ImportError:
        GROQ_AVAILABLE = False
        logger.warning("GroqClient not available. Some features will be limited.")


class LCELRagSystem:
    """A RAG system implemented using LangChain Expression Language (LCEL)."""

    def __init__(self, documents, embeddings, vector_store_type="faiss"):
        """
        Initialize the LCEL RAG system.

        Args:
            documents: List of document chunks
            embeddings: List of embedding vectors for the chunks
            vector_store_type: Type of vector database to use ("faiss" or "chroma")
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for LCEL RAG System. Install with 'pip install langchain'")

        self.documents = documents
        self.vector_store_type = vector_store_type

        # Initialize vector database
        if vector_store_type == "faiss":
            self.vector_db = self._init_faiss(documents, embeddings)
        elif vector_store_type == "chroma":
            self.vector_db = self._init_chroma(documents, embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")

        # Initialize LLM client
        if GROQ_AVAILABLE:
            self.llm_client = GroqClient()
            logger.info("Using GroqClient for text generation.")
        else:
            self.llm_client = SimpleLLMClient()
            logger.info("Using SimpleLLMClient for text generation (simulated responses).")

        # Create the basic RAG chain
        self.rag_chain = self._create_basic_rag_chain()

        # Create the citation RAG chain
        self.citation_rag_chain = self._create_citation_rag_chain()

        # Create the metadata query chain
        self.metadata_query_chain = self._create_metadata_query_chain()

    def _init_faiss(self, documents, embeddings):
        """Initialize a FAISS vector database."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with 'pip install faiss-cpu'")

        import faiss
        import numpy as np

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        return {
            "index": index,
            "documents": documents,
            "embeddings": embeddings_array
        }

    def _init_chroma(self, documents, embeddings):
        """Initialize a ChromaDB vector database."""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with 'pip install chromadb'")

        import chromadb

        # Create ChromaDB client
        client = chromadb.Client()
        collection = client.create_collection("documents")

        # Add documents to collection
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            collection.add(
                ids=[f"doc_{i}"],
                embeddings=[embedding],
                metadatas=[doc.get("metadata", {})]
            )

        return {
            "collection": collection,
            "documents": documents
        }

    def _retrieve(self, query, embedding_model, top_k=5):
        """
        Retrieve relevant documents for a query.

        Args:
            query: User question
            embedding_model: Model to generate query embedding
            top_k: Number of documents to retrieve

        Returns:
            List of relevant document chunks
        """
        # Generate query embedding
        query_embedding = embedding_model.embed_text(query)

        # Semantic search
        if self.vector_store_type == "faiss":
            results = self._faiss_search(query_embedding, top_k)
        else:
            results = self._chroma_search(query_embedding, top_k)

        return results

    def _faiss_search(self, query_embedding, top_k=5):
        """Search using FAISS."""
        import numpy as np

        # Convert query embedding to numpy array
        query_embedding = np.array([query_embedding]).astype('float32')

        # Search
        distances, indices = self.vector_db["index"].search(query_embedding, top_k)

        # Get results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.vector_db["documents"]):
                doc = self.vector_db["documents"][idx]
                results.append({
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(1.0 / (1.0 + dist))  # Convert distance to similarity score
                })

        return results

    def _chroma_search(self, query_embedding, top_k=5):
        """Search using ChromaDB."""
        # Search
        results = self.vector_db["collection"].query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Get results
        output = []
        for i, (id, distance, metadata) in enumerate(zip(
            results["ids"][0], results["distances"][0], results["metadatas"][0]
        )):
            # Get the original document
            idx = int(id.split("_")[1])
            doc = self.vector_db["documents"][idx]

            output.append({
                "content": doc.get("content", ""),
                "metadata": metadata,
                "score": float(1.0 / (1.0 + distance))  # Convert distance to similarity score
            })

        return output

    def _format_docs(self, docs):
        """Format documents for the prompt."""
        return "\n\n".join([doc["content"] for doc in docs])

    def _format_docs_with_sources(self, docs):
        """Format documents with source information."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.get("metadata", {}).get("source", f"Source {i+1}")
            formatted_docs.append(f"[{i+1}] {doc['content']}")

        return "\n\n".join(formatted_docs)

    def _add_source_details(self, answer, docs):
        """Add source details to the answer."""
        sources_text = "\n\nSources:\n"
        for i, doc in enumerate(docs):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "")
            page_info = f", page {page}" if page else ""
            sources_text += f"[{i+1}] {source}{page_info}\n"

        return answer + sources_text

    def _create_basic_rag_chain(self):
        """Create a basic RAG chain using LCEL."""
        # Define the prompt template
        prompt = PromptTemplate.from_template("""
        Answer the following question based on the provided context.
        If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {question}

        Answer:
        """)

        # Create a chain that:
        # 1. Takes a question
        # 2. Passes it to the retriever
        # 3. Formats the retrieved documents
        # 4. Passes the formatted documents and original question to the prompt
        # 5. Passes the prompt to the LLM
        # 6. Returns the LLM's response

        def retrieve_and_format(question):
            # This would normally use a LangChain retriever, but we're using our custom retriever
            docs = self._retrieve(question, self.embedding_model, top_k=5)
            return self._format_docs(docs)

        # Define the chain using LCEL
        chain = (
            {
                "context": RunnableLambda(retrieve_and_format),
                "question": RunnablePassthrough()
            }
            | prompt
            | RunnableLambda(lambda x: self.llm_client.generate_text(x))
            | RunnableLambda(lambda x: self.llm_client.extract_text_from_response(x))
        )

        return chain

    def _create_citation_rag_chain(self):
        """Create a RAG chain with citations using LCEL."""
        # Define the prompt template
        prompt = PromptTemplate.from_template("""
        Answer the following question based on the provided sources.
        Use citation numbers [1], [2], etc. to indicate which source supports each part of your answer.

        Question: {question}

        Sources:
        {context}

        Answer with citations:
        """)

        def retrieve_and_format_with_sources(question):
            # Retrieve documents
            docs = self._retrieve(question, self.embedding_model, top_k=5)
            # Format with source information
            return {"docs": docs, "formatted_context": self._format_docs_with_sources(docs)}

        def add_sources_to_answer(inputs):
            answer = self.llm_client.generate_text(inputs["prompt"])
            answer = self.llm_client.extract_text_from_response(answer)
            return self._add_source_details(answer, inputs["docs"])

        # Define the chain using LCEL
        chain = (
            RunnableLambda(retrieve_and_format_with_sources)
            | {
                "prompt": lambda x: prompt.format(question=x["question"], context=x["formatted_context"]),
                "docs": lambda x: x["docs"],
                "question": lambda x: x["question"]
            }
            | RunnableLambda(add_sources_to_answer)
        )

        return chain

    def _create_metadata_query_chain(self):
        """Create a chain for metadata queries using LCEL."""
        # Define the prompt template
        prompt = PromptTemplate.from_template("""
        Answer the following question about document metadata.

        Question: {question}

        Document Metadata:
        {metadata_text}

        Answer:
        """)

        def retrieve_metadata(question):
            # Extract key terms from query
            query_terms = set(question.lower().split())
            stop_words = {"what", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an"}
            query_terms = query_terms - stop_words

            # Search for documents with matching metadata
            results = []
            metadata_fields = ["author", "title", "date", "source", "type"]

            for i, doc in enumerate(self.documents):
                metadata = doc.get("metadata", {})

                # Calculate a simple relevance score
                score = 0
                matched_fields = []

                for field in metadata_fields:
                    if field in metadata:
                        field_value = str(metadata[field]).lower()

                        # Check if any query term is in the metadata value
                        for term in query_terms:
                            if term in field_value:
                                score += 1
                                matched_fields.append(field)

                if score > 0:
                    results.append({
                        "document_id": i,
                        "metadata": metadata,
                        "score": score,
                        "matched_fields": matched_fields
                    })

            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:5]  # Limit to top 5

        def format_metadata(results):
            if not results:
                return "No matching documents found."

            metadata_context = []
            for i, result in enumerate(results):
                metadata = result["metadata"]
                metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                metadata_context.append(f"Document {i+1}: {metadata_str}")

            return "\n".join(metadata_context)

        # Define the chain using LCEL
        chain = (
            RunnableLambda(lambda x: {"question": x, "metadata_results": retrieve_metadata(x)})
            | {
                "question": lambda x: x["question"],
                "metadata_text": lambda x: format_metadata(x["metadata_results"]),
                "metadata_results": lambda x: x["metadata_results"]
            }
            | prompt
            | RunnableLambda(lambda x: self.llm_client.generate_text(x))
            | RunnableLambda(lambda x: self.llm_client.extract_text_from_response(x))
        )

        return chain

    def answer_question(self, question, embedding_model, use_citations=True):
        """
        Answer a question using the LCEL RAG chain.

        Args:
            question: User question
            embedding_model: Model to generate embeddings
            use_citations: Whether to include citations in the answer

        Returns:
            Generated answer
        """
        self.embedding_model = embedding_model

        # Check if this is a metadata query
        metadata_keywords = {
            "author", "wrote", "written", "published", "publication",
            "date", "year", "when was", "how old", "recent",
            "title", "called", "named", "file", "document",
            "type", "format", "source", "where from", "origin"
        }

        is_metadata_query = any(keyword in question.lower() for keyword in metadata_keywords)

        if is_metadata_query:
            # Use the metadata query chain
            return self.metadata_query_chain.invoke(question)
        elif use_citations:
            # Use the citation RAG chain
            return self.citation_rag_chain.invoke({"question": question})
        else:
            # Use the basic RAG chain
            return self.rag_chain.invoke(question)


# Simple LLM client for testing when GroqClient is not available
class SimpleLLMClient:
    """A simple LLM client for testing."""

    def generate_text(self, prompt, **kwargs):
        """Generate text from a prompt."""
        # This is a very simple simulation
        return {
            "choices": [
                {
                    "message": {
                        "content": f"This is a simulated response to: {prompt[:50]}..."
                    }
                }
            ]
        }

    def extract_text_from_response(self, response):
        """Extract text from a response."""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return response


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        {
            "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs with external knowledge.",
            "metadata": {
                "source": "RAG Paper",
                "author": "Meta AI",
                "date": "2023-01-01"
            }
        },
        {
            "content": "Vector databases store embeddings for efficient similarity search.",
            "metadata": {
                "source": "Vector DB Guide",
                "author": "Database Experts",
                "date": "2022-05-15"
            }
        }
    ]

    # Sample embeddings (simplified for demonstration)
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ]

    # Simple embedding model for testing
    class SimpleEmbedding:
        def embed_text(self, text):
            # Very simple embedding function for testing
            return [0.1, 0.2, 0.3, 0.4]

    embedding_model = SimpleEmbedding()

    try:
        # Check if LangChain is available
        if not LANGCHAIN_AVAILABLE:
            print("LangChain is not available. Install with 'pip install langchain' to run this example.")
            sys.exit(1)

        # Create LCEL RAG system
        rag_system = LCELRagSystem(documents, embeddings, vector_store_type="faiss")
        print("LCEL RAG system initialized successfully.")

        # Test question answering
        query = "What is RAG?"
        print(f"\nQuery: {query}")

        # Answer with citations
        answer = rag_system.answer_question(query, embedding_model, use_citations=True)
        print(f"Answer with citations: {answer}")

        # Answer without citations
        answer = rag_system.answer_question(query, embedding_model, use_citations=False)
        print(f"Answer without citations: {answer}")

        # Test metadata query
        metadata_query = "Who wrote the RAG paper?"
        print(f"\nMetadata Query: {metadata_query}")
        answer = rag_system.answer_question(metadata_query, embedding_model)
        print(f"Answer: {answer}")

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error: {str(e)}\n{error_trace}")
