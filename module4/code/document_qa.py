"""
Document Q&A System Implementation
--------------------------------
This module implements a complete Document Q&A system that combines
document processing, retrieval, and generation to answer questions
about document content and metadata.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Import our RAG system
from rag_system import SimpleRAGSystem, SimpleLLMClient


class DocumentQASystem:
    """Complete Document Q&A system."""

    def __init__(self, rag_system, embedding_model=None, llm_client=None):
        """
        Initialize the Document Q&A system.

        Args:
            rag_system: RAG system for retrieval
            embedding_model: Model for generating embeddings
            llm_client: Client for LLM text generation
        """
        self.rag_system = rag_system
        self.embedding_model = embedding_model

        # Initialize LLM client
        if llm_client is None:
            if GROQ_AVAILABLE:
                self.llm_client = GroqClient()
                logger.info("Using GroqClient for text generation.")
            else:
                self.llm_client = SimpleLLMClient()
                logger.info("Using SimpleLLMClient for text generation (simulated responses).")
        else:
            self.llm_client = llm_client

    def answer_question(self, question, k=5, use_hybrid=False):
        """
        Answer a user question.

        Args:
            question: User question
            k: Number of chunks to retrieve
            use_hybrid: Whether to use hybrid retrieval

        Returns:
            Answer with sources and confidence information
        """
        try:
            logger.info(f"Processing question: {question}")

            # Analyze the question
            logger.info("Analyzing question...")
            analysis = self.rag_system.analyze_question(question)
            logger.info(f"Question analysis: {analysis}")

            # Handle metadata queries differently
            if analysis["is_metadata_query"]:
                logger.info("Handling as metadata query")
                metadata_results = self.rag_system.retrieve_metadata(question)
                answer = self.rag_system.answer_metadata_query(
                    question, metadata_results, self.llm_client
                )
                return {
                    "answer": answer,
                    "sources": [r["metadata"] for r in metadata_results[:3]],
                    "is_metadata_query": True,
                    "confidence": 0.9 if metadata_results else 0.1
                }

            # Use simple retrieval instead of expanded queries
            # This is more reliable for testing
            logger.info(f"Retrieving documents with top_k={k}, use_hybrid={use_hybrid}")
            chunks = self.rag_system.retrieve(
                question, self.embedding_model, top_k=k, use_hybrid=use_hybrid
            )
            logger.info(f"Retrieved {len(chunks)} chunks")

            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "is_metadata_query": False,
                    "confidence": 0.0
                }

            # Assess confidence
            confidence = self.rag_system.assess_confidence(question, chunks)

            # Generate answer with citations
            if confidence >= 0.5:
                answer = self.rag_system.generate_answer_with_citations(
                    question, chunks, self.llm_client
                )
            else:
                # For low confidence, use uncertainty handling
                answer = self.rag_system.generate_response_with_uncertainty(
                    question, chunks, confidence, self.llm_client
                )

            return {
                "answer": answer,
                "sources": [chunk.get("metadata", {}) for chunk in chunks],
                "is_metadata_query": False,
                "confidence": confidence
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error answering question: {str(e)}\n{error_trace}")
            return {
                "answer": f"I encountered an error while trying to answer your question: {str(e)}",
                "sources": [],
                "is_metadata_query": False,
                "confidence": 0.0
            }

    def answer_with_synthesis(self, question, k=5):
        """
        Answer a question with information synthesis from multiple sources.

        Args:
            question: User question
            k: Number of chunks to retrieve

        Returns:
            Synthesized answer with sources
        """
        try:
            # Use simple retrieval instead of retrieve_from_multiple_documents
            chunks = self.rag_system.retrieve(
                question, self.embedding_model, top_k=k
            )

            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information to synthesize an answer.",
                    "sources": [],
                    "is_synthesized": False
                }

            # Synthesize information
            answer = self.rag_system.synthesize_information(
                question, chunks, self.llm_client
            )

            return {
                "answer": answer,
                "sources": [chunk.get("metadata", {}) for chunk in chunks],
                "is_synthesized": True
            }
        except Exception as e:
            logger.error(f"Error synthesizing answer: {str(e)}")
            return {
                "answer": f"I encountered an error while trying to synthesize an answer: {str(e)}",
                "sources": [],
                "is_synthesized": False
            }

    def answer_with_context(self, question, context, k=3):
        """
        Answer a question with additional context.

        Args:
            question: User question
            context: Additional context (e.g., conversation history)
            k: Number of chunks to retrieve

        Returns:
            Answer with context integration
        """
        try:
            # Augment query with context
            augmented_query = self.rag_system.augment_query(question, context)

            # Retrieve relevant chunks
            chunks = self.rag_system.retrieve(
                augmented_query, self.embedding_model, top_k=k
            )

            if not chunks:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context_used": True
                }

            # Generate answer
            answer = self.rag_system.generate_answer(
                question, chunks, self.llm_client
            )

            return {
                "answer": answer,
                "sources": [chunk.get("metadata", {}) for chunk in chunks],
                "context_used": True
            }
        except Exception as e:
            logger.error(f"Error answering with context: {str(e)}")
            return {
                "answer": f"I encountered an error while trying to answer with context: {str(e)}",
                "sources": [],
                "context_used": False
            }

    def get_document_summary(self, document_id=None, source=None):
        """
        Generate a summary of a document.

        Args:
            document_id: ID of the document to summarize
            source: Source name to summarize

        Returns:
            Document summary
        """
        try:
            # Find the document
            if document_id is not None:
                if document_id < 0 or document_id >= len(self.rag_system.documents):
                    return {"error": f"Document ID {document_id} not found"}

                document = self.rag_system.documents[document_id]
                chunks = [document]
            elif source is not None:
                # Find all chunks from the source
                chunks = []
                for doc in self.rag_system.documents:
                    if doc.get("metadata", {}).get("source") == source:
                        chunks.append(doc)

                if not chunks:
                    return {"error": f"No documents found from source '{source}'"}
            else:
                return {"error": "Either document_id or source must be provided"}

            # Create a summary prompt
            content = "\n\n".join([chunk.get("content", "") for chunk in chunks])

            # Limit content length to avoid token limits
            if len(content) > 4000:
                content = content[:4000] + "..."

            prompt = f"""
            Provide a concise summary of the following document content:

            {content}

            Summary:
            """

            # Generate summary
            response = self.llm_client.generate_text(prompt)
            summary = self.llm_client.extract_text_from_response(response)

            return {
                "summary": summary,
                "document_count": len(chunks),
                "metadata": chunks[0].get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            return {
                "error": f"Error generating summary: {str(e)}",
                "document_count": 0,
                "metadata": {}
            }


# Simple embedding model for testing
class SimpleEmbedding:
    """A simple embedding model for testing."""

    def __init__(self, dimension=4):
        """Initialize with a specific dimension."""
        self.dimension = dimension

    def embed_text(self, text):
        """Generate a simple embedding for a text with the correct dimension."""
        # Very simple embedding function for testing
        import hashlib
        import numpy as np

        # Generate a hash-based vector
        hash_value = hashlib.md5(text.encode()).digest()
        # Convert to a list of floats
        values = [float(b) / 255.0 for b in hash_value]

        # Ensure the vector has the correct dimension
        if len(values) > self.dimension:
            values = values[:self.dimension]  # Truncate if too long
        elif len(values) < self.dimension:
            # Pad with zeros if too short
            values = values + [0.0] * (self.dimension - len(values))

        # Normalize the vector
        values_array = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(values_array)
        if norm > 0:
            values_array = values_array / norm

        return values_array.tolist()

    def embed_documents(self, documents):
        """Generate embeddings for multiple documents."""
        return [self.embed_text(doc) for doc in documents]


# Example usage
if __name__ == "__main__":
    # Configure more detailed logging
    logging.basicConfig(level=logging.DEBUG,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Sample documents
    documents = [
        {
            "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs with external knowledge. RAG combines retrieval systems with generative models to improve accuracy and factuality.",
            "metadata": {
                "source": "RAG Paper",
                "author": "Meta AI",
                "date": "2023-01-01"
            }
        },
        {
            "content": "Vector databases store embeddings for efficient similarity search. They are often used in RAG systems to retrieve relevant documents.",
            "metadata": {
                "source": "Vector DB Guide",
                "author": "Database Experts",
                "date": "2022-05-15"
            }
        }
    ]

    # Create embeddings using our SimpleEmbedding class
    embedding_model = SimpleEmbedding(dimension=4)
    embeddings = embedding_model.embed_documents([doc["content"] for doc in documents])

    try:
        logger.info("Starting Document Q&A system test")

        # Create RAG system
        logger.info("Initializing RAG system")
        rag_system = SimpleRAGSystem(documents, embeddings, vector_store_type="faiss")
        logger.info("RAG system initialized successfully")

        # We already created the embedding model above
        logger.info("Using the embedding model we already created")

        # Create Document Q&A system
        logger.info("Creating Document Q&A system")
        qa_system = DocumentQASystem(
            rag_system=rag_system,
            embedding_model=embedding_model
        )
        logger.info("Document Q&A system created successfully")

        # Test question answering
        question = "What is RAG?"
        logger.info(f"Testing question: {question}")
        print(f"Question: {question}")

        # Test the retrieve method directly with hybrid search
        logger.info("Testing retrieve method directly with hybrid search")
        chunks = rag_system.retrieve(question, embedding_model, top_k=2, use_hybrid=True)
        logger.info(f"Direct retrieve result: {chunks}")

        # Test the answer_question method with hybrid search
        logger.info("Testing answer_question method with hybrid search")
        response = qa_system.answer_question(question, k=2, use_hybrid=True)
        logger.info(f"Response: {response}")

        print(f"Answer: {response['answer']}")
        if 'confidence' in response:
            print(f"Confidence: {response['confidence']}")
        print(f"Sources: {response['sources']}")

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error in main: {str(e)}\n{error_trace}")
        print(f"Error: {e}")
