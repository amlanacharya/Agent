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
        # Analyze the question
        analysis = self.rag_system.analyze_question(question)
        
        # Handle metadata queries differently
        if analysis["is_metadata_query"]:
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
        
        # Expand the query
        expanded_queries = self.rag_system.expand_query(question, analysis)
        
        # Retrieve from multiple documents
        chunks = self.rag_system.retrieve_from_multiple_documents(
            expanded_queries[0], self.embedding_model, top_k=k
        )
        
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
    
    def answer_with_synthesis(self, question, k=5):
        """
        Answer a question with information synthesis from multiple sources.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            
        Returns:
            Synthesized answer with sources
        """
        # Retrieve from multiple documents
        chunks = self.rag_system.retrieve_from_multiple_documents(
            question, self.embedding_model, top_k=k, docs_per_source=2
        )
        
        # Synthesize information
        answer = self.rag_system.synthesize_information(
            question, chunks, self.llm_client
        )
        
        return {
            "answer": answer,
            "sources": [chunk.get("metadata", {}) for chunk in chunks],
            "is_synthesized": True
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
        # Augment query with context
        augmented_query = self.rag_system.augment_query(question, context)
        
        # Retrieve relevant chunks
        chunks = self.rag_system.retrieve(
            augmented_query, self.embedding_model, top_k=k
        )
        
        # Generate answer
        answer = self.rag_system.generate_answer(
            question, chunks, self.llm_client
        )
        
        return {
            "answer": answer,
            "sources": [chunk.get("metadata", {}) for chunk in chunks],
            "context_used": True
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


# Simple embedding model for testing
class SimpleEmbedding:
    """A simple embedding model for testing."""
    
    def embed_text(self, text):
        """Generate a simple embedding for a text."""
        # Very simple embedding function for testing
        import hashlib
        hash_value = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_value]
    
    def embed_documents(self, documents):
        """Generate embeddings for multiple documents."""
        return [self.embed_text(doc) for doc in documents]


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
    
    try:
        # Create RAG system
        rag_system = SimpleRAGSystem(documents, embeddings, vector_store_type="faiss")
        
        # Create embedding model
        embedding_model = SimpleEmbedding()
        
        # Create Document Q&A system
        qa_system = DocumentQASystem(
            rag_system=rag_system,
            embedding_model=embedding_model
        )
        
        # Test question answering
        question = "What is RAG?"
        print(f"Question: {question}")
        
        response = qa_system.answer_question(question, k=1)
        print(f"Answer: {response['answer']}")
        print(f"Confidence: {response['confidence']}")
        print(f"Sources: {response['sources']}")
        
    except Exception as e:
        print(f"Error: {e}")
