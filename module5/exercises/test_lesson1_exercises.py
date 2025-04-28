"""
Test cases for Module 5 - Lesson 1 Exercises: Advanced Retrieval Strategies
"""

import unittest
from typing import List, Dict, Any
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the exercises
from lesson1_exercises import (
    exercise1_hybrid_search,
    exercise2_multi_index_retriever,
    exercise3_parent_document_retriever,
    exercise4_contextual_compression,
    exercise5_combined_retrieval_system,
    exercise6_lcel_retrieval_chain
)

# Check if LangChain is available
try:
    from langchain.schema.document import Document
    from langchain.schema.embeddings import Embeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain is not available. Skipping tests.")


# Simple embedding model for testing
class SimpleEmbeddings:
    """Simple embedding model for testing."""
    
    def embed_documents(self, texts):
        """Return simple embeddings for documents."""
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text):
        """Return simple embedding for query."""
        return [0.1, 0.2, 0.3]


# Simple LLM client for testing
class SimpleLLMClient:
    """Simple LLM client for testing."""
    
    def invoke(self, prompt):
        """Return simple response."""
        return {"content": f"This is a simulated response to: {prompt}"}


@unittest.skipIf(not LANGCHAIN_AVAILABLE, "LangChain not available")
class TestLesson1Exercises(unittest.TestCase):
    """Test cases for Lesson 1 exercises."""
    
    def setUp(self):
        """Set up test documents and embedding model."""
        # Create test documents
        self.documents = [
            Document(
                page_content="Python is a programming language that lets you work quickly and integrate systems more effectively.",
                metadata={"type": "technical", "source": "python.org"}
            ),
            Document(
                page_content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
                metadata={"type": "technical", "source": "wikipedia.org"}
            ),
            Document(
                page_content="Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
                metadata={"type": "technical", "source": "wikipedia.org"}
            ),
            Document(
                page_content="The Python programming language is named after the comedy group Monty Python.",
                metadata={"type": "general", "source": "python.org"}
            ),
            Document(
                page_content="Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.",
                metadata={"type": "technical", "source": "wikipedia.org"}
            )
        ]
        
        # Create embedding model
        self.embedding_model = SimpleEmbeddings()
        
        # Create LLM client
        self.llm = SimpleLLMClient()
    
    def test_exercise1_hybrid_search(self):
        """Test hybrid search implementation."""
        # Create hybrid retriever
        hybrid_retriever = exercise1_hybrid_search(self.documents, self.embedding_model)
        
        # Test retrieval
        results = hybrid_retriever.get_relevant_documents("Python programming")
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], Document)
    
    def test_exercise2_multi_index_retriever(self):
        """Test multi-index retriever implementation."""
        # Create multi-index retriever
        multi_index_retriever = exercise2_multi_index_retriever(self.documents, self.embedding_model)
        
        # Test retrieval with type specification
        results_technical = multi_index_retriever.get_relevant_documents("technical machine learning")
        results_general = multi_index_retriever.get_relevant_documents("general Python")
        
        # Check results
        self.assertIsInstance(results_technical, list)
        self.assertIsInstance(results_general, list)
        self.assertGreater(len(results_technical), 0)
        self.assertGreater(len(results_general), 0)
        self.assertIsInstance(results_technical[0], Document)
        self.assertIsInstance(results_general[0], Document)
    
    def test_exercise3_parent_document_retriever(self):
        """Test parent document retriever implementation."""
        # Create parent document retriever
        parent_retriever = exercise3_parent_document_retriever(self.documents, self.embedding_model)
        
        # Test retrieval
        results = parent_retriever.get_relevant_documents("Python programming")
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], Document)
        
        # Check that parent documents are returned
        for doc in results:
            self.assertIn("parent_id", doc.metadata)
    
    def test_exercise4_contextual_compression(self):
        """Test contextual compression implementation."""
        # Create base retriever
        base_retriever = exercise1_hybrid_search(self.documents, self.embedding_model)
        
        # Create compression retriever
        compression_retriever = exercise4_contextual_compression(base_retriever, self.llm)
        
        # Test retrieval
        results = compression_retriever.get_relevant_documents("Python programming")
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], Document)
    
    def test_exercise5_combined_retrieval_system(self):
        """Test combined retrieval system implementation."""
        # Create combined retrieval system
        combined_retriever = exercise5_combined_retrieval_system(
            self.documents,
            self.embedding_model,
            self.llm
        )
        
        # Test retrieval with different query types
        results_hybrid = combined_retriever.get_relevant_documents("exact match for Python")
        results_parent = combined_retriever.get_relevant_documents("full document about machine learning")
        results_compression = combined_retriever.get_relevant_documents("extract relevant information about AI")
        results_multi_index = combined_retriever.get_relevant_documents("technical information about programming")
        
        # Check results
        self.assertIsInstance(results_hybrid, list)
        self.assertIsInstance(results_parent, list)
        self.assertIsInstance(results_compression, list)
        self.assertIsInstance(results_multi_index, list)
        self.assertGreater(len(results_hybrid), 0)
        self.assertGreater(len(results_parent), 0)
        self.assertGreater(len(results_compression), 0)
        self.assertGreater(len(results_multi_index), 0)
    
    def test_exercise6_lcel_retrieval_chain(self):
        """Test LCEL retrieval chain implementation."""
        # Create LCEL retrieval chain
        retrieval_chain = exercise6_lcel_retrieval_chain(
            self.documents,
            self.embedding_model,
            self.llm
        )
        
        # Test retrieval with different query types
        results_hybrid = retrieval_chain.invoke({"query": "exact match for Python"})
        results_parent = retrieval_chain.invoke({"query": "full document about machine learning"})
        results_compression = retrieval_chain.invoke({"query": "extract relevant information about AI"})
        results_multi_index = retrieval_chain.invoke({"query": "technical information about programming"})
        
        # Check results
        self.assertIsInstance(results_hybrid, list)
        self.assertIsInstance(results_parent, list)
        self.assertIsInstance(results_compression, list)
        self.assertIsInstance(results_multi_index, list)
        self.assertGreater(len(results_hybrid), 0)
        self.assertGreater(len(results_parent), 0)
        self.assertGreater(len(results_compression), 0)
        self.assertGreater(len(results_multi_index), 0)


if __name__ == "__main__":
    unittest.main()
