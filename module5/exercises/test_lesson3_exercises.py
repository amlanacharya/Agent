"""
Test cases for Module 5 - Lesson 3 Exercises: Reranking and Result Optimization
"""

import unittest
from typing import List, Dict, Any
import sys
import os
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the exercises
from exercises.lesson3_exercises import (
    exercise1_cross_encoder_reranker,
    exercise2_reciprocal_rank_fusion,
    exercise3_maximal_marginal_relevance,
    exercise4_source_attribution,
    exercise5_combined_reranking,
    exercise6_lcel_reranking
)

# Check if LangChain is available
try:
    from langchain.schema.document import Document
    from langchain.schema.retriever import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Check if sentence-transformers is available
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Simple document class for testing when LangChain is not available
class SimpleDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Simple embedding model for testing
class SimpleEmbeddingModel:
    """Simple embedding model for testing."""
    
    def embed_query(self, text):
        """Return a simple embedding for testing."""
        # Just return a random vector of length 10
        return np.random.rand(10)
    
    def embed_documents(self, documents):
        """Return simple embeddings for testing."""
        # Just return random vectors of length 10
        return [np.random.rand(10) for _ in documents]


# Simple retriever for testing
class SimpleRetriever(BaseRetriever):
    """Simple retriever for testing."""
    
    def __init__(self, documents):
        """Initialize with documents."""
        self.documents = documents
    
    def get_relevant_documents(self, query):
        """Return documents containing query terms."""
        results = []
        for doc in self.documents:
            # Simple keyword matching
            if any(term.lower() in doc.page_content.lower() for term in query.lower().split()):
                results.append(doc)
        
        # Return up to 3 documents
        return results[:3]


@unittest.skipIf(not LANGCHAIN_AVAILABLE, "LangChain not available")
class TestLesson3Exercises(unittest.TestCase):
    """Test cases for lesson3_exercises module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample documents
        self.documents = [
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
        
        # Sample query
        self.query = "How do reranking systems work?"
        
        # Simple embedding model
        self.embedding_model = SimpleEmbeddingModel()
        
        # Create retrievers
        self.retrievers = {
            "retriever1": SimpleRetriever(self.documents[:3]),
            "retriever2": SimpleRetriever(self.documents[2:])
        }
    
    @unittest.skipIf(not SENTENCE_TRANSFORMERS_AVAILABLE, "sentence-transformers not available")
    def test_exercise1_cross_encoder_reranker(self):
        """Test the cross-encoder reranker."""
        # Skip actual model loading in tests
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.skipTest("sentence-transformers not available")
        
        try:
            # Mock the CrossEncoder class
            original_cross_encoder = CrossEncoder
            
            class MockCrossEncoder:
                def __init__(self, model_name):
                    self.model_name = model_name
                
                def predict(self, pairs):
                    # Return mock scores
                    return [0.9, 0.8, 0.7, 0.6, 0.5]
            
            # Replace CrossEncoder with mock
            import exercises.lesson3_exercises
            exercises.lesson3_exercises.CrossEncoder = MockCrossEncoder
            
            # Test the function
            result = exercise1_cross_encoder_reranker(
                self.query, self.documents, model_name="mock-model", top_k=3
            )
            
            # Restore original CrossEncoder
            exercises.lesson3_exercises.CrossEncoder = original_cross_encoder
            
            # Check results
            self.assertEqual(len(result), 3)
            self.assertTrue(all("cross_encoder_score" in doc.metadata for doc in result))
            self.assertEqual(result[0].metadata["cross_encoder_score"], 0.9)
            self.assertEqual(result[1].metadata["cross_encoder_score"], 0.8)
            self.assertEqual(result[2].metadata["cross_encoder_score"], 0.7)
        
        except Exception as e:
            # Restore original CrossEncoder if exception occurs
            exercises.lesson3_exercises.CrossEncoder = original_cross_encoder
            raise e
    
    def test_exercise2_reciprocal_rank_fusion(self):
        """Test the reciprocal rank fusion."""
        result = exercise2_reciprocal_rank_fusion(
            self.query, self.retrievers, k=60, top_k=3
        )
        
        # Check results
        self.assertLessEqual(len(result), 3)
        self.assertTrue(all("rrf_score" in doc.metadata for doc in result))
        self.assertTrue(all("retrieval_sources" in doc.metadata for doc in result))
    
    def test_exercise3_maximal_marginal_relevance(self):
        """Test the maximal marginal relevance reranker."""
        result = exercise3_maximal_marginal_relevance(
            self.query, self.documents, self.embedding_model, lambda_param=0.7, top_k=3
        )
        
        # Check results
        self.assertEqual(len(result), 3)
        self.assertTrue(all("mmr_score" in doc.metadata for doc in result))
        self.assertTrue(all("mmr_rank" in doc.metadata for doc in result))
    
    def test_exercise4_source_attribution(self):
        """Test the source attribution system."""
        result = exercise4_source_attribution(self.documents)
        
        # Check results
        self.assertEqual(len(result), len(self.documents))
        self.assertTrue(all("attribution" in doc.metadata for doc in result))
        
        # Check attribution format
        for doc in result:
            attribution = doc.metadata["attribution"]
            self.assertIn(doc.metadata["source"], attribution)
            if doc.metadata.get("author"):
                self.assertIn(doc.metadata["author"], attribution)
    
    @unittest.skipIf(not SENTENCE_TRANSFORMERS_AVAILABLE, "sentence-transformers not available")
    def test_exercise5_combined_reranking(self):
        """Test the combined reranking system."""
        # Skip actual model loading in tests
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.skipTest("sentence-transformers not available")
        
        try:
            # Mock the CrossEncoder class
            original_cross_encoder = CrossEncoder
            
            class MockCrossEncoder:
                def __init__(self, model_name):
                    self.model_name = model_name
                
                def predict(self, pairs):
                    # Return mock scores
                    return [0.9, 0.8, 0.7, 0.6, 0.5]
            
            # Replace CrossEncoder with mock
            import exercises.lesson3_exercises
            exercises.lesson3_exercises.CrossEncoder = MockCrossEncoder
            
            # Test with different query types
            diverse_query = "Give me diverse information about reranking"
            accurate_query = "I need accurate information about reranking"
            
            diverse_result = exercise5_combined_reranking(
                diverse_query, self.documents, self.embedding_model
            )
            
            accurate_result = exercise5_combined_reranking(
                accurate_query, self.documents, self.embedding_model
            )
            
            # Restore original CrossEncoder
            exercises.lesson3_exercises.CrossEncoder = original_cross_encoder
            
            # Check results
            self.assertEqual(len(diverse_result), 5)
            self.assertEqual(len(accurate_result), 5)
            
            # Check that all documents have attribution
            self.assertTrue(all("attribution" in doc.metadata for doc in diverse_result))
            self.assertTrue(all("attribution" in doc.metadata for doc in accurate_result))
            
            # Check that diverse results have MMR scores
            self.assertTrue(any("mmr_score" in doc.metadata for doc in diverse_result))
            
            # Check that accurate results have cross-encoder scores
            self.assertTrue(any("cross_encoder_score" in doc.metadata for doc in accurate_result))
        
        except Exception as e:
            # Restore original CrossEncoder if exception occurs
            exercises.lesson3_exercises.CrossEncoder = original_cross_encoder
            raise e
    
    @unittest.skipIf(not SENTENCE_TRANSFORMERS_AVAILABLE, "sentence-transformers not available")
    def test_exercise6_lcel_reranking(self):
        """Test the LCEL reranking chain."""
        # Skip actual model loading in tests
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.skipTest("sentence-transformers not available")
        
        try:
            # Mock the CrossEncoder class
            original_cross_encoder = CrossEncoder
            
            class MockCrossEncoder:
                def __init__(self, model_name):
                    self.model_name = model_name
                
                def predict(self, pairs):
                    # Return mock scores
                    return [0.9, 0.8, 0.7, 0.6, 0.5]
            
            # Replace CrossEncoder with mock
            import exercises.lesson3_exercises
            exercises.lesson3_exercises.CrossEncoder = MockCrossEncoder
            
            # Create LCEL chain
            chain = exercise6_lcel_reranking(
                self.documents, self.embedding_model
            )
            
            # Test with different query types
            diverse_query = "Give me diverse information about reranking"
            regular_query = "Tell me about reranking"
            
            # Invoke the chain
            diverse_result = chain.invoke(diverse_query)
            regular_result = chain.invoke(regular_query)
            
            # Restore original CrossEncoder
            exercises.lesson3_exercises.CrossEncoder = original_cross_encoder
            
            # Check results
            self.assertIsInstance(diverse_result, list)
            self.assertIsInstance(regular_result, list)
            
            # Check that all documents have attribution
            self.assertTrue(all("attribution" in doc.metadata for doc in diverse_result))
            self.assertTrue(all("attribution" in doc.metadata for doc in regular_result))
        
        except Exception as e:
            # Restore original CrossEncoder if exception occurs
            exercises.lesson3_exercises.CrossEncoder = original_cross_encoder
            raise e


if __name__ == "__main__":
    unittest.main()
