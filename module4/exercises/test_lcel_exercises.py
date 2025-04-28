"""
Tests for LCEL Exercises
-----------------------
This module contains tests for the LCEL exercises.
"""

import unittest
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import langchain
    from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with 'pip install langchain' for LCEL functionality.")

# Import the exercises
if LANGCHAIN_AVAILABLE:
    from lcel_exercises import (
        exercise1_basic_lcel_chain,
        exercise2_rag_chain,
        exercise3_branching_chain,
        exercise4_memory_chain,
        exercise5_parallel_retrievers
    )


@unittest.skipIf(not LANGCHAIN_AVAILABLE, "LangChain not available")
class TestLCELExercises(unittest.TestCase):
    """Test cases for LCEL exercises."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample documents for testing
        self.documents = [
            {
                "content": "Paris is the capital of France.",
                "metadata": {"source": "Geography Book", "page": 42}
            },
            {
                "content": "The Eiffel Tower is located in Paris.",
                "metadata": {"source": "Travel Guide", "page": 15}
            }
        ]
        
        # Simple embedding model for testing
        class SimpleEmbedding:
            def embed_text(self, text):
                # Very simple embedding function for testing
                import hashlib
                return [float(b) / 255.0 for b in hashlib.md5(text.encode()).digest()[:4]]
        
        self.embedding_model = SimpleEmbedding()
    
    def test_exercise1_basic_lcel_chain(self):
        """Test the basic LCEL chain."""
        try:
            chain = exercise1_basic_lcel_chain()
            self.assertIsNotNone(chain, "Chain should not be None")
            
            # Test the chain
            result = chain.invoke("What is the capital of France?")
            self.assertIsNotNone(result, "Result should not be None")
            self.assertIsInstance(result, str, "Result should be a string")
        except NotImplementedError:
            self.skipTest("Exercise 1 not implemented yet")
    
    def test_exercise2_rag_chain(self):
        """Test the RAG chain."""
        try:
            chain = exercise2_rag_chain(self.documents, self.embedding_model)
            self.assertIsNotNone(chain, "Chain should not be None")
            
            # Test the chain
            result = chain.invoke("What is the capital of France?")
            self.assertIsNotNone(result, "Result should not be None")
            self.assertIsInstance(result, str, "Result should be a string")
        except NotImplementedError:
            self.skipTest("Exercise 2 not implemented yet")
    
    def test_exercise3_branching_chain(self):
        """Test the branching chain."""
        try:
            chain = exercise3_branching_chain()
            self.assertIsNotNone(chain, "Chain should not be None")
            
            # Test the chain with different question types
            factual_result = chain.invoke("What is the capital of France?")
            self.assertIsNotNone(factual_result, "Factual result should not be None")
            self.assertIsInstance(factual_result, str, "Factual result should be a string")
            
            procedural_result = chain.invoke("How do I make a cake?")
            self.assertIsNotNone(procedural_result, "Procedural result should not be None")
            self.assertIsInstance(procedural_result, str, "Procedural result should be a string")
            
            opinion_result = chain.invoke("Do you think AI will replace humans?")
            self.assertIsNotNone(opinion_result, "Opinion result should not be None")
            self.assertIsInstance(opinion_result, str, "Opinion result should be a string")
        except NotImplementedError:
            self.skipTest("Exercise 3 not implemented yet")
    
    def test_exercise4_memory_chain(self):
        """Test the memory chain."""
        try:
            chain = exercise4_memory_chain()
            self.assertIsNotNone(chain, "Chain should not be None")
            
            # Test the chain with conversation history
            history = [
                {"role": "human", "content": "My name is Alice."},
                {"role": "ai", "content": "Hello Alice, nice to meet you!"}
            ]
            result = chain.invoke({"question": "What is my name?", "history": history})
            self.assertIsNotNone(result, "Result should not be None")
            self.assertIsInstance(result, str, "Result should be a string")
        except NotImplementedError:
            self.skipTest("Exercise 4 not implemented yet")
    
    def test_exercise5_parallel_retrievers(self):
        """Test the parallel retrievers chain."""
        try:
            chain = exercise5_parallel_retrievers(self.documents, self.embedding_model)
            self.assertIsNotNone(chain, "Chain should not be None")
            
            # Test the chain
            result = chain.invoke("Tell me about Paris.")
            self.assertIsNotNone(result, "Result should not be None")
            self.assertIsInstance(result, str, "Result should be a string")
        except NotImplementedError:
            self.skipTest("Exercise 5 not implemented yet")


if __name__ == "__main__":
    unittest.main()
