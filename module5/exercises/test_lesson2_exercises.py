"""
Test cases for Module 5 - Lesson 2 Exercises: Query Transformation Techniques
"""

import unittest
from typing import List, Dict, Any
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the exercises
from exercises.lesson2_exercises import (
    exercise1_query_expansion,
    exercise2_query_reformulation,
    exercise3_multi_query_retrieval,
    exercise4_hyde,
    exercise5_step_back_prompting,
    exercise6_combined_query_transformation,
    exercise7_lcel_query_transformation
)

# Check if LangChain is available
try:
    from langchain.schema.document import Document
    from langchain.schema.embeddings import Embeddings
    from langchain.schema.retriever import BaseRetriever
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
        if "expand" in prompt.lower():
            return {"content": "synonym1, synonym2, related term"}
        elif "reformulate" in prompt.lower():
            return {"content": "Reformulated query about " + prompt.split("Original query:")[-1].strip()}
        elif "variations" in prompt.lower():
            return {"content": "Variation 1\nVariation 2\nVariation 3"}
        elif "passage" in prompt.lower():
            return {"content": "This is a hypothetical document that answers the query about " + prompt.split("Question:")[-1].strip()}
        elif "general question" in prompt.lower():
            return {"content": "What is " + prompt.split("Specific question:")[-1].strip().split()[0] + "?"}
        else:
            return {"content": "This is a simulated response to: " + prompt}


# Simple retriever for testing
class SimpleRetriever(BaseRetriever):
    """Simple retriever for testing."""

    def __init__(self, documents, **kwargs):
        """Initialize with documents."""
        super().__init__(**kwargs)
        self._documents = documents

    def _get_relevant_documents(self, query):
        """Return documents containing query terms."""
        results = []
        for doc in self._documents:
            # Simple keyword matching
            if any(term.lower() in doc.page_content.lower() for term in query.lower().split()):
                results.append(doc)

        # Return up to 3 documents
        return results[:3]


# Simple vector store for testing
class SimpleVectorStore:
    """Simple vector store for testing."""

    def __init__(self, documents):
        """Initialize with documents."""
        self._documents = documents

    def similarity_search(self, query, k=3):
        """Return documents containing query terms."""
        results = []
        for doc in self._documents:
            # Simple keyword matching
            if any(term.lower() in doc.page_content.lower() for term in query.lower().split()):
                results.append(doc)

        # Return up to k documents
        return results[:k]

    def similarity_search_by_vector(self, embedding, k=3):
        """Return random documents."""
        # Just return some documents for testing
        return self._documents[:k]


@unittest.skipIf(not LANGCHAIN_AVAILABLE, "LangChain not available")
class TestLesson2Exercises(unittest.TestCase):
    """Test cases for Lesson 2 exercises."""

    def setUp(self):
        """Set up test documents, embedding model, and LLM."""
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

        # Create retriever
        self.retriever = SimpleRetriever(self.documents)

        # Create vector store
        self.vectorstore = SimpleVectorStore(self.documents)

    def test_exercise1_query_expansion(self):
        """Test query expansion implementation."""
        # Test with a simple query
        original_query = "Python programming"
        expanded_query = exercise1_query_expansion(original_query, self.llm)

        # Check that the query was expanded
        self.assertIsInstance(expanded_query, str)
        self.assertGreater(len(expanded_query), len(original_query))
        self.assertTrue(original_query in expanded_query)

    def test_exercise2_query_reformulation(self):
        """Test query reformulation implementation."""
        # Test with a simple query
        original_query = "Python programming"
        reformulated_query = exercise2_query_reformulation(original_query, self.llm)

        # Check that the query was reformulated
        self.assertIsInstance(reformulated_query, str)
        self.assertNotEqual(reformulated_query, original_query)

        # Test with domain-specific reformulation
        domain_query = exercise2_query_reformulation(original_query, self.llm, domain="technical")
        self.assertIsInstance(domain_query, str)

    def test_exercise3_multi_query_retrieval(self):
        """Test multi-query retrieval implementation."""
        # Test with a simple query
        query = "Python programming"
        results = exercise3_multi_query_retrieval(query, self.retriever, self.llm)

        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], Document)

    def test_exercise4_hyde(self):
        """Test HyDE implementation."""
        # Test with a simple query
        query = "Python programming"
        results = exercise4_hyde(query, self.vectorstore, self.llm, self.embedding_model)

        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], Document)

    def test_exercise5_step_back_prompting(self):
        """Test step-back prompting implementation."""
        # Test with a simple query
        query = "How to use Python for machine learning"
        results = exercise5_step_back_prompting(query, self.retriever, self.llm)

        # Check results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], Document)

    def test_exercise6_combined_query_transformation(self):
        """Test combined query transformation implementation."""
        # Test with different query types
        procedural_query = "How to implement machine learning in Python"
        technical_query = "Technical details of Python programming"
        comparison_query = "Compare Python and Java for machine learning"
        similarity_query = "Languages similar to Python"
        general_query = "Python programming"

        # Test each query type
        procedural_results = exercise6_combined_query_transformation(
            procedural_query, self.retriever, self.llm, self.embedding_model, self.vectorstore
        )
        technical_results = exercise6_combined_query_transformation(
            technical_query, self.retriever, self.llm, self.embedding_model, self.vectorstore
        )
        comparison_results = exercise6_combined_query_transformation(
            comparison_query, self.retriever, self.llm, self.embedding_model, self.vectorstore
        )
        similarity_results = exercise6_combined_query_transformation(
            similarity_query, self.retriever, self.llm, self.embedding_model, self.vectorstore
        )
        general_results = exercise6_combined_query_transformation(
            general_query, self.retriever, self.llm, self.embedding_model, self.vectorstore
        )

        # Check results
        for results in [procedural_results, technical_results, comparison_results, similarity_results, general_results]:
            self.assertIsInstance(results, list)
            self.assertGreaterEqual(len(results), 0)
            if len(results) > 0:
                self.assertIsInstance(results[0], Document)

    def test_exercise7_lcel_query_transformation(self):
        """Test LCEL query transformation implementation."""
        # Create LCEL chain
        transformation_chain = exercise7_lcel_query_transformation(
            self.retriever, self.llm, self.embedding_model, self.vectorstore
        )

        # Test with different query types
        procedural_query = {"query": "How to implement machine learning in Python"}
        technical_query = {"query": "Technical details of Python programming"}
        comparison_query = {"query": "Compare Python and Java for machine learning"}
        similarity_query = {"query": "Languages similar to Python"}
        general_query = {"query": "Python programming"}

        # Test each query type
        procedural_results = transformation_chain.invoke(procedural_query)
        technical_results = transformation_chain.invoke(technical_query)
        comparison_results = transformation_chain.invoke(comparison_query)
        similarity_results = transformation_chain.invoke(similarity_query)
        general_results = transformation_chain.invoke(general_query)

        # Check results
        for results in [procedural_results, technical_results, comparison_results, similarity_results, general_results]:
            self.assertIsInstance(results, list)
            self.assertGreaterEqual(len(results), 0)
            if len(results) > 0:
                self.assertIsInstance(results[0], Document)


if __name__ == "__main__":
    unittest.main()
