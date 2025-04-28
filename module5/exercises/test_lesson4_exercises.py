"""
Test cases for Module 5 - Lesson 4 Exercises: Self-Querying and Adaptive RAG
"""

import unittest
from typing import List, Dict, Any
import sys
import os
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the exercises
from exercises.lesson4_exercises import (
    exercise1_self_querying_retrieval,
    exercise2_query_classification,
    exercise3_query_routing,
    exercise4_multi_strategy_retrieval,
    exercise5_adaptive_rag,
    exercise6_lcel_adaptive_rag
)

# Check if LangChain is available
try:
    from langchain.schema.document import Document
    from langchain.schema.retriever import BaseRetriever
    from langchain.chains.query_constructor.base import AttributeInfo
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# Simple document class for testing when LangChain is not available
class SimpleDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Simple retriever for testing
class SimpleRetriever(BaseRetriever):
    """Simple retriever for testing."""
    
    def __init__(self, documents, name="simple"):
        """Initialize with documents."""
        self.documents = documents
        self.name = name
    
    def get_relevant_documents(self, query):
        """Return documents containing query terms."""
        results = []
        for doc in self.documents:
            # Simple keyword matching
            if any(term.lower() in doc.page_content.lower() for term in query.lower().split()):
                results.append(doc)
        
        # Return up to 3 documents
        return results[:3]


# Simple LLM for testing
class SimpleLLM:
    """Simple LLM for testing."""
    
    def __init__(self, responses=None):
        """Initialize with predefined responses."""
        self.responses = responses or {}
        self.default_response = "factual"
    
    def invoke(self, prompt):
        """Return a predefined response based on the prompt."""
        # Check if we have a response for this prompt
        for key, response in self.responses.items():
            if key in str(prompt):
                return SimpleResponse(response)
        
        # Return default response
        return SimpleResponse(self.default_response)


# Simple response class for testing
class SimpleResponse:
    """Simple response class for testing."""
    
    def __init__(self, content):
        """Initialize with content."""
        self.content = content


# Simple vector store for testing
class SimpleVectorStore:
    """Simple vector store for testing."""
    
    def __init__(self, documents):
        """Initialize with documents."""
        self.documents = documents
    
    def as_retriever(self, search_type=None, search_kwargs=None):
        """Return a simple retriever."""
        return SimpleRetriever(self.documents, name=search_type or "similarity")
    
    def similarity_search(self, query, k=3):
        """Return documents containing query terms."""
        results = []
        for doc in self.documents:
            # Simple keyword matching
            if any(term.lower() in doc.page_content.lower() for term in query.lower().split()):
                results.append(doc)
        
        # Return up to k documents
        return results[:k]


# Simple embedding model for testing
class SimpleEmbeddingModel:
    """Simple embedding model for testing."""
    
    def embed_query(self, text):
        """Return a simple embedding for testing."""
        # Just return a list of 10 zeros
        return [0] * 10
    
    def embed_documents(self, documents):
        """Return simple embeddings for testing."""
        # Just return a list of 10 zeros for each document
        return [[0] * 10 for _ in documents]


@unittest.skipIf(not LANGCHAIN_AVAILABLE, "LangChain not available")
class TestLesson4Exercises(unittest.TestCase):
    """Test cases for lesson4_exercises module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample documents
        self.documents = [
            Document(
                page_content="Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs with external knowledge.",
                metadata={"source": "RAG Paper", "author": "Meta AI", "date": "2023-01-01", "topic": "RAG"}
            ),
            Document(
                page_content="Vector databases store embeddings for efficient similarity search.",
                metadata={"source": "Vector DB Guide", "author": "Database Experts", "date": "2023-02-15", "topic": "Vector Databases"}
            ),
            Document(
                page_content="Self-querying retrieval extracts metadata filters from natural language queries.",
                metadata={"source": "Adaptive RAG Guide", "author": "RAG Experts", "date": "2023-03-10", "topic": "Self-Querying"}
            ),
            Document(
                page_content="Query classification routes queries to specialized retrievers based on query type.",
                metadata={"source": "Adaptive RAG Guide", "author": "RAG Experts", "date": "2023-03-15", "topic": "Query Classification"}
            ),
            Document(
                page_content="Multi-strategy retrieval applies different retrieval strategies based on query characteristics.",
                metadata={"source": "Adaptive RAG Guide", "author": "RAG Experts", "date": "2023-03-20", "topic": "Multi-Strategy Retrieval"}
            )
        ]
        
        # Sample queries
        self.queries = {
            "factual": "What is RAG?",
            "conceptual": "Explain how vector databases work.",
            "procedural": "How do I implement self-querying retrieval?",
            "comparative": "Compare different retrieval strategies.",
            "exploratory": "Tell me about adaptive RAG systems."
        }
        
        # Metadata field info
        self.metadata_field_info = [
            AttributeInfo(
                name="source",
                description="The source of the document",
                type="string",
            ),
            AttributeInfo(
                name="author",
                description="The author of the document",
                type="string",
            ),
            AttributeInfo(
                name="date",
                description="The publication date of the document",
                type="string",
            ),
            AttributeInfo(
                name="topic",
                description="The main topic of the document",
                type="string",
            ),
        ]
        
        # Create simple vector store
        self.vectorstore = SimpleVectorStore(self.documents)
        
        # Create simple embedding model
        self.embedding_model = SimpleEmbeddingModel()
        
        # Create simple LLM with predefined responses
        self.llm_responses = {
            "What is RAG?": "factual",
            "Explain how vector databases work": "conceptual",
            "How do I implement": "procedural",
            "Compare different": "comparative",
            "Tell me about": "exploratory",
            "Analysis": json.dumps({
                "query_type": "factual",
                "metadata_filters": {"author": "Meta AI"},
                "complexity": "simple"
            })
        }
        
        self.llm = SimpleLLM(self.llm_responses)
        
        # Create retrievers
        self.retrievers = {
            "factual": SimpleRetriever(self.documents, "factual"),
            "conceptual": SimpleRetriever(self.documents, "conceptual"),
            "procedural": SimpleRetriever(self.documents, "procedural"),
            "comparative": SimpleRetriever(self.documents, "comparative"),
            "exploratory": SimpleRetriever(self.documents, "exploratory")
        }
        
        # Create strategies
        self.strategies = {
            "semantic": SimpleRetriever(self.documents, "semantic"),
            "keyword": SimpleRetriever(self.documents, "keyword"),
            "compression": SimpleRetriever(self.documents, "compression"),
            "mmr": SimpleRetriever(self.documents, "mmr"),
            "ensemble": SimpleRetriever(self.documents, "ensemble"),
            "self_query": SimpleRetriever(self.documents, "self_query")
        }
    
    def test_exercise1_self_querying_retrieval(self):
        """Test the self-querying retrieval implementation."""
        # Mock SelfQueryRetriever
        original_self_query_retriever = None
        
        try:
            # Import the original
            from langchain.retrievers import SelfQueryRetriever as OriginalSQR
            original_self_query_retriever = OriginalSQR
            
            # Create mock
            class MockSelfQueryRetriever:
                @classmethod
                def from_llm(cls, llm, vectorstore, document_contents, metadata_field_info, verbose=False):
                    return SimpleRetriever(vectorstore.documents, "self_query")
            
            # Replace with mock
            import langchain.retrievers
            langchain.retrievers.SelfQueryRetriever = MockSelfQueryRetriever
            
            # Test the function
            result = exercise1_self_querying_retrieval(
                self.vectorstore,
                self.llm,
                self.metadata_field_info,
                "Test documents"
            )
            
            # Check result
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "self_query")
            
            # Test retrieval
            docs = result.get_relevant_documents("What is RAG?")
            self.assertGreater(len(docs), 0)
            
            # Restore original
            if original_self_query_retriever:
                langchain.retrievers.SelfQueryRetriever = original_self_query_retriever
        
        except Exception as e:
            # Restore original if exception
            if original_self_query_retriever:
                import langchain.retrievers
                langchain.retrievers.SelfQueryRetriever = original_self_query_retriever
            raise e
    
    def test_exercise2_query_classification(self):
        """Test the query classification implementation."""
        for query_type, query in self.queries.items():
            # Test classification
            result = exercise2_query_classification(query, self.llm)
            
            # For our mock LLM, the result should match the query type
            # except for exploratory which returns the default "factual"
            expected = query_type if query_type != "exploratory" else "factual"
            self.assertEqual(result, expected)
    
    def test_exercise3_query_routing(self):
        """Test the query routing implementation."""
        for query_type, query in self.queries.items():
            # Test routing
            result = exercise3_query_routing(query, self.retrievers, self.llm)
            
            # Check result
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 3)
    
    def test_exercise4_multi_strategy_retrieval(self):
        """Test the multi-strategy retrieval implementation."""
        # Test with different query types
        queries = {
            "semantic": "Find information about RAG",
            "compression": "Explain vector databases",
            "ensemble": "Compare retrieval strategies",
            "mmr": "Give me diverse information about RAG",
            "self_query": "Find documents by Meta AI"
        }
        
        for strategy, query in queries.items():
            # Test retrieval
            result = exercise4_multi_strategy_retrieval(query, self.strategies)
            
            # Check result
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 3)
    
    def test_exercise5_adaptive_rag(self):
        """Test the adaptive RAG implementation."""
        # Test with different query types
        for query_type, query in self.queries.items():
            # Test retrieval
            result = exercise5_adaptive_rag(
                query,
                self.vectorstore,
                self.llm,
                self.embedding_model,
                self.documents,
                self.metadata_field_info
            )
            
            # Check result
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 5)
    
    def test_exercise6_lcel_adaptive_rag(self):
        """Test the LCEL adaptive RAG implementation."""
        # Create LCEL chain
        chain = exercise6_lcel_adaptive_rag(
            self.vectorstore,
            self.llm,
            self.embedding_model,
            self.documents,
            self.metadata_field_info
        )
        
        # Test with different query types
        for query_type, query in self.queries.items():
            # Invoke the chain
            result = chain.invoke({"query": query})
            
            # Check result
            self.assertIsInstance(result, list)
            self.assertLessEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
