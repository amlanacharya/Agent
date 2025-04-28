"""
Test suite for Lesson 5 exercises on Building a Document Q&A System.
"""

import os
import sys
import json
import unittest
from typing import List, Dict, Any, Optional

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lesson5_exercises import (
    CompleteRAGSystem,
    QuestionProcessor,
    QuestionType,
    SynthesisEngine,
    SourceAttributionSystem,
    ConfidenceScorer
)

# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Check if ChromaDB is available
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Check if GroqClient is available
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


class TestCompleteRAGSystem(unittest.TestCase):
    """Test the CompleteRAGSystem implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Skip if FAISS is not available
        if not FAISS_AVAILABLE and not CHROMA_AVAILABLE:
            self.skipTest("Neither FAISS nor ChromaDB is available")
        
        # Create a RAG system
        vector_store_type = "faiss" if FAISS_AVAILABLE else "chroma"
        self.rag_system = CompleteRAGSystem(vector_store_type=vector_store_type)
        
        # Sample documents
        self.documents = [
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
    
    def test_add_documents(self):
        """Test adding documents to the RAG system."""
        # Check if add_documents is implemented
        if not hasattr(self.rag_system, 'add_documents') or not callable(getattr(self.rag_system, 'add_documents')):
            self.skipTest("add_documents method not implemented")
        
        # Add documents
        result = self.rag_system.add_documents(self.documents)
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result, 2)  # Should return number of documents added
    
    def test_search(self):
        """Test searching for documents."""
        # Check if search is implemented
        if not hasattr(self.rag_system, 'search') or not callable(getattr(self.rag_system, 'search')):
            self.skipTest("search method not implemented")
        
        # Add documents first
        if hasattr(self.rag_system, 'add_documents') and callable(getattr(self.rag_system, 'add_documents')):
            self.rag_system.add_documents(self.documents)
        else:
            self.skipTest("add_documents method not implemented")
        
        # Search for documents
        results = self.rag_system.search("What is RAG?", top_k=1)
        
        # Check results
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        if results:
            self.assertIn("content", results[0])
            self.assertIn("metadata", results[0])
            self.assertIn("score", results[0])
    
    def test_generate_answer(self):
        """Test generating an answer."""
        # Check if generate_answer is implemented
        if not hasattr(self.rag_system, 'generate_answer') or not callable(getattr(self.rag_system, 'generate_answer')):
            self.skipTest("generate_answer method not implemented")
        
        # Generate answer
        query = "What is RAG?"
        context = "RAG stands for Retrieval-Augmented Generation, a technique that enhances LLMs with external knowledge."
        answer = self.rag_system.generate_answer(query, context)
        
        # Check answer
        self.assertIsNotNone(answer)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


class TestQuestionProcessor(unittest.TestCase):
    """Test the QuestionProcessor implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = QuestionProcessor()
        
        # Sample questions
        self.factoid_question = "What is RAG?"
        self.definition_question = "Define vector database."
        self.comparison_question = "How do FAISS and ChromaDB compare?"
        self.causal_question = "Why are embeddings important for RAG?"
        self.procedural_question = "How do you implement a RAG system?"
        self.metadata_question = "Who wrote the RAG paper?"
    
    def test_analyze_question(self):
        """Test question analysis."""
        # Check if analyze_question is implemented
        if not hasattr(self.processor, 'analyze_question') or not callable(getattr(self.processor, 'analyze_question')):
            self.skipTest("analyze_question method not implemented")
        
        # Analyze question
        analysis = self.processor.analyze_question(self.factoid_question)
        
        # Check analysis
        self.assertIsNotNone(analysis)
        self.assertIsInstance(analysis, dict)
        self.assertIn("type", analysis)
        self.assertIn("entities", analysis)
    
    def test_generate_query_variations(self):
        """Test query variation generation."""
        # Check if generate_query_variations is implemented
        if not hasattr(self.processor, 'generate_query_variations') or not callable(getattr(self.processor, 'generate_query_variations')):
            self.skipTest("generate_query_variations method not implemented")
        
        # Generate variations
        variations = self.processor.generate_query_variations(self.definition_question)
        
        # Check variations
        self.assertIsNotNone(variations)
        self.assertIsInstance(variations, list)
        self.assertGreater(len(variations), 1)  # Should generate at least one variation
        self.assertIn(self.definition_question, variations)  # Original question should be included
    
    def test_reformulate_complex_question(self):
        """Test complex question reformulation."""
        # Check if reformulate_complex_question is implemented
        if not hasattr(self.processor, 'reformulate_complex_question') or not callable(getattr(self.processor, 'reformulate_complex_question')):
            self.skipTest("reformulate_complex_question method not implemented")
        
        # Complex question
        complex_question = "How do RAG systems work and what are their advantages over traditional LLMs?"
        
        # Reformulate question
        sub_questions = self.processor.reformulate_complex_question(complex_question)
        
        # Check sub-questions
        self.assertIsNotNone(sub_questions)
        self.assertIsInstance(sub_questions, list)
        self.assertGreater(len(sub_questions), 1)  # Should break into at least two questions


class TestSynthesisEngine(unittest.TestCase):
    """Test the SynthesisEngine implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = SynthesisEngine()
        
        # Sample chunks
        self.chunks = [
            {
                "content": "RAG systems enhance LLMs with external knowledge.",
                "metadata": {"source": "Source A"},
                "score": 0.9
            },
            {
                "content": "RAG stands for Retrieval-Augmented Generation.",
                "metadata": {"source": "Source B"},
                "score": 0.8
            },
            {
                "content": "Vector databases are essential for efficient RAG systems.",
                "metadata": {"source": "Source C"},
                "score": 0.7
            }
        ]
    
    def test_identify_contradictions(self):
        """Test contradiction identification."""
        # Check if identify_contradictions is implemented
        if not hasattr(self.engine, 'identify_contradictions') or not callable(getattr(self.engine, 'identify_contradictions')):
            self.skipTest("identify_contradictions method not implemented")
        
        # Create contradicting chunks
        contradicting_chunks = self.chunks + [
            {
                "content": "RAG systems do not require external knowledge.",
                "metadata": {"source": "Source D"},
                "score": 0.6
            }
        ]
        
        # Identify contradictions
        contradictions = self.engine.identify_contradictions(contradicting_chunks)
        
        # Check contradictions
        self.assertIsNotNone(contradictions)
        self.assertIsInstance(contradictions, list)
    
    def test_synthesize_information(self):
        """Test information synthesis."""
        # Check if synthesize_information is implemented
        if not hasattr(self.engine, 'synthesize_information') or not callable(getattr(self.engine, 'synthesize_information')):
            self.skipTest("synthesize_information method not implemented")
        
        # Synthesize information
        query = "What is RAG?"
        answer = self.engine.synthesize_information(query, self.chunks)
        
        # Check answer
        self.assertIsNotNone(answer)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


class TestSourceAttributionSystem(unittest.TestCase):
    """Test the SourceAttributionSystem implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = SourceAttributionSystem()
        
        # Sample chunks
        self.chunks = [
            {
                "content": "RAG systems enhance LLMs with external knowledge.",
                "metadata": {
                    "source": "RAG Paper",
                    "author": "Meta AI",
                    "date": "2023-01-01"
                },
                "score": 0.9
            },
            {
                "content": "Vector databases store embeddings for efficient similarity search.",
                "metadata": {
                    "source": "Vector DB Guide",
                    "author": "Database Experts",
                    "date": "2022-05-15"
                },
                "score": 0.8
            }
        ]
    
    def test_track_sources(self):
        """Test source tracking."""
        # Check if track_sources is implemented
        if not hasattr(self.system, 'track_sources') or not callable(getattr(self.system, 'track_sources')):
            self.skipTest("track_sources method not implemented")
        
        # Track sources
        source_map = self.system.track_sources(self.chunks)
        
        # Check source map
        self.assertIsNotNone(source_map)
        self.assertIsInstance(source_map, dict)
        self.assertEqual(len(source_map), 2)  # Should have one entry per unique content
    
    def test_format_citation(self):
        """Test citation formatting."""
        # Check if format_citation is implemented
        if not hasattr(self.system, 'format_citation') or not callable(getattr(self.system, 'format_citation')):
            self.skipTest("format_citation method not implemented")
        
        # Format citation
        source = self.chunks[0]["metadata"]
        citation = self.system.format_citation(source)
        
        # Check citation
        self.assertIsNotNone(citation)
        self.assertIsInstance(citation, str)
        self.assertGreater(len(citation), 0)
    
    def test_generate_answer_with_citations(self):
        """Test answer generation with citations."""
        # Check if generate_answer_with_citations is implemented
        if not hasattr(self.system, 'generate_answer_with_citations') or not callable(getattr(self.system, 'generate_answer_with_citations')):
            self.skipTest("generate_answer_with_citations method not implemented")
        
        # Generate answer
        query = "What is RAG?"
        answer = self.system.generate_answer_with_citations(query, self.chunks)
        
        # Check answer
        self.assertIsNotNone(answer)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        
        # Should contain citation markers
        self.assertTrue("[" in answer and "]" in answer)


class TestConfidenceScorer(unittest.TestCase):
    """Test the ConfidenceScorer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = ConfidenceScorer()
        
        # Sample query and chunks
        self.query = "What is RAG and how does it work?"
        self.high_relevance_chunks = [
            {
                "content": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLMs with external knowledge.",
                "metadata": {"source": "Source A"},
                "score": 0.9
            },
            {
                "content": "RAG works by retrieving relevant information and incorporating it into the generation process.",
                "metadata": {"source": "Source B"},
                "score": 0.85
            }
        ]
        
        self.low_relevance_chunks = [
            {
                "content": "Vector databases store embeddings for efficient similarity search.",
                "metadata": {"source": "Source C"},
                "score": 0.4
            },
            {
                "content": "Embeddings are numerical representations of text.",
                "metadata": {"source": "Source D"},
                "score": 0.3
            }
        ]
    
    def test_calculate_relevance(self):
        """Test relevance calculation."""
        # Check if calculate_relevance is implemented
        if not hasattr(self.scorer, 'calculate_relevance') or not callable(getattr(self.scorer, 'calculate_relevance')):
            self.skipTest("calculate_relevance method not implemented")
        
        # Calculate relevance
        high_relevance = self.scorer.calculate_relevance(self.query, self.high_relevance_chunks)
        low_relevance = self.scorer.calculate_relevance(self.query, self.low_relevance_chunks)
        
        # Check relevance scores
        self.assertIsNotNone(high_relevance)
        self.assertIsNotNone(low_relevance)
        self.assertGreater(high_relevance, low_relevance)  # High relevance should have higher score
    
    def test_assess_confidence(self):
        """Test confidence assessment."""
        # Check if assess_confidence is implemented
        if not hasattr(self.scorer, 'assess_confidence') or not callable(getattr(self.scorer, 'assess_confidence')):
            self.skipTest("assess_confidence method not implemented")
        
        # Assess confidence
        high_confidence = self.scorer.assess_confidence(self.query, self.high_relevance_chunks)
        low_confidence = self.scorer.assess_confidence(self.query, self.low_relevance_chunks)
        
        # Check confidence scores
        self.assertIsNotNone(high_confidence)
        self.assertIsNotNone(low_confidence)
        self.assertGreater(high_confidence, low_confidence)  # High relevance should have higher confidence
    
    def test_generate_response_with_uncertainty(self):
        """Test response generation with uncertainty."""
        # Check if generate_response_with_uncertainty is implemented
        if not hasattr(self.scorer, 'generate_response_with_uncertainty') or not callable(getattr(self.scorer, 'generate_response_with_uncertainty')):
            self.skipTest("generate_response_with_uncertainty method not implemented")
        
        # Generate responses
        high_confidence_response = self.scorer.generate_response_with_uncertainty(
            self.query, self.high_relevance_chunks, 0.9
        )
        low_confidence_response = self.scorer.generate_response_with_uncertainty(
            self.query, self.low_relevance_chunks, 0.3
        )
        
        # Check responses
        self.assertIsNotNone(high_confidence_response)
        self.assertIsNotNone(low_confidence_response)
        self.assertIsInstance(high_confidence_response, str)
        self.assertIsInstance(low_confidence_response, str)
        
        # Low confidence response should contain uncertainty language
        uncertainty_phrases = ["uncertain", "might", "possibly", "suggests", "limited information"]
        has_uncertainty = any(phrase in low_confidence_response.lower() for phrase in uncertainty_phrases)
        self.assertTrue(has_uncertainty)


if __name__ == "__main__":
    unittest.main()
