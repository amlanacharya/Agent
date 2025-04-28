"""
Exercises for Lesson 5: Building a Document Q&A System.

This module contains exercises to practice building a complete
Document Q&A system with RAG capabilities.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import optional dependencies
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

# Try to import from code directory
try:
    from code.embedding_pipelines import BaseEmbeddings, SentenceTransformerEmbeddings, HashEmbeddings
except ImportError:
    # Define minimal versions for exercises
    class BaseEmbeddings:
        def embed_text(self, text: str) -> List[float]:
            raise NotImplementedError("Subclasses must implement embed_text")
        
        def embed_documents(self, documents: List[str]) -> List[List[float]]:
            return [self.embed_text(doc) for doc in documents]
    
    class HashEmbeddings(BaseEmbeddings):
        def embed_text(self, text: str) -> List[float]:
            import hashlib
            hash_value = hashlib.md5(text.encode()).digest()
            return [float(b) / 255.0 for b in hash_value]
    
    class SentenceTransformerEmbeddings(BaseEmbeddings):
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self.model_name = model_name
            self.model = None
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            except ImportError:
                logger.warning("SentenceTransformers not available. Using fallback.")
                self.fallback = HashEmbeddings()
        
        def embed_text(self, text: str) -> List[float]:
            if self.model is not None:
                try:
                    embedding = self.model.encode(text)
                    return embedding.tolist()
                except Exception as e:
                    logger.warning(f"Error generating embedding: {e}")
                    return self.fallback.embed_text(text)
            else:
                return self.fallback.embed_text(text)


# Exercise 1: Implement a Complete RAG System
class CompleteRAGSystem:
    """
    Exercise 1: Implement a complete RAG system.
    
    Create a RAG system that integrates with a vector database (FAISS or ChromaDB)
    and includes retrieval, context augmentation, and generation components.
    
    For this exercise, implement a RAG system that can:
    - Initialize and manage a vector database
    - Perform semantic search with customizable parameters
    - Augment queries with additional context
    - Generate answers using an LLM
    - Support hybrid retrieval (combining semantic and keyword search)
    """
    
    def __init__(self, vector_store_type="faiss", embedding_model=None):
        """
        Initialize the RAG system.
        
        Args:
            vector_store_type: Type of vector database to use ("faiss" or "chroma")
            embedding_model: Model to use for generating embeddings
        """
        # TODO: Initialize the RAG system
        self.vector_store_type = vector_store_type
        
        # Initialize embedding model
        if embedding_model is None:
            self.embedding_model = SentenceTransformerEmbeddings()
        else:
            self.embedding_model = embedding_model
        
        # Initialize vector database
        self.documents = []
        self.vector_db = None
    
    def add_documents(self, documents):
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            Number of documents added
        """
        # TODO: Implement document addition
        # 1. Generate embeddings for documents
        # 2. Add to vector database
        # 3. Store documents for retrieval
        pass
    
    def search(self, query, top_k=5, use_hybrid=False):
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search
            
        Returns:
            List of relevant documents
        """
        # TODO: Implement search functionality
        # 1. Generate query embedding
        # 2. Perform vector search
        # 3. If hybrid, combine with keyword search
        # 4. Return results
        pass
    
    def generate_answer(self, query, context, llm_client=None):
        """
        Generate an answer based on the query and context.
        
        Args:
            query: User question
            context: Retrieved context
            llm_client: LLM client for generation
            
        Returns:
            Generated answer
        """
        # TODO: Implement answer generation
        # 1. Format prompt with query and context
        # 2. Call LLM to generate answer
        # 3. Return formatted response
        pass


# Exercise 2: Build a Question Processing System
class QuestionType(Enum):
    """Enum for question types."""
    FACTOID = "factoid"
    DEFINITION = "definition"
    COMPARISON = "comparison"
    CAUSAL = "causal"
    PROCEDURAL = "procedural"
    OPINION = "opinion"
    UNKNOWN = "unknown"


class QuestionProcessor:
    """
    Exercise 2: Build a question processing system.
    
    Develop a system that analyzes questions, identifies their type,
    extracts key entities, and generates query variations for improved retrieval.
    
    For this exercise, implement a question processor that can:
    - Classify questions by type
    - Extract key entities and terms
    - Generate query variations
    - Identify metadata-specific queries
    - Reformulate complex questions
    """
    
    def __init__(self, use_llm=False):
        """
        Initialize the question processor.
        
        Args:
            use_llm: Whether to use an LLM for advanced processing
        """
        # TODO: Initialize the question processor
        self.use_llm = use_llm
        
        # Initialize LLM client if needed
        if use_llm and GROQ_AVAILABLE:
            self.llm_client = GroqClient()
        else:
            self.llm_client = None
    
    def analyze_question(self, question):
        """
        Analyze a question to identify its type and key entities.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with question analysis
        """
        # TODO: Implement question analysis
        # 1. Identify question type
        # 2. Extract key entities
        # 3. Determine if it's a metadata query
        # 4. Return analysis dictionary
        pass
    
    def generate_query_variations(self, question, analysis=None):
        """
        Generate variations of the query for improved retrieval.
        
        Args:
            question: Original question
            analysis: Question analysis (optional)
            
        Returns:
            List of query variations
        """
        # TODO: Implement query variation generation
        # 1. If analysis not provided, analyze the question
        # 2. Generate variations based on question type
        # 3. Add entity-focused queries
        # 4. Return list of variations
        pass
    
    def reformulate_complex_question(self, question):
        """
        Reformulate a complex question into simpler sub-questions.
        
        Args:
            question: Complex question
            
        Returns:
            List of simpler questions
        """
        # TODO: Implement complex question reformulation
        # 1. Analyze question complexity
        # 2. Break down into sub-questions
        # 3. Return list of simpler questions
        pass


# Exercise 3: Create a Multi-Document Synthesis Engine
class SynthesisEngine:
    """
    Exercise 3: Create a multi-document synthesis engine.
    
    Implement a system that can retrieve information from multiple documents
    and synthesize it into a coherent answer.
    
    For this exercise, implement a synthesis engine that can:
    - Retrieve information from multiple sources
    - Identify and resolve contradictions
    - Combine information coherently
    - Maintain source attribution
    - Handle information gaps
    """
    
    def __init__(self, rag_system=None):
        """
        Initialize the synthesis engine.
        
        Args:
            rag_system: RAG system for retrieval
        """
        # TODO: Initialize the synthesis engine
        self.rag_system = rag_system
        
        # Initialize LLM client
        if GROQ_AVAILABLE:
            self.llm_client = GroqClient()
        else:
            self.llm_client = None
    
    def retrieve_from_multiple_sources(self, query, top_k=3, sources_per_query=2):
        """
        Retrieve information from multiple sources.
        
        Args:
            query: User question
            top_k: Total number of chunks to retrieve
            sources_per_query: Maximum chunks per source
            
        Returns:
            List of relevant chunks from different sources
        """
        # TODO: Implement multi-source retrieval
        # 1. Retrieve initial results
        # 2. Group by source
        # 3. Select top chunks from each source
        # 4. Return balanced results
        pass
    
    def identify_contradictions(self, chunks):
        """
        Identify contradictions between chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            List of contradictions
        """
        # TODO: Implement contradiction detection
        # 1. Compare information across chunks
        # 2. Identify potential contradictions
        # 3. Return list of contradictions
        pass
    
    def synthesize_information(self, query, chunks):
        """
        Synthesize information from multiple chunks.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            
        Returns:
            Synthesized answer
        """
        # TODO: Implement information synthesis
        # 1. Format chunks with source information
        # 2. Create synthesis prompt
        # 3. Generate synthesized answer
        # 4. Return formatted response
        pass


# Exercise 4: Design a Source Attribution System
class SourceAttributionSystem:
    """
    Exercise 4: Design a source attribution system.
    
    Build a system that tracks the sources of information and
    generates answers with proper citations.
    
    For this exercise, implement a source attribution system that can:
    - Track sources for retrieved information
    - Generate in-line citations
    - Create a bibliography
    - Assess source reliability
    - Handle multiple citation formats
    """
    
    def __init__(self, citation_format="numeric"):
        """
        Initialize the source attribution system.
        
        Args:
            citation_format: Format for citations ("numeric", "author-date", "footnote")
        """
        # TODO: Initialize the source attribution system
        self.citation_format = citation_format
    
    def track_sources(self, chunks):
        """
        Track sources for retrieved chunks.
        
        Args:
            chunks: Retrieved document chunks
            
        Returns:
            Dictionary mapping content to sources
        """
        # TODO: Implement source tracking
        # 1. Create mapping from content to sources
        # 2. Extract metadata from chunks
        # 3. Return source map
        pass
    
    def format_citation(self, source, format_type=None):
        """
        Format a citation according to the specified format.
        
        Args:
            source: Source metadata
            format_type: Citation format (defaults to self.citation_format)
            
        Returns:
            Formatted citation
        """
        # TODO: Implement citation formatting
        # 1. Extract source metadata
        # 2. Format according to specified style
        # 3. Return formatted citation
        pass
    
    def generate_answer_with_citations(self, query, chunks, llm_client=None):
        """
        Generate an answer with citations.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            llm_client: LLM client for generation
            
        Returns:
            Answer with citations
        """
        # TODO: Implement answer generation with citations
        # 1. Track sources
        # 2. Format prompt with citation instructions
        # 3. Generate answer
        # 4. Add bibliography
        # 5. Return formatted response
        pass
    
    def assess_source_reliability(self, source):
        """
        Assess the reliability of a source.
        
        Args:
            source: Source metadata
            
        Returns:
            Reliability score (0-1)
        """
        # TODO: Implement source reliability assessment
        # 1. Check for author, date, publisher
        # 2. Look for academic or official sources
        # 3. Calculate reliability score
        # 4. Return score
        pass


# Exercise 5: Implement Confidence Scoring
class ConfidenceScorer:
    """
    Exercise 5: Implement confidence scoring.
    
    Develop a confidence scoring system that assesses the reliability
    of answers and communicates uncertainty appropriately.
    
    For this exercise, implement a confidence scorer that can:
    - Assess relevance of retrieved chunks
    - Evaluate consistency across sources
    - Calculate information coverage
    - Assess source quality
    - Generate responses with appropriate uncertainty language
    """
    
    def __init__(self):
        """Initialize the confidence scorer."""
        # TODO: Initialize the confidence scorer
        pass
    
    def calculate_relevance(self, query, chunks):
        """
        Calculate relevance score based on semantic similarity.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            
        Returns:
            Relevance score (0-1)
        """
        # TODO: Implement relevance calculation
        # 1. Calculate similarity between query and chunks
        # 2. Normalize scores
        # 3. Return relevance score
        pass
    
    def calculate_consistency(self, chunks):
        """
        Calculate consistency score based on agreement between chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Consistency score (0-1)
        """
        # TODO: Implement consistency calculation
        # 1. Compare information across chunks
        # 2. Identify agreements and disagreements
        # 3. Calculate consistency score
        # 4. Return score
        pass
    
    def calculate_coverage(self, query, chunks):
        """
        Calculate coverage score based on how well chunks cover the query.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            
        Returns:
            Coverage score (0-1)
        """
        # TODO: Implement coverage calculation
        # 1. Extract key terms from query
        # 2. Check presence in chunks
        # 3. Calculate coverage ratio
        # 4. Return score
        pass
    
    def assess_confidence(self, query, chunks):
        """
        Assess overall confidence in the answer.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            
        Returns:
            Confidence score (0-1)
        """
        # TODO: Implement confidence assessment
        # 1. Calculate individual factors
        # 2. Apply weights
        # 3. Return weighted average
        pass
    
    def generate_response_with_uncertainty(self, query, chunks, confidence, llm_client=None):
        """
        Generate a response with appropriate uncertainty language.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            confidence: Confidence score (0-1)
            llm_client: LLM client for generation
            
        Returns:
            Response with uncertainty language
        """
        # TODO: Implement uncertainty communication
        # 1. Determine confidence level
        # 2. Select appropriate language
        # 3. Generate base answer
        # 4. Add uncertainty prefix
        # 5. Return formatted response
        pass


# Example usage
if __name__ == "__main__":
    print("Lesson 5 Exercises: Building a Document Q&A System")
    
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
    
    # Exercise 1: Complete RAG System
    print("\nExercise 1: Complete RAG System")
    rag_system = CompleteRAGSystem()
    
    # Exercise 2: Question Processing System
    print("\nExercise 2: Question Processing System")
    question_processor = QuestionProcessor()
    
    # Exercise 3: Multi-Document Synthesis Engine
    print("\nExercise 3: Multi-Document Synthesis Engine")
    synthesis_engine = SynthesisEngine()
    
    # Exercise 4: Source Attribution System
    print("\nExercise 4: Source Attribution System")
    attribution_system = SourceAttributionSystem()
    
    # Exercise 5: Confidence Scoring
    print("\nExercise 5: Confidence Scoring")
    confidence_scorer = ConfidenceScorer()
