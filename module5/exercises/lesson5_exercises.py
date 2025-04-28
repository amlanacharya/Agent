"""
Exercises for Module 5 - Lesson 5: Building a Research Literature Assistant

This module contains exercises for implementing a specialized RAG system for academic papers
and research literature, including:
- Academic paper processing with section extraction
- Citation tracking and verification
- Research question analysis
- Literature review generation
- A complete Research Literature Assistant using LCEL
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Set
import re
import json
import os
from collections import defaultdict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate

# Check if required packages are available
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# Exercise 1: Implement Academic Paper Processing
class AcademicPaperProcessor:
    """Process academic papers with specialized techniques."""
    
    def __init__(self):
        self.section_headers = [
            "abstract", "introduction", "background", "related work", 
            "methodology", "methods", "experiments", "results", 
            "discussion", "conclusion", "references"
        ]
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from academic paper.
        
        Args:
            text: Full text of the academic paper
            
        Returns:
            Dictionary mapping section names to section content
        """
        # TODO: Implement section extraction based on common academic paper structure
        # 1. Split text into lines
        # 2. Identify section headers based on self.section_headers
        # 3. Extract content for each section
        # 4. Return a dictionary mapping section names to content
        
        sections = {}
        current_section = "unknown"
        current_content = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a section header
            line_lower = line.lower().strip()
            is_header = False
            
            # Check if line matches any section header pattern
            for header in self.section_headers:
                # Match exact header or header with number (e.g., "1. Introduction")
                if (line_lower == header or
                    re.match(r'^\d+\.?\s+' + header + r'$', line_lower) or
                    re.match(r'^' + header + r'\s*$', line_lower)):
                    
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = header
                    current_content = []
                    is_header = True
                    break
            
            # If not a header, add to current section content
            if not is_header:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract citations from academic paper.
        
        Args:
            text: Full text of the academic paper
            
        Returns:
            List of extracted citations with metadata
        """
        # TODO: Implement citation extraction
        # 1. Extract citations in format [Author, Year]
        # 2. Extract citations in format [1], [2], etc.
        # 3. Return a list of dictionaries with citation information
        
        citations = []
        
        # Your implementation here
        
        return citations
    
    def extract_figures_and_tables(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract figures and tables from academic paper.
        
        Args:
            text: Full text of the academic paper
            
        Returns:
            Dictionary with lists of figures and tables
        """
        # TODO: Implement figure and table extraction
        # 1. Extract figures (Figure X: Caption)
        # 2. Extract tables (Table X: Caption)
        # 3. Return a dictionary with lists of figures and tables
        
        figures = []
        tables = []
        
        # Your implementation here
        
        return {
            "figures": figures,
            "tables": tables
        }
    
    def process_paper(self, file_path: str) -> Dict[str, Any]:
        """
        Process academic paper from PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with processed paper data
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Extract metadata
        metadata = self._extract_metadata(pages)
        
        # Extract full text
        full_text = "\n\n".join([page.page_content for page in pages])
        
        # Extract sections
        sections = self.extract_sections(full_text)
        
        # Extract citations
        citations = self.extract_citations(full_text)
        
        # Extract figures and tables
        figures_tables = self.extract_figures_and_tables(full_text)
        
        # Create specialized chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for section_name, section_text in sections.items():
            section_chunks = splitter.create_documents(
                texts=[section_text],
                metadatas=[{
                    "source": file_path,
                    "section": section_name,
                    **metadata
                }]
            )
            chunks.extend(section_chunks)
        
        return {
            "chunks": chunks,
            "metadata": metadata,
            "citations": citations,
            "figures": figures_tables["figures"],
            "tables": figures_tables["tables"],
            "sections": sections
        }
    
    def _extract_metadata(self, pages: List[Document]) -> Dict[str, Any]:
        """
        Extract metadata from academic paper.
        
        Args:
            pages: List of document pages
            
        Returns:
            Dictionary with extracted metadata
        """
        # TODO: Implement metadata extraction
        # 1. Extract title (usually the first line)
        # 2. Extract authors (usually after title)
        # 3. Extract publication year
        # 4. Extract DOI if present
        # 5. Return a dictionary with the extracted metadata
        
        metadata = {}
        
        # Your implementation here
        
        return metadata


# Exercise 2: Implement Citation Tracking and Verification
class CitationTracker:
    """Track and verify citations across academic papers."""
    
    def __init__(self):
        self.citation_graph = {}  # Paper ID -> [Cited Paper IDs]
        self.papers = {}  # Paper ID -> Paper Metadata
        self.citing_papers = defaultdict(list)  # Paper ID -> [Papers that cite it]
    
    def add_paper(self, paper_id: str, metadata: Dict[str, Any], citations: List[Dict[str, Any]]):
        """
        Add a paper and its citations to the tracker.
        
        Args:
            paper_id: Unique identifier for the paper
            metadata: Paper metadata
            citations: List of citations in the paper
        """
        # TODO: Implement adding a paper to the citation tracker
        # 1. Store paper metadata
        # 2. Extract cited paper IDs from citations
        # 3. Store outgoing citations
        # 4. Update incoming citations
        
        # Your implementation here
    
    def get_citing_papers(self, paper_id: str) -> List[str]:
        """
        Get papers that cite the given paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            List of paper IDs that cite the given paper
        """
        # TODO: Implement citation lookup
        # Return a list of paper IDs that cite the given paper
        
        return self.citing_papers.get(paper_id, [])
    
    def get_cited_papers(self, paper_id: str) -> List[str]:
        """
        Get papers cited by the given paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            List of paper IDs cited by the given paper
        """
        # TODO: Implement reference lookup
        # Return a list of paper IDs cited by the given paper
        
        return self.citation_graph.get(paper_id, [])
    
    def verify_citation(self, source_id: str, target_id: str, claim: str) -> Dict[str, Any]:
        """
        Verify if a citation supports a claim.
        
        Args:
            source_id: ID of the citing paper
            target_id: ID of the cited paper
            claim: The claim to verify
            
        Returns:
            Verification result with confidence score
        """
        # TODO: Implement citation verification
        # 1. Check if the citation exists
        # 2. In a real implementation, this would involve:
        #    - Retrieving the content of the cited paper
        #    - Using an LLM to check if the claim is supported by the content
        #    - Calculating a confidence score
        # 3. Return a dictionary with verification results
        
        # Your implementation here
        
        return {
            "verified": False,
            "confidence": 0.0,
            "reason": "Not implemented"
        }
    
    def generate_citation_path(self, start_id: str, end_id: str) -> List[str]:
        """
        Find citation path between papers using breadth-first search.
        
        Args:
            start_id: ID of the starting paper
            end_id: ID of the target paper
            
        Returns:
            List of paper IDs forming a path from start to end, or empty list if no path exists
        """
        # TODO: Implement citation path finding
        # 1. Check if papers exist
        # 2. Implement breadth-first search to find a path
        # 3. Return the path as a list of paper IDs
        
        # Your implementation here
        
        return []


# Exercise 3: Implement Research Question Analysis
class ResearchQuestionAnalyzer:
    """Analyze and decompose research questions."""
    
    def __init__(self, llm):
        """
        Initialize the research question analyzer.
        
        Args:
            llm: Language model for analysis
        """
        self.llm = llm
        
        # Define question types
        self.question_types = {
            "factual": "Seeking specific facts or information",
            "conceptual": "Seeking explanation of concepts or theories",
            "comparative": "Comparing multiple concepts, methods, or studies",
            "causal": "Seeking cause-effect relationships",
            "methodological": "Asking about research methods or techniques",
            "gap": "Identifying research gaps or future directions",
            "synthesis": "Requiring synthesis of multiple sources"
        }
        
        # TODO: Create analysis prompt
        # Create a ChatPromptTemplate for analyzing research questions
        
        self.analysis_prompt = None  # Replace with your implementation
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze a research question to identify key components.
        
        Args:
            question: The research question to analyze
            
        Returns:
            Dictionary with question analysis
        """
        # TODO: Implement research question analysis
        # 1. Invoke the LLM with the analysis prompt
        # 2. Parse the response as JSON
        # 3. Fall back to simple analysis if JSON parsing fails
        # 4. Return the analysis as a dictionary
        
        # Your implementation here
        
        return self._simple_analysis(question)
    
    def identify_question_type(self, question: str) -> str:
        """
        Identify the type of research question.
        
        Args:
            question: The research question
            
        Returns:
            Question type
        """
        # TODO: Implement question type classification
        # 1. Analyze the question text
        # 2. Determine the question type based on keywords or patterns
        # 3. Return the question type
        
        # Your implementation here
        
        return "factual"  # Default type
    
    def extract_research_concepts(self, question: str) -> List[str]:
        """
        Extract key research concepts from the question.
        
        Args:
            question: The research question
            
        Returns:
            List of key concepts
        """
        # TODO: Implement concept extraction
        # 1. Remove common question words and stopwords
        # 2. Tokenize and filter
        # 3. Extract potential noun phrases
        # 4. Return a list of key concepts
        
        # Your implementation here
        
        return []
    
    def generate_search_queries(self, question: str) -> List[str]:
        """
        Generate multiple search queries for the research question.
        
        Args:
            question: The research question
            
        Returns:
            List of search queries
        """
        # TODO: Implement search query generation
        # 1. Extract concepts from the question
        # 2. Generate a base query
        # 3. Create variations of the query
        # 4. Return a list of search queries
        
        # Your implementation here
        
        return [question]  # Default to just using the original question
    
    def _simple_analysis(self, question: str) -> Dict[str, Any]:
        """
        Perform simple rule-based analysis when LLM fails.
        
        Args:
            question: The research question
            
        Returns:
            Dictionary with question analysis
        """
        question_type = self.identify_question_type(question)
        concepts = self.extract_research_concepts(question)
        queries = self.generate_search_queries(question)
        
        # Determine complexity based on length and concepts
        if len(question.split()) > 20 or len(concepts) > 4:
            complexity = "complex"
        elif len(question.split()) > 10 or len(concepts) > 2:
            complexity = "moderate"
        else:
            complexity = "simple"
        
        return {
            "question_type": question_type,
            "key_concepts": concepts,
            "required_background": [f"{concept} fundamentals" for concept in concepts[:2]],
            "search_queries": queries,
            "complexity": complexity
        }


# Exercise 4: Implement Literature Review Generation
class LiteratureReviewGenerator:
    """Generate literature reviews from multiple papers."""
    
    def __init__(self, llm):
        """
        Initialize the literature review generator.
        
        Args:
            llm: Language model for review generation
        """
        self.llm = llm
        
        # TODO: Create prompts for literature review generation
        # 1. Create a review generation prompt
        # 2. Create a paper summarization prompt
        # 3. Create a methodology comparison prompt
        # 4. Create a research gap prompt
        # 5. Create a citation formatting prompt
        
        # Your implementation here
        
        self.review_prompt = None  # Replace with your implementation
        self.summary_prompt = None  # Replace with your implementation
        self.methodology_prompt = None  # Replace with your implementation
        self.gap_prompt = None  # Replace with your implementation
        self.citation_prompt = None  # Replace with your implementation
    
    def generate_review(self, question: str, relevant_chunks: List[Document], papers_metadata: List[Dict[str, Any]]) -> str:
        """
        Generate a literature review for a research question.
        
        Args:
            question: The research question
            relevant_chunks: List of relevant document chunks
            papers_metadata: Metadata for the papers
            
        Returns:
            Generated literature review
        """
        # TODO: Implement literature review generation
        # 1. Format paper excerpts
        # 2. Format paper metadata
        # 3. Generate the review using the LLM
        # 4. Format with citations
        # 5. Return the formatted review
        
        # Your implementation here
        
        return "Literature review not implemented."
    
    def summarize_paper(self, paper_chunks: List[Document], metadata: Dict[str, Any]) -> str:
        """
        Summarize a single paper.
        
        Args:
            paper_chunks: Chunks from the paper
            metadata: Paper metadata
            
        Returns:
            Paper summary
        """
        # TODO: Implement paper summarization
        # 1. Extract metadata (title, authors, year)
        # 2. Format excerpts
        # 3. Generate summary using the LLM
        # 4. Return the summary
        
        # Your implementation here
        
        return "Paper summary not implemented."
    
    def compare_methodologies(self, papers: List[Dict[str, Any]]) -> str:
        """
        Compare methodologies across papers.
        
        Args:
            papers: List of papers with methodology sections
            
        Returns:
            Methodology comparison
        """
        # TODO: Implement methodology comparison
        # 1. Format methodologies from each paper
        # 2. Generate comparison using the LLM
        # 3. Return the comparison
        
        # Your implementation here
        
        return "Methodology comparison not implemented."
    
    def identify_research_gaps(self, papers: List[Dict[str, Any]], question: str) -> str:
        """
        Identify research gaps related to the question.
        
        Args:
            papers: List of papers
            question: Research question
            
        Returns:
            Research gaps analysis
        """
        # TODO: Implement research gap identification
        # 1. Extract relevant excerpts from papers
        # 2. Generate gaps analysis using the LLM
        # 3. Return the analysis
        
        # Your implementation here
        
        return "Research gaps analysis not implemented."
    
    def format_with_citations(self, review_text: str, citations: str) -> str:
        """
        Format review text with proper citations.
        
        Args:
            review_text: The review text
            citations: Available citations
            
        Returns:
            Formatted review with citations
        """
        # TODO: Implement citation formatting
        # 1. Generate formatted review using the LLM
        # 2. Return the formatted review
        
        # Your implementation here
        
        return review_text  # Return unformatted text as fallback


# Exercise 5: Implement a Complete Research Literature Assistant
class ResearchLiteratureAssistant:
    """Complete Research Literature Assistant using LCEL."""
    
    def __init__(self, llm, embedding_model=None):
        """
        Initialize the research literature assistant.
        
        Args:
            llm: Language model for various tasks
            embedding_model: Model to generate embeddings (optional)
        """
        self.llm = llm
        
        # Initialize embedding model
        if embedding_model:
            self.embedding_model = embedding_model
        elif HUGGINGFACE_AVAILABLE:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        else:
            # Mock embedding model
            class MockEmbeddings:
                def embed_documents(self, texts):
                    return [[0.1] * 384 for _ in texts]
                
                def embed_query(self, text):
                    return [0.1] * 384
            
            self.embedding_model = MockEmbeddings()
        
        # Initialize components
        self.paper_processor = AcademicPaperProcessor()
        self.citation_tracker = CitationTracker()
        self.question_analyzer = ResearchQuestionAnalyzer(llm)
        self.review_generator = LiteratureReviewGenerator(llm)
        self.vectorstore = None
        self.papers = {}
    
    def add_paper(self, file_path: str) -> Dict[str, Any]:
        """
        Process and add a paper to the assistant.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Processed paper data
        """
        # TODO: Implement paper processing and indexing
        # 1. Process the paper using the paper processor
        # 2. Generate a paper ID
        # 3. Add to citation tracker
        # 4. Store the paper data
        # 5. Add chunks to vectorstore if it exists
        # 6. Return the paper data
        
        # Your implementation here
        
        return {}
    
    def build_index(self):
        """
        Build vector index from processed papers.
        
        Returns:
            The built vectorstore
        """
        # TODO: Implement index building
        # 1. Collect all chunks from papers
        # 2. Create vectorstore from chunks
        # 3. Return the vectorstore
        
        # Your implementation here
        
        return None
    
    def create_chain(self):
        """
        Create LCEL chain for the assistant.
        
        Returns:
            The LCEL chain
        """
        # TODO: Implement LCEL chain
        # 1. Ensure vectorstore is built
        # 2. Create retriever
        # 3. Define chain components:
        #    - Question analysis
        #    - Generate search queries
        #    - Retrieve documents
        #    - Extract paper metadata
        #    - Generate literature review
        # 4. Create and return the chain
        
        # Your implementation here
        
        return None
    
    def answer_research_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a research question using the assistant.
        
        Args:
            question: The research question
            
        Returns:
            Dictionary with question analysis, retrieved documents, and generated review
        """
        # TODO: Implement research question answering
        # 1. Create chain if not already created
        # 2. Run the chain
        # 3. Return the result
        
        # Your implementation here
        
        return {
            "query": question,
            "review": "Research question answering not implemented."
        }
    
    def generate_literature_review(self, topic: str) -> str:
        """
        Generate a literature review on a topic.
        
        Args:
            topic: The research topic
            
        Returns:
            Generated literature review
        """
        # TODO: Implement literature review generation
        # 1. Format topic as a research question if needed
        # 2. Use answer_research_question method
        # 3. Return the review
        
        # Your implementation here
        
        return "Literature review generation not implemented."


# Helper function to create a mock LLM for testing
def create_mock_llm():
    """Create a mock LLM for testing."""
    class MockLLM:
        def invoke(self, prompt):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            
            # Return simple responses based on prompt content
            if "analyze" in str(prompt).lower():
                return MockResponse(json.dumps({
                    "question_type": "factual",
                    "key_concepts": ["research", "literature", "assistant"],
                    "required_background": ["academic papers", "citation analysis"],
                    "search_queries": ["research literature assistant", "academic paper processing"],
                    "complexity": "moderate"
                }))
            elif "literature review" in str(prompt).lower():
                return MockResponse("This is a mock literature review about the requested topic.")
            elif "summarize" in str(prompt).lower():
                return MockResponse("This is a mock summary of the paper.")
            elif "compare" in str(prompt).lower():
                return MockResponse("This is a mock comparison of methodologies.")
            elif "gaps" in str(prompt).lower():
                return MockResponse("This is a mock analysis of research gaps.")
            elif "citation" in str(prompt).lower():
                return MockResponse("This is a mock formatted review with citations [Author, 2023].")
            else:
                return MockResponse("This is a mock response.")
    
    return MockLLM()


# Example usage
def example_usage():
    """Example of how to use the Research Literature Assistant."""
    # Create a mock LLM for testing
    llm = create_mock_llm()
    
    # Create the Research Literature Assistant
    assistant = ResearchLiteratureAssistant(llm)
    
    # Test question analysis
    question = "What are the recent advances in transformer models for natural language processing?"
    analysis = assistant.question_analyzer.analyze_question(question)
    print(f"Question Analysis: {analysis}")
    
    # Test literature review generation
    review = assistant.generate_literature_review("transformer models")
    print(f"Literature Review: {review}")


if __name__ == "__main__":
    # Run example usage
    example_usage()
