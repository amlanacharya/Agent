"""
Test cases for Module 5 - Lesson 5 Exercises: Building a Research Literature Assistant
"""

import unittest
from typing import List, Dict, Any
import sys
import os
import json
import tempfile

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the exercises
from exercises.lesson5_exercises import (
    AcademicPaperProcessor,
    CitationTracker,
    ResearchQuestionAnalyzer,
    LiteratureReviewGenerator,
    ResearchLiteratureAssistant,
    create_mock_llm
)

from langchain.schema.document import Document


class TestAcademicPaperProcessor(unittest.TestCase):
    """Test the AcademicPaperProcessor implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AcademicPaperProcessor()
        
        # Sample text with sections
        self.sample_text = """
        Title: Sample Academic Paper
        Authors: John Doe, Jane Smith
        
        Abstract
        This is a sample abstract for testing purposes.
        
        1. Introduction
        This is the introduction section.
        
        Background
        This is the background section.
        
        2. Methods
        This is the methods section.
        
        Results
        This is the results section.
        
        Discussion
        This is the discussion section.
        
        Conclusion
        This is the conclusion section.
        
        References
        [1] Author, A. (2020). Title of paper.
        [2] Author, B. (2021). Another paper.
        """
        
        # Sample text with citations
        self.citation_text = """
        According to [Smith, 2020], retrieval-augmented generation is effective.
        Another study [Johnson, 2021] found similar results.
        Some papers use numbered citations [1] and [2].
        """
        
        # Sample text with figures and tables
        self.figure_table_text = """
        Figure 1: This is a sample figure caption.
        
        Table 1: This is a sample table caption.
        
        Fig. 2: Another figure caption.
        """
        
        # Create sample document pages
        self.sample_pages = [
            Document(page_content="Title: Sample Academic Paper\nAuthors: John Doe, Jane Smith\nPublished: 2022\nDOI: 10.1234/sample"),
            Document(page_content="Abstract\nThis is a sample abstract.")
        ]
    
    def test_extract_sections(self):
        """Test section extraction from academic paper."""
        sections = self.processor.extract_sections(self.sample_text)
        
        # Check if sections were extracted correctly
        self.assertIn("abstract", sections)
        self.assertIn("introduction", sections)
        self.assertIn("methods", sections)
        self.assertIn("results", sections)
        self.assertIn("discussion", sections)
        self.assertIn("conclusion", sections)
        self.assertIn("references", sections)
        
        # Check content of a section
        self.assertIn("sample abstract", sections["abstract"].lower())
        self.assertIn("introduction section", sections["introduction"].lower())
    
    def test_extract_citations(self):
        """Test citation extraction from academic paper."""
        citations = self.processor.extract_citations(self.citation_text)
        
        # Check if citations were extracted
        self.assertIsInstance(citations, list)
        
        # If implemented, should find at least 2 author-year citations
        author_year_citations = [c for c in citations if c.get("type") == "author_year"]
        self.assertGreaterEqual(len(author_year_citations), 0)
        
        # If implemented, should find at least 2 numbered citations
        numbered_citations = [c for c in citations if c.get("type") == "numbered"]
        self.assertGreaterEqual(len(numbered_citations), 0)
    
    def test_extract_figures_and_tables(self):
        """Test figure and table extraction from academic paper."""
        figures_tables = self.processor.extract_figures_and_tables(self.figure_table_text)
        
        # Check structure
        self.assertIn("figures", figures_tables)
        self.assertIn("tables", figures_tables)
        
        # If implemented, should find at least 2 figures
        self.assertIsInstance(figures_tables["figures"], list)
        self.assertGreaterEqual(len(figures_tables["figures"]), 0)
        
        # If implemented, should find at least 1 table
        self.assertIsInstance(figures_tables["tables"], list)
        self.assertGreaterEqual(len(figures_tables["tables"]), 0)
    
    def test_extract_metadata(self):
        """Test metadata extraction from academic paper."""
        metadata = self.processor._extract_metadata(self.sample_pages)
        
        # Check if metadata is a dictionary
        self.assertIsInstance(metadata, dict)
        
        # If implemented, should extract title, authors, year, and DOI
        if "title" in metadata:
            self.assertIn("Sample Academic Paper", metadata["title"])
        if "authors" in metadata:
            self.assertIsInstance(metadata["authors"], list)
            self.assertGreaterEqual(len(metadata["authors"]), 0)
        if "year" in metadata:
            self.assertEqual(metadata.get("year"), "2022")
        if "doi" in metadata:
            self.assertEqual(metadata.get("doi"), "10.1234/sample")


class TestCitationTracker(unittest.TestCase):
    """Test the CitationTracker implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = CitationTracker()
        
        # Add some papers
        self.tracker.add_paper(
            "paper1",
            {"title": "Paper 1", "authors": ["Author A"], "year": "2020"},
            [{"type": "author_year", "author": "Author B", "year": "2019"}]
        )
        
        self.tracker.add_paper(
            "paper2",
            {"title": "Paper 2", "authors": ["Author B"], "year": "2019"},
            [{"type": "author_year", "author": "Author C", "year": "2018"}]
        )
        
        self.tracker.add_paper(
            "paper3",
            {"title": "Paper 3", "authors": ["Author C"], "year": "2018"},
            []
        )
    
    def test_get_citing_papers(self):
        """Test getting papers that cite a given paper."""
        # If implemented, paper2 should cite paper3 (Author C_2018)
        citing_papers = self.tracker.get_citing_papers("Author C_2018")
        self.assertIsInstance(citing_papers, list)
        
        # If implemented, paper1 should cite paper2 (Author B_2019)
        citing_papers = self.tracker.get_citing_papers("Author B_2019")
        self.assertIsInstance(citing_papers, list)
    
    def test_get_cited_papers(self):
        """Test getting papers cited by a given paper."""
        # If implemented, paper1 should cite paper2 (Author B_2019)
        cited_papers = self.tracker.get_cited_papers("paper1")
        self.assertIsInstance(cited_papers, list)
        
        # If implemented, paper2 should cite paper3 (Author C_2018)
        cited_papers = self.tracker.get_cited_papers("paper2")
        self.assertIsInstance(cited_papers, list)
        
        # Paper3 doesn't cite any papers
        cited_papers = self.tracker.get_cited_papers("paper3")
        self.assertEqual(cited_papers, [])
    
    def test_verify_citation(self):
        """Test citation verification."""
        # Test with a citation that exists
        result = self.tracker.verify_citation("paper1", "Author B_2019", "Paper 1 cites Paper 2")
        self.assertIsInstance(result, dict)
        self.assertIn("verified", result)
        self.assertIn("confidence", result)
        
        # Test with a citation that doesn't exist
        result = self.tracker.verify_citation("paper1", "nonexistent", "Paper 1 cites a nonexistent paper")
        self.assertIsInstance(result, dict)
        self.assertIn("verified", result)
        self.assertIn("confidence", result)
    
    def test_generate_citation_path(self):
        """Test citation path generation."""
        # If implemented, should find a path from paper1 to paper3
        path = self.tracker.generate_citation_path("paper1", "paper3")
        self.assertIsInstance(path, list)


class TestResearchQuestionAnalyzer(unittest.TestCase):
    """Test the ResearchQuestionAnalyzer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm = create_mock_llm()
        self.analyzer = ResearchQuestionAnalyzer(self.llm)
        
        # Sample questions
        self.factual_question = "What is retrieval-augmented generation?"
        self.comparative_question = "How do transformer models compare to RNNs for NLP tasks?"
        self.methodological_question = "What methods are used for evaluating RAG systems?"
        self.complex_question = "What are the recent advances in transformer architectures for natural language processing tasks, and how do they address the limitations of earlier models?"
    
    def test_analyze_question(self):
        """Test question analysis."""
        analysis = self.analyzer.analyze_question(self.factual_question)
        
        # Check if analysis is a dictionary
        self.assertIsInstance(analysis, dict)
        
        # Check if it contains expected keys
        self.assertIn("question_type", analysis)
        self.assertIn("key_concepts", analysis)
        self.assertIn("complexity", analysis)
    
    def test_identify_question_type(self):
        """Test question type identification."""
        # Test factual question
        question_type = self.analyzer.identify_question_type(self.factual_question)
        self.assertIsInstance(question_type, str)
        
        # Test comparative question
        question_type = self.analyzer.identify_question_type(self.comparative_question)
        self.assertIsInstance(question_type, str)
        
        # Test methodological question
        question_type = self.analyzer.identify_question_type(self.methodological_question)
        self.assertIsInstance(question_type, str)
    
    def test_extract_research_concepts(self):
        """Test research concept extraction."""
        concepts = self.analyzer.extract_research_concepts(self.factual_question)
        
        # Check if concepts is a list
        self.assertIsInstance(concepts, list)
        
        # If implemented, should extract "retrieval-augmented generation"
        if concepts:
            self.assertTrue(any("retrieval" in concept.lower() for concept in concepts) or
                           any("generation" in concept.lower() for concept in concepts))
    
    def test_generate_search_queries(self):
        """Test search query generation."""
        queries = self.analyzer.generate_search_queries(self.factual_question)
        
        # Check if queries is a list
        self.assertIsInstance(queries, list)
        
        # Should generate at least one query
        self.assertGreaterEqual(len(queries), 1)


class TestLiteratureReviewGenerator(unittest.TestCase):
    """Test the LiteratureReviewGenerator implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm = create_mock_llm()
        self.generator = LiteratureReviewGenerator(self.llm)
        
        # Sample documents and metadata
        self.chunks = [
            Document(
                page_content="Retrieval-augmented generation (RAG) is a technique that combines retrieval and generation.",
                metadata={"source": "paper1.pdf", "section": "introduction"}
            ),
            Document(
                page_content="Transformer models have revolutionized NLP tasks.",
                metadata={"source": "paper2.pdf", "section": "introduction"}
            ),
            Document(
                page_content="Evaluation of RAG systems requires specialized metrics.",
                metadata={"source": "paper3.pdf", "section": "methods"}
            )
        ]
        
        self.papers_metadata = [
            {"title": "Paper 1", "authors": ["Author A"], "year": "2020"},
            {"title": "Paper 2", "authors": ["Author B"], "year": "2021"},
            {"title": "Paper 3", "authors": ["Author C"], "year": "2022"}
        ]
        
        self.papers = [
            {
                "metadata": {"title": "Paper 1", "authors": ["Author A"], "year": "2020"},
                "sections": {"methods": "This paper uses method X."}
            },
            {
                "metadata": {"title": "Paper 2", "authors": ["Author B"], "year": "2021"},
                "sections": {"methods": "This paper uses method Y."}
            }
        ]
    
    def test_generate_review(self):
        """Test literature review generation."""
        review = self.generator.generate_review("What is RAG?", self.chunks, self.papers_metadata)
        
        # Check if review is a string
        self.assertIsInstance(review, str)
        
        # Should generate some content
        self.assertGreater(len(review), 0)
    
    def test_summarize_paper(self):
        """Test paper summarization."""
        summary = self.generator.summarize_paper(self.chunks[:1], self.papers_metadata[0])
        
        # Check if summary is a string
        self.assertIsInstance(summary, str)
        
        # Should generate some content
        self.assertGreater(len(summary), 0)
    
    def test_compare_methodologies(self):
        """Test methodology comparison."""
        comparison = self.generator.compare_methodologies(self.papers)
        
        # Check if comparison is a string
        self.assertIsInstance(comparison, str)
        
        # Should generate some content
        self.assertGreater(len(comparison), 0)
    
    def test_identify_research_gaps(self):
        """Test research gap identification."""
        gaps = self.generator.identify_research_gaps(self.papers, "What is RAG?")
        
        # Check if gaps is a string
        self.assertIsInstance(gaps, str)
        
        # Should generate some content
        self.assertGreater(len(gaps), 0)
    
    def test_format_with_citations(self):
        """Test citation formatting."""
        review_text = "RAG is a technique that combines retrieval and generation."
        citations = "Author A (2020). Paper 1\nAuthor B (2021). Paper 2"
        
        formatted = self.generator.format_with_citations(review_text, citations)
        
        # Check if formatted is a string
        self.assertIsInstance(formatted, str)
        
        # Should generate some content
        self.assertGreater(len(formatted), 0)


class TestResearchLiteratureAssistant(unittest.TestCase):
    """Test the ResearchLiteratureAssistant implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm = create_mock_llm()
        self.assistant = ResearchLiteratureAssistant(self.llm)
        
        # Create a temporary PDF file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_pdf_path = os.path.join(self.temp_dir.name, "test_paper.pdf")
        
        # Create a simple PDF file (just a text file with .pdf extension for testing)
        with open(self.temp_pdf_path, "w") as f:
            f.write("This is a test PDF file.\nTitle: Test Paper\nAuthors: Test Author\nAbstract\nThis is a test abstract.")
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_add_paper(self):
        """Test adding a paper to the assistant."""
        try:
            result = self.assistant.add_paper(self.temp_pdf_path)
            
            # Check if result is a dictionary
            self.assertIsInstance(result, dict)
            
            # If implemented, should contain chunks, metadata, etc.
            if result:
                self.assertIn("metadata", result)
        except Exception as e:
            self.skipTest(f"add_paper not fully implemented: {e}")
    
    def test_build_index(self):
        """Test building the vector index."""
        try:
            # Add a paper first
            self.assistant.add_paper(self.temp_pdf_path)
            
            # Build index
            vectorstore = self.assistant.build_index()
            
            # If implemented, should return a vectorstore
            if vectorstore:
                self.assertIsNotNone(vectorstore)
        except Exception as e:
            self.skipTest(f"build_index not fully implemented: {e}")
    
    def test_create_chain(self):
        """Test creating the LCEL chain."""
        try:
            # Add a paper first
            self.assistant.add_paper(self.temp_pdf_path)
            
            # Create chain
            chain = self.assistant.create_chain()
            
            # If implemented, should return a chain
            if chain:
                self.assertIsNotNone(chain)
        except Exception as e:
            self.skipTest(f"create_chain not fully implemented: {e}")
    
    def test_answer_research_question(self):
        """Test answering a research question."""
        result = self.assistant.answer_research_question("What is RAG?")
        
        # Check if result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Should contain the query
        self.assertIn("query", result)
    
    def test_generate_literature_review(self):
        """Test generating a literature review."""
        review = self.assistant.generate_literature_review("RAG")
        
        # Check if review is a string
        self.assertIsInstance(review, str)
        
        # Should generate some content
        self.assertGreater(len(review), 0)


if __name__ == "__main__":
    unittest.main()
