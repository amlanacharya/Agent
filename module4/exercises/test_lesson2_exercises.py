"""
Tests for Lesson 2: Chunking Strategies for Optimal Retrieval
"""

import unittest
import re
from lesson2_exercises import (
    SizeBasedChunker, 
    DelimiterBasedChunker, 
    RecursiveChunker, 
    SemanticChunker, 
    TokenAwareChunker,
    ChunkingEvaluator,
    ChunkingError
)

class TestSizeBasedChunker(unittest.TestCase):
    """Tests for the SizeBasedChunker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        This is a sample text for testing chunking strategies.
        It contains multiple sentences and paragraphs.
        
        This is the second paragraph with different content.
        We want to see how the chunker handles this.
        
        And here's a third paragraph to make the text longer.
        This should give us enough content to test various chunking parameters.
        """
        
        self.long_text = self.sample_text * 5  # Repeat to make it longer
    
    def test_initialization(self):
        """Test that the chunker initializes with valid parameters."""
        chunker = SizeBasedChunker(chunk_size=100, overlap=20)
        self.assertEqual(chunker.chunk_size, 100)
        self.assertEqual(chunker.overlap, 20)
        self.assertFalse(chunker.respect_sentences)
        
        # Test with respect_sentences=True
        chunker = SizeBasedChunker(chunk_size=100, overlap=20, respect_sentences=True)
        self.assertTrue(chunker.respect_sentences)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate exceptions."""
        # Test negative chunk size
        with self.assertRaises(ValueError):
            SizeBasedChunker(chunk_size=-100)
        
        # Test negative overlap
        with self.assertRaises(ValueError):
            SizeBasedChunker(chunk_size=100, overlap=-20)
        
        # Test overlap >= chunk_size
        with self.assertRaises(ValueError):
            SizeBasedChunker(chunk_size=100, overlap=100)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SizeBasedChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk_text("")
        self.assertEqual(chunks, [])
    
    def test_basic_chunking(self):
        """Test basic chunking without overlap."""
        chunker = SizeBasedChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
        
        # Check that the concatenated chunks equal the original text
        self.assertEqual("".join(chunks), self.sample_text)
    
    def test_chunking_with_overlap(self):
        """Test chunking with overlap."""
        chunker = SizeBasedChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
        
        # Check that there is overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            end_of_first = chunks[i][-20:]
            start_of_second = chunks[i+1][:20]
            
            # There should be some overlap (not necessarily exact due to chunk boundaries)
            self.assertTrue(
                end_of_first in chunks[i+1] or start_of_second in chunks[i],
                f"No overlap found between chunks {i} and {i+1}"
            )
    
    def test_respect_sentences(self):
        """Test chunking with respect_sentences=True."""
        chunker = SizeBasedChunker(chunk_size=100, overlap=0, respect_sentences=True)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that chunks tend to end with sentence boundaries
        sentence_end_pattern = r'[.!?]\s*$'
        sentence_endings = 0
        
        for chunk in chunks[:-1]:  # Skip the last chunk which might not end with a sentence
            if re.search(sentence_end_pattern, chunk):
                sentence_endings += 1
        
        # At least some chunks should end with sentence boundaries
        self.assertGreater(sentence_endings, 0)
    
    def test_error_handling(self):
        """Test that errors are properly caught and wrapped."""
        chunker = SizeBasedChunker(chunk_size=100, overlap=0)
        
        # Mock a situation that would cause an error
        original_method = chunker.chunk_text
        chunker.chunk_text = lambda text: 1/0  # This will raise ZeroDivisionError
        
        with self.assertRaises(ChunkingError):
            chunker.chunk_text("test")
        
        # Restore the original method
        chunker.chunk_text = original_method

class TestDelimiterBasedChunker(unittest.TestCase):
    """Tests for the DelimiterBasedChunker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        First paragraph with some content.
        Still the first paragraph.
        
        Second paragraph with different content.
        This is still part of the second paragraph.
        
        Third paragraph for more testing.
        
        Fourth paragraph to make the text longer.
        This should give us enough content to test.
        
        Fifth and final paragraph.
        """
    
    def test_initialization(self):
        """Test that the chunker initializes with valid parameters."""
        chunker = DelimiterBasedChunker(delimiter="\n\n", max_chunk_size=200, min_chunk_size=50)
        self.assertEqual(chunker.delimiter, "\n\n")
        self.assertEqual(chunker.max_chunk_size, 200)
        self.assertEqual(chunker.min_chunk_size, 50)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate exceptions."""
        # Test negative max_chunk_size
        with self.assertRaises(ValueError):
            DelimiterBasedChunker(max_chunk_size=-200)
        
        # Test negative min_chunk_size
        with self.assertRaises(ValueError):
            DelimiterBasedChunker(min_chunk_size=-50)
        
        # Test min_chunk_size > max_chunk_size
        with self.assertRaises(ValueError):
            DelimiterBasedChunker(max_chunk_size=100, min_chunk_size=200)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = DelimiterBasedChunker()
        chunks = chunker.chunk_text("")
        self.assertEqual(chunks, [])
    
    def test_basic_chunking(self):
        """Test basic delimiter-based chunking."""
        chunker = DelimiterBasedChunker(delimiter="\n\n", max_chunk_size=200)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
        
        # Check that delimiters are preserved in the output
        for chunk in chunks:
            if "\n\n" in chunk:
                # If a chunk contains the delimiter, it should be combining multiple sections
                sections = chunk.split("\n\n")
                self.assertGreater(len(sections), 1)
    
    def test_different_delimiter(self):
        """Test chunking with a different delimiter."""
        # Use period as delimiter
        chunker = DelimiterBasedChunker(delimiter=".", max_chunk_size=200)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
    
    def test_error_handling(self):
        """Test that errors are properly caught and wrapped."""
        chunker = DelimiterBasedChunker()
        
        # Mock a situation that would cause an error
        original_method = chunker.chunk_text
        chunker.chunk_text = lambda text: 1/0  # This will raise ZeroDivisionError
        
        with self.assertRaises(ChunkingError):
            chunker.chunk_text("test")
        
        # Restore the original method
        chunker.chunk_text = original_method

class TestRecursiveChunker(unittest.TestCase):
    """Tests for the RecursiveChunker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        # Section 1
        
        First paragraph with some content.
        Still the first paragraph.
        
        Second paragraph with different content.
        This is still part of the second paragraph.
        
        # Section 2
        
        Third paragraph for more testing.
        
        Fourth paragraph to make the text longer.
        This should give us enough content to test.
        
        # Section 3
        
        Fifth and final paragraph.
        """
    
    def test_initialization(self):
        """Test that the chunker initializes with valid parameters."""
        chunker = RecursiveChunker(delimiters=["\n\n", "\n", ". "], max_chunk_size=200)
        self.assertEqual(chunker.delimiters, ["\n\n", "\n", ". "])
        self.assertEqual(chunker.max_chunk_size, 200)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate exceptions."""
        # Test negative max_chunk_size
        with self.assertRaises(ValueError):
            RecursiveChunker(max_chunk_size=-200)
        
        # Test empty delimiters list
        with self.assertRaises(ValueError):
            RecursiveChunker(delimiters=[])
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = RecursiveChunker()
        chunks = chunker.chunk_text("")
        self.assertEqual(chunks, [])
    
    def test_basic_chunking(self):
        """Test basic recursive chunking."""
        chunker = RecursiveChunker(delimiters=["\n\n", "\n", ". "], max_chunk_size=200)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
    
    def test_recursive_behavior(self):
        """Test that the chunker behaves recursively."""
        # Create a text that requires recursive chunking
        large_paragraph = "This is a very long paragraph. " * 20  # ~600 characters
        text_with_large_paragraph = f"Small paragraph.\n\n{large_paragraph}\n\nAnother small paragraph."
        
        chunker = RecursiveChunker(delimiters=["\n\n", ". "], max_chunk_size=200)
        chunks = chunker.chunk_text(text_with_large_paragraph)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 2)  # Should be more than just splitting by \n\n
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
    
    def test_error_handling(self):
        """Test that errors are properly caught and wrapped."""
        chunker = RecursiveChunker()
        
        # Mock a situation that would cause an error
        original_method = chunker.chunk_text
        chunker.chunk_text = lambda text: 1/0  # This will raise ZeroDivisionError
        
        with self.assertRaises(ChunkingError):
            chunker.chunk_text("test")
        
        # Restore the original method
        chunker.chunk_text = original_method

class TestSemanticChunker(unittest.TestCase):
    """Tests for the SemanticChunker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        # Introduction
        
        This is a sample text for testing semantic chunking.
        It contains multiple sentences with different meanings.
        The semantic chunker should try to keep related sentences together.
        
        # Main Content
        
        Semantic chunking is important for maintaining context.
        When text is split arbitrarily, meaning can be lost.
        This is especially true for complex topics that require
        multiple sentences to fully explain.
        
        # Conclusion
        
        In summary, semantic chunking improves retrieval quality.
        It helps ensure that related information stays together.
        This leads to better answers from RAG systems.
        """
    
    def test_initialization(self):
        """Test that the chunker initializes with valid parameters."""
        chunker = SemanticChunker(max_chunk_size=200, min_chunk_size=50)
        self.assertEqual(chunker.max_chunk_size, 200)
        self.assertEqual(chunker.min_chunk_size, 50)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate exceptions."""
        # Test negative max_chunk_size
        with self.assertRaises(ValueError):
            SemanticChunker(max_chunk_size=-200)
        
        # Test negative min_chunk_size
        with self.assertRaises(ValueError):
            SemanticChunker(min_chunk_size=-50)
        
        # Test min_chunk_size > max_chunk_size
        with self.assertRaises(ValueError):
            SemanticChunker(max_chunk_size=100, min_chunk_size=200)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk_text("")
        self.assertEqual(chunks, [])
    
    def test_basic_chunking(self):
        """Test basic semantic chunking."""
        chunker = SemanticChunker(max_chunk_size=200)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Check that each chunk is no larger than the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)
    
    def test_sentence_preservation(self):
        """Test that the chunker preserves sentences when possible."""
        chunker = SemanticChunker(max_chunk_size=200)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Count how many chunks end with sentence-ending punctuation
        sentence_endings = 0
        for chunk in chunks:
            if re.search(r'[.!?]$', chunk.strip()):
                sentence_endings += 1
        
        # Most chunks should end with sentence-ending punctuation
        self.assertGreaterEqual(sentence_endings, len(chunks) // 2)
    
    def test_paragraph_handling(self):
        """Test that the chunker handles paragraphs appropriately."""
        chunker = SemanticChunker(max_chunk_size=500)  # Larger size to fit paragraphs
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check if paragraphs are preserved
        paragraphs_preserved = 0
        for chunk in chunks:
            if "\n\n" in chunk:
                paragraphs_preserved += 1
        
        # At least some chunks should contain paragraph breaks
        self.assertGreater(paragraphs_preserved, 0)
    
    def test_error_handling(self):
        """Test that errors are properly caught and wrapped."""
        chunker = SemanticChunker()
        
        # Mock a situation that would cause an error
        original_method = chunker.chunk_text
        chunker.chunk_text = lambda text: 1/0  # This will raise ZeroDivisionError
        
        with self.assertRaises(ChunkingError):
            chunker.chunk_text("test")
        
        # Restore the original method
        chunker.chunk_text = original_method

class TestTokenAwareChunker(unittest.TestCase):
    """Tests for the TokenAwareChunker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        This is a sample text for testing token-aware chunking.
        It contains multiple sentences that will be tokenized.
        The token-aware chunker should split based on token count.
        
        This is another paragraph with different content.
        Token-aware chunking is important for LLM context windows.
        It ensures that chunks fit within token limits.
        
        Here's a final paragraph to make the text longer.
        This should give us enough tokens to test various parameters.
        """
        
        # Simple word tokenizer for testing
        self.simple_tokenizer = lambda text: text.split()
    
    def test_initialization(self):
        """Test that the chunker initializes with valid parameters."""
        chunker = TokenAwareChunker(max_tokens=100, overlap_tokens=20)
        self.assertEqual(chunker.max_tokens, 100)
        self.assertEqual(chunker.overlap_tokens, 20)
        
        # Test with custom tokenizer
        chunker = TokenAwareChunker(max_tokens=100, overlap_tokens=20, tokenizer=self.simple_tokenizer)
        self.assertEqual(chunker.tokenizer, self.simple_tokenizer)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate exceptions."""
        # Test negative max_tokens
        with self.assertRaises(ValueError):
            TokenAwareChunker(max_tokens=-100)
        
        # Test negative overlap_tokens
        with self.assertRaises(ValueError):
            TokenAwareChunker(max_tokens=100, overlap_tokens=-20)
        
        # Test overlap_tokens >= max_tokens
        with self.assertRaises(ValueError):
            TokenAwareChunker(max_tokens=100, overlap_tokens=100)
    
    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = TokenAwareChunker(tokenizer=self.simple_tokenizer)
        chunks = chunker.chunk_text("")
        self.assertEqual(chunks, [])
    
    def test_basic_chunking(self):
        """Test basic token-aware chunking."""
        chunker = TokenAwareChunker(max_tokens=20, overlap_tokens=0, tokenizer=self.simple_tokenizer)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Check that each chunk has no more than max_tokens
        for chunk in chunks:
            tokens = self.simple_tokenizer(chunk)
            self.assertLessEqual(len(tokens), 20)
    
    def test_chunking_with_overlap(self):
        """Test token-aware chunking with overlap."""
        chunker = TokenAwareChunker(max_tokens=20, overlap_tokens=5, tokenizer=self.simple_tokenizer)
        chunks = chunker.chunk_text(self.sample_text)
        
        # Check that we have multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Check for overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            tokens1 = self.simple_tokenizer(chunks[i])
            tokens2 = self.simple_tokenizer(chunks[i+1])
            
            # Get the last 5 tokens of the first chunk
            last_tokens = tokens1[-5:] if len(tokens1) >= 5 else tokens1
            
            # Get the first 5 tokens of the second chunk
            first_tokens = tokens2[:5] if len(tokens2) >= 5 else tokens2
            
            # Check for overlap
            overlap_found = False
            for token in last_tokens:
                if token in first_tokens:
                    overlap_found = True
                    break
            
            self.assertTrue(overlap_found, f"No overlap found between chunks {i} and {i+1}")
    
    def test_error_handling(self):
        """Test that errors are properly caught and wrapped."""
        chunker = TokenAwareChunker(tokenizer=self.simple_tokenizer)
        
        # Mock a situation that would cause an error
        original_method = chunker.chunk_text
        chunker.chunk_text = lambda text: 1/0  # This will raise ZeroDivisionError
        
        with self.assertRaises(ChunkingError):
            chunker.chunk_text("test")
        
        # Restore the original method
        chunker.chunk_text = original_method

class TestChunkingEvaluator(unittest.TestCase):
    """Tests for the ChunkingEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """
        # Document Chunking Strategies
        
        Effective document chunking is critical for retrieval-augmented generation systems.
        The way documents are split affects both retrieval accuracy and generation quality.
        
        ## Size-Based Chunking
        
        The simplest approach is to split documents based on a fixed size. This works well
        for homogeneous documents but can break semantic connections.
        
        ## Delimiter-Based Chunking
        
        This strategy splits text based on specific delimiters like paragraphs or sections.
        It preserves natural document boundaries but can result in uneven chunk sizes.
        """
        
        # Create some test chunks
        self.uniform_chunks = [
            "This is chunk one with exactly fifty characters in it.",
            "This is chunk two with exactly fifty characters in it.",
            "This is chunk three with exactly fifty characters in."
        ]
        
        self.variable_chunks = [
            "Short chunk.",
            "This is a medium-sized chunk with some content in it.",
            "This is a much longer chunk that contains multiple sentences. It has significantly more content than the other chunks. This should result in a lower uniformity score when evaluated."
        ]
        
        self.semantic_chunks = [
            "This is a complete sentence. This is another complete sentence.",
            "This paragraph has multiple sentences. They are all complete. The paragraph is preserved intact.",
            "Another paragraph with complete sentences. It ends properly."
        ]
        
        self.broken_chunks = [
            "This sentence is cut in the",
            "middle and doesn't make sense on its own.",
            "This paragraph is also split across",
            "multiple chunks which breaks the semantic meaning."
        ]
    
    def test_initialization(self):
        """Test that the evaluator initializes correctly."""
        evaluator = ChunkingEvaluator()
        self.assertEqual(evaluator.metrics, {})
    
    def test_size_distribution_evaluation(self):
        """Test the size distribution evaluation."""
        evaluator = ChunkingEvaluator()
        
        # Test with uniform chunks
        uniform_metrics = evaluator.evaluate_size_distribution(self.uniform_chunks)
        self.assertAlmostEqual(uniform_metrics["size_uniformity"], 1.0, places=1)
        
        # Test with variable chunks
        variable_metrics = evaluator.evaluate_size_distribution(self.variable_chunks)
        self.assertLess(variable_metrics["size_uniformity"], uniform_metrics["size_uniformity"])
        
        # Test with empty chunks
        empty_metrics = evaluator.evaluate_size_distribution([])
        self.assertEqual(empty_metrics["avg_size"], 0)
        self.assertEqual(empty_metrics["min_size"], 0)
        self.assertEqual(empty_metrics["max_size"], 0)
        self.assertEqual(empty_metrics["std_dev"], 0)
        self.assertEqual(empty_metrics["size_uniformity"], 0)
    
    def test_semantic_coherence_evaluation(self):
        """Test the semantic coherence evaluation."""
        evaluator = ChunkingEvaluator()
        
        # Test with semantically coherent chunks
        semantic_metrics = evaluator.evaluate_semantic_coherence(self.semantic_chunks)
        self.assertGreater(semantic_metrics["complete_sentences_ratio"], 0.5)
        
        # Test with broken chunks
        broken_metrics = evaluator.evaluate_semantic_coherence(self.broken_chunks)
        self.assertLess(broken_metrics["complete_sentences_ratio"], semantic_metrics["complete_sentences_ratio"])
        
        # Test with empty chunks
        empty_metrics = evaluator.evaluate_semantic_coherence([])
        self.assertEqual(empty_metrics["complete_sentences_ratio"], 0)
        self.assertEqual(empty_metrics["paragraph_preservation"], 0)
    
    def test_compare_strategies(self):
        """Test the strategy comparison functionality."""
        evaluator = ChunkingEvaluator()
        
        # Create mock chunkers
        class MockChunker:
            def __init__(self, chunks):
                self.chunks = chunks
            
            def chunk_text(self, text):
                return self.chunks
        
        uniform_chunker = MockChunker(self.uniform_chunks)
        variable_chunker = MockChunker(self.variable_chunks)
        semantic_chunker = MockChunker(self.semantic_chunks)
        broken_chunker = MockChunker(self.broken_chunks)
        error_chunker = MockChunker(None)  # Will cause an error
        error_chunker.chunk_text = lambda text: 1/0  # This will raise ZeroDivisionError
        
        # Compare strategies
        results = evaluator.compare_strategies(
            self.sample_text,
            {
                "uniform": uniform_chunker,
                "variable": variable_chunker,
                "semantic": semantic_chunker,
                "broken": broken_chunker,
                "error": error_chunker
            }
        )
        
        # Check that all strategies are in the results
        self.assertEqual(set(results.keys()), {"uniform", "variable", "semantic", "broken", "error"})
        
        # Check that error strategy has an error message
        self.assertIn("error", results["error"])
        
        # Check that other strategies have metrics
        for name in ["uniform", "variable", "semantic", "broken"]:
            self.assertIn("chunk_count", results[name])
            self.assertIn("size_metrics", results[name])
            self.assertIn("semantic_metrics", results[name])
            self.assertIn("overall_score", results[name])
        
        # Check that semantic chunker has a higher score than broken chunker
        self.assertGreater(results["semantic"]["overall_score"], results["broken"]["overall_score"])

if __name__ == '__main__':
    unittest.main()
