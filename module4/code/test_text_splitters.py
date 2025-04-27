"""
Tests for text splitters.

This module contains tests for the text splitting strategies implemented in text_splitters.py.
"""

import unittest
from text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, SemanticTextSplitter

class TestCharacterTextSplitter(unittest.TestCase):
    """Tests for the CharacterTextSplitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = CharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separator="\n\n"
        )
    
    def test_split_text_small(self):
        """Test splitting a small text that fits in one chunk."""
        text = "This is a small text that should fit in one chunk."
        chunks = self.splitter.split_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_split_text_large(self):
        """Test splitting a large text into multiple chunks."""
        # Create a text with multiple paragraphs
        paragraphs = [
            "Paragraph 1 with some text.",
            "Paragraph 2 with some more text that is a bit longer than the first one.",
            "Paragraph 3 with even more text to ensure we have enough content to split into multiple chunks.",
            "Paragraph 4 with additional text to push us over the chunk size limit.",
            "Paragraph 5 with final text to ensure we have multiple chunks."
        ]
        text = "\n\n".join(paragraphs)
        
        chunks = self.splitter.split_text(text)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be no larger than the chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
        
        # Check for overlap
        if len(chunks) > 1:
            # The end of the first chunk should appear at the beginning of the second chunk
            overlap = chunks[0][-20:]
            self.assertTrue(chunks[1].startswith(overlap))
    
    def test_split_documents(self):
        """Test splitting a list of documents."""
        documents = [
            {
                'content': [
                    {'type': 'paragraph', 'text': 'Paragraph 1 with some text.'},
                    {'type': 'paragraph', 'text': 'Paragraph 2 with some more text that is a bit longer than the first one.'},
                    {'type': 'paragraph', 'text': 'Paragraph 3 with even more text to ensure we have enough content to split into multiple chunks.'}
                ],
                'metadata': {'title': 'Test Document'},
                'document_type': 'test'
            }
        ]
        
        chunked_docs = self.splitter.split_documents(documents)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunked_docs), 1)
        
        # Each chunked document should have the original metadata plus chunk info
        for i, doc in enumerate(chunked_docs):
            self.assertEqual(doc['document_type'], 'test')
            self.assertEqual(doc['metadata']['title'], 'Test Document')
            self.assertEqual(doc['metadata']['chunk_index'], i)
            self.assertEqual(doc['metadata']['chunk_count'], len(chunked_docs))

class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    """Tests for the RecursiveCharacterTextSplitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def test_split_text_small(self):
        """Test splitting a small text that fits in one chunk."""
        text = "This is a small text that should fit in one chunk."
        chunks = self.splitter.split_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_split_text_paragraphs(self):
        """Test splitting text with paragraphs."""
        # Create a text with multiple paragraphs
        paragraphs = [
            "Paragraph 1 with some text.",
            "Paragraph 2 with some more text that is a bit longer than the first one.",
            "Paragraph 3 with even more text to ensure we have enough content to split into multiple chunks.",
            "Paragraph 4 with additional text to push us over the chunk size limit.",
            "Paragraph 5 with final text to ensure we have multiple chunks."
        ]
        text = "\n\n".join(paragraphs)
        
        chunks = self.splitter.split_text(text)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be no larger than the chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
    
    def test_split_text_sentences(self):
        """Test splitting text with sentences."""
        # Create a text with a long paragraph of multiple sentences
        text = "Sentence 1 is short. Sentence 2 is a bit longer and has more words. Sentence 3 is even longer and contains more information that might be useful. Sentence 4 continues the paragraph with additional details and context. Sentence 5 concludes the paragraph with a summary of the main points."
        
        chunks = self.splitter.split_text(text)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be no larger than the chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
    
    def test_split_documents(self):
        """Test splitting a list of documents."""
        documents = [
            {
                'content': [
                    {'type': 'paragraph', 'text': 'Paragraph 1 with some text.'},
                    {'type': 'paragraph', 'text': 'Paragraph 2 with some more text that is a bit longer than the first one.'},
                    {'type': 'paragraph', 'text': 'Paragraph 3 with even more text to ensure we have enough content to split into multiple chunks.'}
                ],
                'metadata': {'title': 'Test Document'},
                'document_type': 'test'
            }
        ]
        
        chunked_docs = self.splitter.split_documents(documents)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunked_docs), 1)
        
        # Each chunked document should have the original metadata plus chunk info
        for i, doc in enumerate(chunked_docs):
            self.assertEqual(doc['document_type'], 'test')
            self.assertEqual(doc['metadata']['title'], 'Test Document')
            self.assertEqual(doc['metadata']['chunk_index'], i)
            self.assertEqual(doc['metadata']['chunk_count'], len(chunked_docs))

class TestSemanticTextSplitter(unittest.TestCase):
    """Tests for the SemanticTextSplitter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = SemanticTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            paragraph_separator="\n\n",
            sentence_separator=". "
        )
    
    def test_split_text_small(self):
        """Test splitting a small text that fits in one chunk."""
        text = "This is a small text that should fit in one chunk."
        chunks = self.splitter.split_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_split_text_paragraphs(self):
        """Test splitting text with paragraphs."""
        # Create a text with multiple paragraphs
        paragraphs = [
            "Paragraph 1 with some text.",
            "Paragraph 2 with some more text that is a bit longer than the first one.",
            "Paragraph 3 with even more text to ensure we have enough content to split into multiple chunks.",
            "Paragraph 4 with additional text to push us over the chunk size limit.",
            "Paragraph 5 with final text to ensure we have multiple chunks."
        ]
        text = "\n\n".join(paragraphs)
        
        chunks = self.splitter.split_text(text)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be no larger than the chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
    
    def test_split_text_sentences(self):
        """Test splitting text with sentences."""
        # Create a text with a long paragraph of multiple sentences
        text = "Sentence 1 is short. Sentence 2 is a bit longer and has more words. Sentence 3 is even longer and contains more information that might be useful. Sentence 4 continues the paragraph with additional details and context. Sentence 5 concludes the paragraph with a summary of the main points."
        
        chunks = self.splitter.split_text(text)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be no larger than the chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
    
    def test_split_documents(self):
        """Test splitting a list of documents."""
        documents = [
            {
                'content': [
                    {'type': 'paragraph', 'text': 'Paragraph 1 with some text.'},
                    {'type': 'paragraph', 'text': 'Paragraph 2 with some more text that is a bit longer than the first one.'},
                    {'type': 'paragraph', 'text': 'Paragraph 3 with even more text to ensure we have enough content to split into multiple chunks.'}
                ],
                'metadata': {'title': 'Test Document'},
                'document_type': 'test'
            }
        ]
        
        chunked_docs = self.splitter.split_documents(documents)
        
        # Should be split into multiple chunks
        self.assertGreater(len(chunked_docs), 1)
        
        # Each chunked document should have the original metadata plus chunk info
        for i, doc in enumerate(chunked_docs):
            self.assertEqual(doc['document_type'], 'test')
            self.assertEqual(doc['metadata']['title'], 'Test Document')
            self.assertEqual(doc['metadata']['chunk_index'], i)
            self.assertEqual(doc['metadata']['chunk_count'], len(chunked_docs))

if __name__ == '__main__':
    unittest.main()
