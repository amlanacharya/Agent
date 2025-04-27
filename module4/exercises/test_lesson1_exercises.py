"""
Tests for Lesson 1 exercises.

This module contains tests for the document processing exercises in lesson1_exercises.py.
"""

import os
import unittest
import tempfile
from lesson1_exercises import (
    BaseDocumentLoader, CSVLoader, HTMLLoader, 
    TextNormalizer, DocumentProcessor, DocumentLoadingError
)

class TestCSVLoader(unittest.TestCase):
    """Tests for the CSVLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = CSVLoader()
        
        # Create a temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.temp_file.write(b"Name,Age,City\nJohn,30,New York\nJane,25,San Francisco\nBob,40,Chicago")
        self.temp_file.close()
    
    def tearDown(self):
        """Tear down test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_load_csv_file(self):
        """Test loading a CSV file."""
        result = self.loader.load(self.temp_file.name)
        
        # Check document type
        self.assertEqual(result['document_type'], 'csv')
        
        # Check metadata
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['file_name'], os.path.basename(self.temp_file.name))
        self.assertEqual(result['metadata']['extension'], 'csv')
        self.assertIn('headers', result['metadata'])
        self.assertEqual(result['metadata']['headers'], ['Name', 'Age', 'City'])
        self.assertEqual(result['metadata']['row_count'], 3)
        
        # Check content
        self.assertIn('content', result)
        self.assertEqual(len(result['content']), 3)  # Three rows
        self.assertEqual(result['content'][0]['Name'], 'John')
        self.assertEqual(result['content'][0]['Age'], '30')
        self.assertEqual(result['content'][0]['City'], 'New York')
    
    def test_load_csv_with_different_delimiter(self):
        """Test loading a CSV file with a different delimiter."""
        # Create a temporary CSV file with semicolon delimiter
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        temp_file.write(b"Name;Age;City\nJohn;30;New York\nJane;25;San Francisco\nBob;40;Chicago")
        temp_file.close()
        
        try:
            # Create loader with semicolon delimiter
            semicolon_loader = CSVLoader(delimiter=';')
            result = semicolon_loader.load(temp_file.name)
            
            # Check content
            self.assertEqual(len(result['content']), 3)  # Three rows
            self.assertEqual(result['content'][0]['Name'], 'John')
            self.assertEqual(result['content'][0]['Age'], '30')
            self.assertEqual(result['content'][0]['City'], 'New York')
        finally:
            os.unlink(temp_file.name)
    
    def test_load_csv_without_header(self):
        """Test loading a CSV file without a header row."""
        # Create a temporary CSV file without header
        temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        temp_file.write(b"John,30,New York\nJane,25,San Francisco\nBob,40,Chicago")
        temp_file.close()
        
        try:
            # Create loader with no header
            no_header_loader = CSVLoader(has_header=False)
            result = no_header_loader.load(temp_file.name)
            
            # Check content
            self.assertEqual(len(result['content']), 3)  # Three rows
            self.assertEqual(result['content'][0]['column_0'], 'John')
            self.assertEqual(result['content'][0]['column_1'], '30')
            self.assertEqual(result['content'][0]['column_2'], 'New York')
        finally:
            os.unlink(temp_file.name)

class TestHTMLLoader(unittest.TestCase):
    """Tests for the HTMLLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = HTMLLoader()
        
        # Create a temporary HTML file for testing
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
    <meta name="author" content="Test Author">
    <meta name="description" content="A test HTML document">
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph with <a href="https://example.com">a link</a>.</p>
    <h2>Subheading</h2>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
    <p>Another paragraph with some text.</p>
    <img src="image.jpg" alt="Test Image">
</body>
</html>"""
        self.temp_file.write(html_content.encode('utf-8'))
        self.temp_file.close()
    
    def tearDown(self):
        """Tear down test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_load_html_file(self):
        """Test loading an HTML file."""
        result = self.loader.load(self.temp_file.name)
        
        # Check document type
        self.assertEqual(result['document_type'], 'html')
        
        # Check metadata
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['file_name'], os.path.basename(self.temp_file.name))
        self.assertEqual(result['metadata']['extension'], 'html')
        self.assertEqual(result['metadata']['title'], 'Test Document')
        self.assertEqual(result['metadata']['author'], 'Test Author')
        self.assertEqual(result['metadata']['description'], 'A test HTML document')
        
        # Check content
        self.assertIn('content', result)
        
        # Find headings
        headings = [item for item in result['content'] if item.get('type') == 'heading']
        self.assertEqual(len(headings), 2)
        self.assertEqual(headings[0]['level'], 1)
        self.assertEqual(headings[0]['text'], 'Main Heading')
        self.assertEqual(headings[1]['level'], 2)
        self.assertEqual(headings[1]['text'], 'Subheading')
        
        # Find paragraphs
        paragraphs = [item for item in result['content'] if item.get('type') == 'paragraph']
        self.assertEqual(len(paragraphs), 2)
        
        # Find lists
        lists = [item for item in result['content'] if item.get('type') == 'list']
        self.assertEqual(len(lists), 1)
        self.assertEqual(len(lists[0]['items']), 3)
        
        # Check links if extract_links is True
        links = []
        for item in result['content']:
            if item.get('type') == 'paragraph' and 'links' in item:
                links.extend(item['links'])
        
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['url'], 'https://example.com')
        self.assertEqual(links[0]['text'], 'a link')
    
    def test_load_html_without_links(self):
        """Test loading an HTML file without extracting links."""
        loader = HTMLLoader(extract_links=False)
        result = loader.load(self.temp_file.name)
        
        # Check that no links were extracted
        for item in result['content']:
            self.assertNotIn('links', item)

class TestTextNormalizer(unittest.TestCase):
    """Tests for the TextNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = TextNormalizer()
    
    def test_normalize_whitespace(self):
        """Test normalizing whitespace."""
        text = "This  has   extra    whitespace\nand\nnewlines."
        normalized = self.normalizer.normalize(text)
        
        self.assertEqual(normalized, "This has extra whitespace and newlines.")
    
    def test_normalize_lowercase(self):
        """Test converting to lowercase."""
        normalizer = TextNormalizer(lowercase=True)
        text = "This Has MIXED Case."
        normalized = normalizer.normalize(text)
        
        self.assertEqual(normalized, "this has mixed case.")
    
    def test_normalize_punctuation(self):
        """Test removing punctuation."""
        normalizer = TextNormalizer(remove_punctuation=True)
        text = "Hello, world! This is a test."
        normalized = normalizer.normalize(text)
        
        self.assertEqual(normalized, "Hello world This is a test")
    
    def test_normalize_document(self):
        """Test normalizing a document."""
        document = {
            'content': [
                {'type': 'paragraph', 'text': 'This  has   extra    whitespace.'},
                {'type': 'paragraph', 'text': 'Another\nparagraph.'}
            ],
            'metadata': {'title': 'Test Document'},
            'document_type': 'test'
        }
        
        normalized_doc = self.normalizer.normalize_document(document)
        
        self.assertEqual(normalized_doc['content'][0]['text'], 'This has extra whitespace.')
        self.assertEqual(normalized_doc['content'][1]['text'], 'Another paragraph.')

class TestDocumentProcessor(unittest.TestCase):
    """Tests for the DocumentProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create loaders
        self.csv_loader = CSVLoader()
        self.html_loader = HTMLLoader()
        
        # Create normalizer
        self.normalizer = TextNormalizer()
        
        # Create document processor
        self.processor = DocumentProcessor(
            loaders={
                'csv': self.csv_loader,
                'html': self.html_loader,
                'htm': self.html_loader
            },
            normalizer=self.normalizer
        )
        
        # Create temporary files for testing
        self.csv_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.csv_file.write(b"Name,Age,City\nJohn,30,New York\nJane,25,San Francisco")
        self.csv_file.close()
        
        self.html_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph.</p>
</body>
</html>"""
        self.html_file.write(html_content.encode('utf-8'))
        self.html_file.close()
    
    def tearDown(self):
        """Tear down test fixtures."""
        os.unlink(self.csv_file.name)
        os.unlink(self.html_file.name)
    
    def test_process_csv_file(self):
        """Test processing a CSV file."""
        result = self.processor.process(self.csv_file.name)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['document']['document_type'], 'csv')
        self.assertEqual(len(result['document']['content']), 2)  # Two rows
    
    def test_process_html_file(self):
        """Test processing an HTML file."""
        result = self.processor.process(self.html_file.name)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['document']['document_type'], 'html')
        self.assertEqual(result['document']['metadata']['title'], 'Test Document')
    
    def test_process_unsupported_file(self):
        """Test processing an unsupported file type."""
        # Create a temporary file with unsupported extension
        temp_file = tempfile.NamedTemporaryFile(suffix='.xyz', delete=False)
        temp_file.write(b"Some content")
        temp_file.close()
        
        try:
            result = self.processor.process(temp_file.name)
            
            self.assertFalse(result['success'])
            self.assertGreaterEqual(len(result['errors']), 1)
        finally:
            os.unlink(temp_file.name)
    
    def test_process_nonexistent_file(self):
        """Test processing a file that doesn't exist."""
        result = self.processor.process('nonexistent_file.csv')
        
        self.assertFalse(result['success'])
        self.assertGreaterEqual(len(result['errors']), 1)
    
    def test_fallback_processing(self):
        """Test fallback processing for problematic files."""
        # Create a corrupted CSV file
        corrupted_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        corrupted_file.write(b"Name,Age,City\nJohn,30,\"New York\nJane,25,San Francisco")  # Missing closing quote
        corrupted_file.close()
        
        try:
            result = self.processor.process(corrupted_file.name)
            
            # Should fall back to basic processing
            self.assertFalse(result['success'])
            self.assertIn('document', result)
            self.assertIn('recovery_method', result['document']['metadata'])
        finally:
            os.unlink(corrupted_file.name)

if __name__ == '__main__':
    unittest.main()
