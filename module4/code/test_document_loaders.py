"""
Tests for document loaders.

This module contains tests for the document loaders implemented in document_loaders.py.
"""

import os
import unittest
import tempfile
from document_loaders import TextLoader, MarkdownLoader, DocumentLoadingError, get_loader_for_file

class TestTextLoader(unittest.TestCase):
    """Tests for the TextLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = TextLoader()
        
        # Create a temporary text file for testing
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        self.temp_file.write(b"This is a test document.\n\nIt has multiple paragraphs.\n\nThird paragraph.")
        self.temp_file.close()
    
    def tearDown(self):
        """Tear down test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_load_text_file(self):
        """Test loading a text file."""
        result = self.loader.load(self.temp_file.name)
        
        # Check document type
        self.assertEqual(result['document_type'], 'text')
        
        # Check metadata
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['file_name'], os.path.basename(self.temp_file.name))
        self.assertEqual(result['metadata']['extension'], 'txt')
        
        # Check content
        self.assertIn('content', result)
        self.assertEqual(len(result['content']), 3)  # Three paragraphs
        self.assertEqual(result['content'][0]['type'], 'paragraph')
        self.assertEqual(result['content'][0]['text'], 'This is a test document.')
        self.assertEqual(result['content'][1]['text'], 'It has multiple paragraphs.')
        self.assertEqual(result['content'][2]['text'], 'Third paragraph.')
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with self.assertRaises(DocumentLoadingError):
            self.loader.load('nonexistent_file.txt')

class TestMarkdownLoader(unittest.TestCase):
    """Tests for the MarkdownLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = MarkdownLoader()
        
        # Create a temporary markdown file for testing
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.md', delete=False)
        markdown_content = """---
title: Test Document
author: Test Author
date: 2023-01-01
---

# Heading 1

This is a paragraph.

## Heading 2

Another paragraph with **bold** and *italic* text.

```python
def hello_world():
    print("Hello, World!")
```

### Heading 3

Final paragraph.
"""
        self.temp_file.write(markdown_content.encode('utf-8'))
        self.temp_file.close()
    
    def tearDown(self):
        """Tear down test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_load_markdown_file(self):
        """Test loading a markdown file."""
        result = self.loader.load(self.temp_file.name)
        
        # Check document type
        self.assertEqual(result['document_type'], 'markdown')
        
        # Check metadata
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['file_name'], os.path.basename(self.temp_file.name))
        self.assertEqual(result['metadata']['extension'], 'md')
        self.assertEqual(result['metadata']['title'], 'Test Document')
        self.assertEqual(result['metadata']['author'], 'Test Author')
        self.assertEqual(result['metadata']['date'], '2023-01-01')
        
        # Check content
        self.assertIn('content', result)
        
        # Find headings
        headings = [item for item in result['content'] if item.get('type') == 'heading']
        self.assertEqual(len(headings), 3)
        self.assertEqual(headings[0]['level'], 1)
        self.assertEqual(headings[0]['text'], 'Heading 1')
        self.assertEqual(headings[1]['level'], 2)
        self.assertEqual(headings[1]['text'], 'Heading 2')
        self.assertEqual(headings[2]['level'], 3)
        self.assertEqual(headings[2]['text'], 'Heading 3')
        
        # Find paragraphs
        paragraphs = [item for item in result['content'] if item.get('type') == 'paragraph']
        self.assertGreaterEqual(len(paragraphs), 3)
        
        # Find code blocks
        code_blocks = [item for item in result['content'] if item.get('type') == 'code_block']
        self.assertEqual(len(code_blocks), 1)
        self.assertEqual(code_blocks[0]['language'], 'python')
        self.assertIn('def hello_world():', code_blocks[0]['text'])

class TestGetLoaderForFile(unittest.TestCase):
    """Tests for the get_loader_for_file function."""
    
    def test_get_loader_for_text_file(self):
        """Test getting a loader for a text file."""
        loader = get_loader_for_file('test.txt')
        self.assertIsInstance(loader, TextLoader)
    
    def test_get_loader_for_markdown_file(self):
        """Test getting a loader for a markdown file."""
        loader = get_loader_for_file('test.md')
        self.assertIsInstance(loader, MarkdownLoader)
        
        loader = get_loader_for_file('test.markdown')
        self.assertIsInstance(loader, MarkdownLoader)
    
    def test_get_loader_for_unsupported_file(self):
        """Test getting a loader for an unsupported file type."""
        with self.assertRaises(ValueError):
            get_loader_for_file('test.unsupported')

if __name__ == '__main__':
    unittest.main()
