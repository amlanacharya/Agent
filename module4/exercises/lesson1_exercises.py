"""
Exercise 1: Document Processing Fundamentals

This exercise focuses on implementing document loaders for different file formats
and creating a document processing pipeline.
"""

import os
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class DocumentLoadingError(Exception):
    """Exception raised when document loading fails."""
    pass

class BaseDocumentLoader:
    """Base class for all document loaders."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a document from a file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            A dictionary containing the document content and metadata
            
        Raises:
            DocumentLoadingError: If the document cannot be loaded
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract basic metadata from file properties.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of file metadata
        """
        try:
            file_stats = os.stat(file_path)
            
            return {
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'file_size': file_stats.st_size,
                'created_time': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'extension': os.path.splitext(file_path)[1].lower()[1:]
            }
        except Exception as e:
            return {
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'error': str(e)
            }

# Exercise 1: Implement a CSV Loader
class CSVLoader(BaseDocumentLoader):
    """
    Loader for CSV files.
    
    TODO: Implement this class to load CSV files and convert them to a structured document format.
    The loader should:
    1. Read the CSV file
    2. Extract headers and data
    3. Create a structured representation with metadata
    4. Handle different CSV formats (delimiters, quoting, etc.)
    """
    
    def __init__(self, delimiter: str = ',', has_header: bool = True, encoding: str = 'utf-8'):
        """
        Initialize the CSV loader.
        
        Args:
            delimiter: Character used to separate fields
            has_header: Whether the CSV file has a header row
            encoding: Character encoding to use when reading the file
        """
        # TODO: Implement this method
        pass
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a CSV document from a file path.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            A dictionary containing the document content and metadata
            
        Raises:
            DocumentLoadingError: If the document cannot be loaded
        """
        # TODO: Implement this method
        # 1. Read the CSV file
        # 2. Extract headers and data
        # 3. Create a structured representation
        # 4. Include metadata
        # 5. Handle errors appropriately
        pass

# Exercise 2: Implement an HTML Loader
class HTMLLoader(BaseDocumentLoader):
    """
    Loader for HTML files.
    
    TODO: Implement this class to load HTML files and extract structured content.
    The loader should:
    1. Parse the HTML document
    2. Extract text content while preserving structure (headings, paragraphs, lists, etc.)
    3. Extract metadata from meta tags
    4. Handle encoding issues
    """
    
    def __init__(self, extract_links: bool = True, include_images: bool = False, encoding: str = 'utf-8'):
        """
        Initialize the HTML loader.
        
        Args:
            extract_links: Whether to extract and include links
            include_images: Whether to include image references
            encoding: Character encoding to use when reading the file
        """
        # TODO: Implement this method
        pass
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load an HTML document from a file path.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            A dictionary containing the document content and metadata
            
        Raises:
            DocumentLoadingError: If the document cannot be loaded
        """
        # TODO: Implement this method
        # 1. Read and parse the HTML file
        # 2. Extract text content with structure
        # 3. Extract metadata from meta tags
        # 4. Include links if requested
        # 5. Handle errors appropriately
        pass

# Exercise 3: Implement a Text Normalizer
class TextNormalizer:
    """
    Normalizes text from various document sources.
    
    TODO: Implement this class to normalize text content.
    The normalizer should:
    1. Handle different character encodings
    2. Normalize whitespace
    3. Optionally convert to lowercase
    4. Optionally remove punctuation
    5. Handle special characters and symbols
    """
    
    def __init__(self, 
                 lowercase: bool = False,
                 remove_punctuation: bool = False,
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True):
        """
        Initialize the text normalizer.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_extra_whitespace: Whether to normalize whitespace
            normalize_unicode: Whether to normalize Unicode characters
        """
        # TODO: Implement this method
        pass
    
    def normalize(self, text: str) -> str:
        """
        Normalize text according to the configured options.
        
        Args:
            text: The input text to normalize
            
        Returns:
            Normalized text
        """
        # TODO: Implement this method
        # 1. Normalize Unicode if requested
        # 2. Remove extra whitespace if requested
        # 3. Remove punctuation if requested
        # 4. Convert to lowercase if requested
        pass
    
    def normalize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all text content in a document.
        
        Args:
            document: The document dictionary from a loader
            
        Returns:
            Document with normalized text
        """
        # TODO: Implement this method
        # 1. Create a copy of the document
        # 2. Normalize text content based on document structure
        # 3. Handle different content types (paragraphs, tables, etc.)
        pass

# Exercise 4: Implement a Document Processor
class DocumentProcessor:
    """
    Main document processing pipeline with error handling.
    
    TODO: Implement this class to process documents through a complete pipeline.
    The processor should:
    1. Select the appropriate loader based on file extension
    2. Load the document
    3. Normalize the text
    4. Handle errors at each stage
    5. Provide fallback processing for problematic documents
    """
    
    def __init__(self, loaders: Dict[str, BaseDocumentLoader], normalizer: TextNormalizer):
        """
        Initialize the document processor.
        
        Args:
            loaders: Dictionary of document loaders by file extension
            normalizer: Text normalizer instance
        """
        # TODO: Implement this method
        pass
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document with comprehensive error handling.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed document dictionary or error information
        """
        # TODO: Implement this method
        # 1. Determine file extension
        # 2. Get appropriate loader
        # 3. Load document
        # 4. Normalize text
        # 5. Handle errors at each stage
        # 6. Provide fallback processing for problematic documents
        pass
    
    def _fallback_processing(self, file_path: str) -> Dict[str, Any]:
        """
        Attempt to extract at least some content from a problematic document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Basic document dictionary with whatever content could be extracted
        """
        # TODO: Implement this method
        # 1. Try to read as plain text
        # 2. If that fails, return minimal information
        pass

# Example usage (uncomment when implementations are complete)
"""
if __name__ == "__main__":
    # Create loaders
    csv_loader = CSVLoader()
    html_loader = HTMLLoader()
    
    # Create normalizer
    normalizer = TextNormalizer()
    
    # Create document processor
    processor = DocumentProcessor(
        loaders={
            'csv': csv_loader,
            'html': html_loader,
            'htm': html_loader
        },
        normalizer=normalizer
    )
    
    # Process a CSV file
    csv_result = processor.process('example.csv')
    print(f"CSV Processing Result: {csv_result['success']}")
    if csv_result['success']:
        print(f"CSV Headers: {csv_result['document']['metadata'].get('headers')}")
        print(f"CSV Row Count: {csv_result['document']['metadata'].get('row_count')}")
    
    # Process an HTML file
    html_result = processor.process('example.html')
    print(f"HTML Processing Result: {html_result['success']}")
    if html_result['success']:
        print(f"HTML Title: {html_result['document']['metadata'].get('title')}")
        print(f"HTML Content Types: {[item['type'] for item in html_result['document']['content']]}")
"""
