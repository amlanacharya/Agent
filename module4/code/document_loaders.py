"""
Document loaders for various file formats.

This module provides document loaders for different file formats including:
- PDF
- DOCX (Microsoft Word)
- TXT (Plain text)
- HTML
- Markdown

Each loader extracts text content and metadata from the respective file format
and returns a standardized document representation.
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

class TextLoader(BaseDocumentLoader):
    """Loader for plain text documents."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the text loader.
        
        Args:
            encoding: Character encoding to use when reading the file
        """
        self.encoding = encoding
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a text document from a file path.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            A dictionary containing the document content and metadata
            
        Raises:
            DocumentLoadingError: If the document cannot be loaded
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as file:
                text = file.read()
            
            # Get basic file metadata
            metadata = self._get_file_metadata(file_path)
            metadata.update({
                'line_count': text.count('\n') + 1,
                'word_count': len(text.split())
            })
            
            # Process content with basic structure preservation
            lines = text.split('\n')
            paragraphs = []
            current_paragraph = []
            
            for line in lines:
                if line.strip():
                    current_paragraph.append(line)
                elif current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            # Add the last paragraph if it exists
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            content = [{'type': 'paragraph', 'text': p} for p in paragraphs]
            
            return {
                'metadata': metadata,
                'content': content,
                'document_type': 'text'
            }
            
        except Exception as e:
            raise DocumentLoadingError(f"Error loading text document: {str(e)}")

class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown documents."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the Markdown loader.
        
        Args:
            encoding: Character encoding to use when reading the file
        """
        self.encoding = encoding
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a Markdown document from a file path.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            A dictionary containing the document content and metadata
            
        Raises:
            DocumentLoadingError: If the document cannot be loaded
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as file:
                text = file.read()
            
            # Get basic file metadata
            metadata = self._get_file_metadata(file_path)
            
            # Extract front matter if present (YAML between --- markers)
            front_matter = {}
            content_text = text
            front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
            if front_matter_match:
                front_matter_text = front_matter_match.group(1)
                content_text = text[front_matter_match.end():]
                
                # Simple parsing of front matter
                for line in front_matter_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        front_matter[key.strip()] = value.strip()
            
            metadata.update(front_matter)
            
            # Parse Markdown structure
            content = []
            current_section = None
            
            for line in content_text.split('\n'):
                # Check for headings
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if heading_match:
                    level = len(heading_match.group(1))
                    text = heading_match.group(2)
                    
                    content.append({
                        'type': 'heading',
                        'level': level,
                        'text': text
                    })
                    continue
                
                # Check for code blocks
                if line.startswith('```'):
                    language = line[3:].strip()
                    code_block = []
                    line_idx = content_text.split('\n').index(line) + 1
                    
                    while line_idx < len(content_text.split('\n')):
                        next_line = content_text.split('\n')[line_idx]
                        if next_line.startswith('```'):
                            break
                        code_block.append(next_line)
                        line_idx += 1
                    
                    content.append({
                        'type': 'code_block',
                        'language': language,
                        'text': '\n'.join(code_block)
                    })
                    continue
                
                # Regular paragraph text
                if line.strip():
                    content.append({
                        'type': 'paragraph',
                        'text': line
                    })
            
            return {
                'metadata': metadata,
                'content': content,
                'document_type': 'markdown'
            }
            
        except Exception as e:
            raise DocumentLoadingError(f"Error loading Markdown document: {str(e)}")

# Note: The following loaders would require additional dependencies
# They are included as placeholders with implementation notes

"""
class PDFLoader(BaseDocumentLoader):
    # Requires: PyPDF2 or PyMuPDF (fitz)
    # Implementation would extract text, metadata, and optionally images/tables
    pass

class DOCXLoader(BaseDocumentLoader):
    # Requires: python-docx
    # Implementation would extract text, styles, tables, and document properties
    pass

class HTMLLoader(BaseDocumentLoader):
    # Requires: BeautifulSoup4
    # Implementation would extract text while preserving document structure
    pass
"""

def get_loader_for_file(file_path: str) -> BaseDocumentLoader:
    """
    Get the appropriate loader for a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        An instance of the appropriate document loader
        
    Raises:
        ValueError: If no loader is available for the file type
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.txt':
        return TextLoader()
    elif extension in ['.md', '.markdown']:
        return MarkdownLoader()
    # Add more loaders as they are implemented
    # elif extension == '.pdf':
    #     return PDFLoader()
    # elif extension in ['.docx', '.doc']:
    #     return DOCXLoader()
    # elif extension in ['.html', '.htm']:
    #     return HTMLLoader()
    else:
        raise ValueError(f"No loader available for file type: {extension}")
