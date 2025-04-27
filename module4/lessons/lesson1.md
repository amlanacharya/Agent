# 📄 Module 4: Document Processing & RAG Foundations - Lesson 1: Document Processing Fundamentals 📚

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 📄 Understand the document processing pipeline architecture
- 🔍 Learn how to implement document loaders for various file formats
- 🧹 Master text extraction and normalization techniques
- 🏗️ Implement document structure preservation mechanisms
- 🛡️ Develop error handling for corrupted or unsupported documents

---

## 📚 Introduction to Document Processing

<img src="https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif" width="50%" height="50%"/>

Document processing is the foundation of any Retrieval-Augmented Generation (RAG) system. Before we can retrieve and generate content based on documents, we need to extract, normalize, and structure the information they contain. This process transforms raw documents into a format that can be effectively chunked, embedded, and retrieved.

> 💡 **Key Insight**: The quality of your document processing directly impacts the effectiveness of your entire RAG system. Garbage in, garbage out applies strongly here - if your document processing is poor, even the best retrieval and generation components won't save your system.

---

## 🏗️ Document Processing Pipeline Architecture

A comprehensive document processing pipeline typically consists of the following stages:

### 1. Document Loading

The first step is to load documents from various sources and formats into a standardized representation. This involves:

- **Format Detection**: Identifying the document format (PDF, DOCX, TXT, etc.)
- **Content Extraction**: Pulling raw text and metadata from the document
- **Initial Structuring**: Creating a basic document object with content and metadata

### 2. Text Normalization

Once the raw content is extracted, it needs to be normalized to ensure consistency:

- **Character Encoding**: Standardizing to UTF-8 or another consistent encoding
- **Whitespace Handling**: Normalizing spaces, tabs, and line breaks
- **Special Character Processing**: Handling non-standard characters and symbols
- **Case Normalization**: Converting to lowercase for case-insensitive operations (when appropriate)

### 3. Structure Preservation

Many documents have important structural elements that should be preserved:

- **Headings and Sections**: Identifying document hierarchy
- **Lists and Tables**: Preserving tabular and list data
- **Formatting Cues**: Retaining bold, italic, or other emphasis markers when relevant
- **Page Boundaries**: Tracking page numbers and boundaries for reference

### 4. Metadata Extraction

Documents often contain valuable metadata that can enhance retrieval:

- **Document Properties**: Author, creation date, title, etc.
- **Content-Based Metadata**: Automatically extracted topics, entities, or keywords
- **Custom Metadata**: User-defined tags or categories
- **Structural Metadata**: Table of contents, section headings, etc.

### 5. Error Handling and Validation

Robust document processing requires comprehensive error handling:

- **Corrupted File Detection**: Identifying and handling damaged documents
- **Unsupported Format Handling**: Graceful fallbacks for unsupported formats
- **Content Validation**: Ensuring extracted content meets quality thresholds
- **Recovery Mechanisms**: Partial extraction when complete processing fails

---

## 📄 Implementing Document Loaders

Document loaders are specialized components that handle specific file formats. Let's look at how to implement loaders for common formats:

### PDF Loader

PDF documents are complex and can contain text, images, tables, and other elements. A robust PDF loader needs to handle these complexities:

```python
import PyPDF2
from typing import Dict, List, Optional, Any

class PDFLoader:
    """Loader for PDF documents."""
    
    def __init__(self, extract_images: bool = False, extract_tables: bool = False):
        """
        Initialize the PDF loader.
        
        Args:
            extract_images: Whether to extract images from the PDF
            extract_tables: Whether to attempt table extraction
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a PDF document from a file path.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            A dictionary containing the document content and metadata
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'creation_date': reader.metadata.get('/CreationDate', ''),
                    'page_count': len(reader.pages),
                    'file_path': file_path
                }
                
                # Extract text content
                content = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    content.append({
                        'page_number': i + 1,
                        'text': page_text,
                        'images': self._extract_images(page) if self.extract_images else [],
                        'tables': self._extract_tables(page) if self.extract_tables else []
                    })
                
                return {
                    'metadata': metadata,
                    'content': content,
                    'document_type': 'pdf'
                }
                
        except Exception as e:
            raise DocumentLoadingError(f"Error loading PDF document: {str(e)}")
    
    def _extract_images(self, page) -> List[Dict[str, Any]]:
        """Extract images from a PDF page."""
        # Implementation would use libraries like pdf2image or PyMuPDF
        # This is a placeholder for the actual implementation
        return []
    
    def _extract_tables(self, page) -> List[Dict[str, Any]]:
        """Extract tables from a PDF page."""
        # Implementation would use libraries like tabula-py or camelot
        # This is a placeholder for the actual implementation
        return []

class DocumentLoadingError(Exception):
    """Exception raised when document loading fails."""
    pass
```

### DOCX Loader

Microsoft Word documents have their own structure and challenges:

```python
import docx
from typing import Dict, List, Any

class DOCXLoader:
    """Loader for Microsoft Word DOCX documents."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a DOCX document from a file path.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            A dictionary containing the document content and metadata
        """
        try:
            doc = docx.Document(file_path)
            
            # Extract metadata
            metadata = {
                'title': doc.core_properties.title or '',
                'author': doc.core_properties.author or '',
                'created': doc.core_properties.created or '',
                'paragraph_count': len(doc.paragraphs),
                'file_path': file_path
            }
            
            # Extract content with structure
            content = []
            
            # Process paragraphs
            for para in doc.paragraphs:
                if para.text.strip():  # Skip empty paragraphs
                    content.append({
                        'type': 'paragraph',
                        'text': para.text,
                        'style': para.style.name
                    })
            
            # Process tables
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    table_data.append(row_data)
                
                content.append({
                    'type': 'table',
                    'data': table_data
                })
            
            return {
                'metadata': metadata,
                'content': content,
                'document_type': 'docx'
            }
            
        except Exception as e:
            raise DocumentLoadingError(f"Error loading DOCX document: {str(e)}")
```

### Text Loader

Plain text files are simpler but still need proper handling:

```python
import os
from typing import Dict, Any

class TextLoader:
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
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as file:
                text = file.read()
            
            # Extract basic metadata from file properties
            file_stats = os.stat(file_path)
            
            metadata = {
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'file_size': file_stats.st_size,
                'created': file_stats.st_ctime,
                'modified': file_stats.st_mtime,
                'line_count': text.count('\n') + 1
            }
            
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
```

---

## 🧹 Text Extraction and Normalization

Once documents are loaded, the raw text often needs normalization to ensure consistency:

```python
import re
import unicodedata
from typing import Dict, List, Any

class TextNormalizer:
    """Normalizes text from various document sources."""
    
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
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
    
    def normalize(self, text: str) -> str:
        """
        Normalize text according to the configured options.
        
        Args:
            text: The input text to normalize
            
        Returns:
            Normalized text
        """
        # Unicode normalization (NFC form)
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def normalize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all text content in a document.
        
        Args:
            document: The document dictionary from a loader
            
        Returns:
            Document with normalized text
        """
        normalized_document = document.copy()
        
        # Normalize content based on its structure
        if 'content' in normalized_document:
            for i, item in enumerate(normalized_document['content']):
                if isinstance(item, dict):
                    if 'text' in item:
                        item['text'] = self.normalize(item['text'])
                    elif 'data' in item and isinstance(item['data'], list):
                        # Handle table data
                        for j, row in enumerate(item['data']):
                            if isinstance(row, list):
                                item['data'][j] = [self.normalize(cell) if isinstance(cell, str) else cell for cell in row]
                            elif isinstance(row, str):
                                item['data'][j] = self.normalize(row)
                elif isinstance(item, str):
                    normalized_document['content'][i] = self.normalize(item)
        
        return normalized_document
```

---

## 🏗️ Document Structure Preservation

Preserving document structure is crucial for maintaining context and improving retrieval relevance:

```python
from typing import Dict, List, Any, Optional

class StructurePreserver:
    """Preserves and enhances document structure."""
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document to enhance and preserve its structure.
        
        Args:
            document: The document dictionary from a loader
            
        Returns:
            Document with enhanced structure
        """
        document_type = document.get('document_type', '')
        
        if document_type == 'pdf':
            return self._process_pdf(document)
        elif document_type == 'docx':
            return self._process_docx(document)
        elif document_type == 'text':
            return self._process_text(document)
        else:
            # Default processing for unknown types
            return self._process_generic(document)
    
    def _process_pdf(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF document structure."""
        structured_doc = document.copy()
        
        # Extract sections based on font size and formatting
        sections = []
        current_section = None
        
        for page in structured_doc.get('content', []):
            page_text = page.get('text', '')
            
            # Simple heuristic: lines that are shorter and possibly in larger font
            # might be headings (this is a simplification - real implementation would
            # use font information from the PDF)
            lines = page_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Heuristic for heading detection (simplified)
                if len(line) < 100 and line.isupper() or line.endswith(':'):
                    # Likely a heading - start new section
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'heading': line,
                        'content': '',
                        'page': page.get('page_number')
                    }
                elif current_section:
                    # Add to current section
                    current_section['content'] += line + '\n'
                else:
                    # Text before any heading
                    current_section = {
                        'heading': 'Introduction',
                        'content': line + '\n',
                        'page': page.get('page_number')
                    }
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        structured_doc['sections'] = sections
        return structured_doc
    
    def _process_docx(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process DOCX document structure."""
        structured_doc = document.copy()
        
        # Extract sections based on paragraph styles
        sections = []
        current_section = None
        
        for item in structured_doc.get('content', []):
            if item.get('type') == 'paragraph':
                style = item.get('style', '')
                text = item.get('text', '')
                
                # Check if this is a heading paragraph
                if 'Heading' in style or style == 'Title':
                    # Start a new section
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'heading': text,
                        'content': '',
                        'level': int(style.replace('Heading', '')) if 'Heading' in style and style[-1].isdigit() else 0
                    }
                elif current_section:
                    # Add to current section
                    current_section['content'] += text + '\n'
                else:
                    # Text before any heading
                    current_section = {
                        'heading': 'Introduction',
                        'content': text + '\n',
                        'level': 0
                    }
            elif item.get('type') == 'table' and current_section:
                # Add table reference to the current section
                current_section['content'] += '[TABLE]\n'
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        structured_doc['sections'] = sections
        return structured_doc
    
    def _process_text(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process plain text document structure."""
        structured_doc = document.copy()
        
        # For plain text, try to identify sections based on blank lines
        # and potential heading patterns
        sections = []
        current_section = None
        
        for item in structured_doc.get('content', []):
            if item.get('type') == 'paragraph':
                text = item.get('text', '')
                
                # Simple heuristic for headings in plain text
                if text.isupper() or (len(text) < 100 and text.endswith(':')) or text.startswith('#'):
                    # Likely a heading - start new section
                    if current_section:
                        sections.append(current_section)
                    
                    # Clean up heading (remove # from markdown-style headings)
                    heading = text.lstrip('#').strip()
                    
                    current_section = {
                        'heading': heading,
                        'content': ''
                    }
                elif current_section:
                    # Add to current section
                    current_section['content'] += text + '\n'
                else:
                    # Text before any heading
                    current_section = {
                        'heading': 'Introduction',
                        'content': text + '\n'
                    }
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        structured_doc['sections'] = sections
        return structured_doc
    
    def _process_generic(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process generic document structure."""
        # Default processing for unknown document types
        return document
```

---

## 🛡️ Error Handling for Document Processing

Robust error handling is essential for a production-ready document processing system:

```python
from typing import Dict, List, Any, Optional, Union
import logging

class DocumentProcessor:
    """Main document processing pipeline with error handling."""
    
    def __init__(self, loaders: Dict[str, Any], normalizer: Any, structure_preserver: Any):
        """
        Initialize the document processor.
        
        Args:
            loaders: Dictionary of document loaders by file extension
            normalizer: Text normalizer instance
            structure_preserver: Structure preserver instance
        """
        self.loaders = loaders
        self.normalizer = normalizer
        self.structure_preserver = structure_preserver
        self.logger = logging.getLogger(__name__)
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document with comprehensive error handling.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed document dictionary or error information
        """
        result = {
            'file_path': file_path,
            'success': False,
            'errors': [],
            'document': None
        }
        
        try:
            # Determine file extension
            extension = file_path.split('.')[-1].lower()
            
            # Get appropriate loader
            loader = self.loaders.get(extension)
            if not loader:
                raise ValueError(f"No loader available for file extension: {extension}")
            
            # Load document
            document = loader.load(file_path)
            result['document_type'] = document.get('document_type', extension)
            
            # Normalize text
            try:
                document = self.normalizer.normalize_document(document)
            except Exception as e:
                self.logger.warning(f"Text normalization error: {str(e)}")
                result['errors'].append({
                    'stage': 'normalization',
                    'error': str(e)
                })
                # Continue with unnormalized document
            
            # Preserve structure
            try:
                document = self.structure_preserver.process_document(document)
            except Exception as e:
                self.logger.warning(f"Structure preservation error: {str(e)}")
                result['errors'].append({
                    'stage': 'structure_preservation',
                    'error': str(e)
                })
                # Continue with basic structure
            
            # Set success and document
            result['success'] = True
            result['document'] = document
            
        except Exception as e:
            self.logger.error(f"Document processing error: {str(e)}")
            result['errors'].append({
                'stage': 'loading',
                'error': str(e)
            })
            
            # Try fallback processing for partially corrupted documents
            try:
                result['document'] = self._fallback_processing(file_path)
                result['errors'].append({
                    'stage': 'recovery',
                    'message': 'Used fallback processing'
                })
            except Exception as fallback_error:
                self.logger.error(f"Fallback processing failed: {str(fallback_error)}")
                result['errors'].append({
                    'stage': 'recovery',
                    'error': str(fallback_error)
                })
        
        return result
    
    def _fallback_processing(self, file_path: str) -> Dict[str, Any]:
        """
        Attempt to extract at least some content from a problematic document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Basic document dictionary with whatever content could be extracted
        """
        # Try to read as plain text as a last resort
        try:
            with open(file_path, 'r', errors='ignore') as file:
                text = file.read()
            
            return {
                'metadata': {
                    'file_path': file_path,
                    'recovery_method': 'plain_text_fallback'
                },
                'content': [{'type': 'paragraph', 'text': text}],
                'document_type': 'recovered_text'
            }
        except:
            # If even that fails, return minimal information
            return {
                'metadata': {
                    'file_path': file_path,
                    'recovery_method': 'metadata_only'
                },
                'content': [],
                'document_type': 'unrecoverable'
            }
```

---

## 💪 Practice Exercises

1. **Implement a HTML Loader**: Create a document loader for HTML files that extracts text while preserving the document structure (headings, paragraphs, lists, etc.).

2. **Enhance the PDF Loader**: Extend the PDF loader to extract and process tables using a library like tabula-py or camelot.

3. **Create a CSV Loader**: Implement a loader for CSV files that converts the tabular data into a structured document format.

4. **Build a Markdown Loader**: Develop a loader for Markdown files that preserves the document structure including headings, lists, code blocks, and links.

5. **Implement a Document Converter**: Create a utility that can convert between different document formats (e.g., from PDF to structured text).

---

## 🔍 Key Takeaways

1. **Document processing** is the foundation of any RAG system, transforming raw documents into a format suitable for chunking, embedding, and retrieval.

2. **Document loaders** handle the specifics of different file formats, extracting text and metadata while preserving important structural elements.

3. **Text normalization** ensures consistency across documents, addressing issues like character encoding, whitespace, and special characters.

4. **Structure preservation** maintains the document's organization, improving context retention and retrieval relevance.

5. **Error handling** is crucial for robust document processing, allowing the system to gracefully handle corrupted files and unexpected formats.

---

## 📚 Resources

- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/en/latest/)
- [python-docx Documentation](https://python-docx.readthedocs.io/en/latest/)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Unstructured.io](https://unstructured.io/) - A library for preprocessing documents
- [Tabula-py](https://github.com/chezou/tabula-py) - Extract tables from PDFs

---

## 🚀 Next Steps

In the next lesson, we'll explore **Chunking Strategies for Optimal Retrieval**, focusing on how to divide processed documents into chunks that balance context preservation with retrieval efficiency. We'll cover size-based chunking, semantic chunking, recursive chunking, and token-aware approaches.
