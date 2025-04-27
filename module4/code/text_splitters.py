"""
Text splitting strategies for document chunking.

This module provides various text splitting strategies for chunking documents
into smaller pieces for embedding and retrieval. Strategies include:
- Character-based splitting
- Token-based splitting
- Semantic splitting
- Recursive splitting
"""

import re
from typing import List, Dict, Any, Optional, Callable, Union

class TextSplitter:
    """Base class for text splitters."""
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        raise NotImplementedError("Subclasses must implement split_text()")
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            A list of document chunk dictionaries
        """
        raise NotImplementedError("Subclasses must implement split_documents()")

class CharacterTextSplitter(TextSplitter):
    """Split text based on character count."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize the character text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: String to use as a separator between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on character count.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # Split the text on separator
        splits = text.split(self.separator)
        
        # Remove empty splits
        splits = [s for s in splits if s.strip()]
        
        # Initialize chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Process each split
        for split in splits:
            split_length = len(split)
            
            # If adding this split would exceed chunk size, finalize the current chunk
            if current_length + split_length + len(self.separator) > self.chunk_size and current_chunk:
                chunks.append(self.separator.join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s) for s in current_chunk) + len(self.separator) * (len(current_chunk) - 1)
            
            # Add the current split to the chunk
            current_chunk.append(split)
            current_length += split_length + len(self.separator)
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            A list of document chunk dictionaries
        """
        chunked_documents = []
        
        for doc in documents:
            # Extract content based on document structure
            if 'content' in doc and isinstance(doc['content'], list):
                # Handle structured content (list of paragraphs, etc.)
                text_content = []
                for item in doc['content']:
                    if isinstance(item, dict) and 'text' in item:
                        text_content.append(item['text'])
                    elif isinstance(item, str):
                        text_content.append(item)
                
                full_text = self.separator.join(text_content)
            elif 'content' in doc and isinstance(doc['content'], str):
                # Handle plain text content
                full_text = doc['content']
            else:
                # Skip documents without proper content
                continue
            
            # Split the text
            chunks = self.split_text(full_text)
            
            # Create new document for each chunk
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    'content': chunk,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'chunk_index': i,
                        'chunk_count': len(chunks)
                    },
                    'document_type': doc.get('document_type', 'unknown')
                }
                chunked_documents.append(chunked_doc)
        
        return chunked_documents

class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Split text recursively based on a list of separators.
    
    This splitter tries to split on larger semantic units first (e.g., paragraphs),
    then falls back to smaller units if needed.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    ):
        """
        Initialize the recursive character text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use, in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively based on separators.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # If text is short enough, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # If we've reached the empty separator, split by character
                return [
                    text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
                ]
            
            if separator in text:
                splits = text.split(separator)
                
                # Process each split recursively
                final_chunks = []
                current_chunk = []
                current_length = 0
                
                for split in splits:
                    split_length = len(split)
                    
                    # If adding this split would exceed chunk size, process the current chunk
                    if current_length + split_length + len(separator) > self.chunk_size and current_chunk:
                        # Join the current chunk
                        current_text = separator.join(current_chunk)
                        
                        # Recursively split the current chunk with the next separator
                        next_separator_index = self.separators.index(separator) + 1
                        if next_separator_index < len(self.separators):
                            sub_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                                separators=self.separators[next_separator_index:]
                            )
                            final_chunks.extend(sub_splitter.split_text(current_text))
                        else:
                            final_chunks.append(current_text)
                        
                        # Start a new chunk with overlap
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:]
                        current_length = sum(len(s) for s in current_chunk) + len(separator) * (len(current_chunk) - 1)
                    
                    # Add the current split to the chunk
                    if split:  # Skip empty splits
                        current_chunk.append(split)
                        current_length += split_length + len(separator)
                
                # Process the final chunk
                if current_chunk:
                    current_text = separator.join(current_chunk)
                    next_separator_index = self.separators.index(separator) + 1
                    if next_separator_index < len(self.separators):
                        sub_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap,
                            separators=self.separators[next_separator_index:]
                        )
                        final_chunks.extend(sub_splitter.split_text(current_text))
                    else:
                        final_chunks.append(current_text)
                
                return final_chunks
        
        # If no separator was found, return the text as a single chunk
        return [text]
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            A list of document chunk dictionaries
        """
        chunked_documents = []
        
        for doc in documents:
            # Extract content based on document structure
            if 'content' in doc and isinstance(doc['content'], list):
                # Handle structured content (list of paragraphs, etc.)
                text_content = []
                for item in doc['content']:
                    if isinstance(item, dict) and 'text' in item:
                        text_content.append(item['text'])
                    elif isinstance(item, str):
                        text_content.append(item)
                
                full_text = "\n\n".join(text_content)
            elif 'content' in doc and isinstance(doc['content'], str):
                # Handle plain text content
                full_text = doc['content']
            else:
                # Skip documents without proper content
                continue
            
            # Split the text
            chunks = self.split_text(full_text)
            
            # Create new document for each chunk
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    'content': chunk,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'chunk_index': i,
                        'chunk_count': len(chunks)
                    },
                    'document_type': doc.get('document_type', 'unknown')
                }
                chunked_documents.append(chunked_doc)
        
        return chunked_documents

class SemanticTextSplitter(TextSplitter):
    """
    Split text based on semantic units like paragraphs, sentences, etc.
    
    This splitter tries to preserve semantic meaning by respecting natural
    language boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        paragraph_separator: str = "\n\n",
        sentence_separator: str = ". "
    ):
        """
        Initialize the semantic text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            paragraph_separator: String that separates paragraphs
            sentence_separator: String that separates sentences
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        self.sentence_separator = sentence_separator
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text based on semantic units.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # If text is short enough, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split into paragraphs
        paragraphs = text.split(self.paragraph_separator)
        paragraphs = [p for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If the paragraph fits in the current chunk, add it
            if current_length + paragraph_length + len(self.paragraph_separator) <= self.chunk_size:
                current_chunk.append(paragraph)
                current_length += paragraph_length + len(self.paragraph_separator)
            else:
                # If the paragraph is too large, split it into sentences
                if paragraph_length > self.chunk_size:
                    sentences = paragraph.split(self.sentence_separator)
                    sentences = [s + self.sentence_separator for s in sentences[:-1]] + [sentences[-1]]
                    
                    for sentence in sentences:
                        sentence_length = len(sentence)
                        
                        # If the sentence fits in the current chunk, add it
                        if current_length + sentence_length <= self.chunk_size:
                            current_chunk.append(sentence)
                            current_length += sentence_length
                        else:
                            # Finalize the current chunk
                            if current_chunk:
                                chunks.append(self.paragraph_separator.join(current_chunk))
                            
                            # If the sentence is too large, split it by character
                            if sentence_length > self.chunk_size:
                                # Split the sentence into fixed-size chunks
                                for i in range(0, sentence_length, self.chunk_size - self.chunk_overlap):
                                    chunks.append(sentence[i:i + self.chunk_size])
                                
                                # Reset the current chunk
                                current_chunk = []
                                current_length = 0
                            else:
                                # Start a new chunk with this sentence
                                current_chunk = [sentence]
                                current_length = sentence_length
                else:
                    # Finalize the current chunk
                    if current_chunk:
                        chunks.append(self.paragraph_separator.join(current_chunk))
                    
                    # Start a new chunk with this paragraph
                    current_chunk = [paragraph]
                    current_length = paragraph_length
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(self.paragraph_separator.join(current_chunk))
        
        return chunks
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            A list of document chunk dictionaries
        """
        chunked_documents = []
        
        for doc in documents:
            # Extract content based on document structure
            if 'content' in doc and isinstance(doc['content'], list):
                # Handle structured content (list of paragraphs, etc.)
                text_content = []
                for item in doc['content']:
                    if isinstance(item, dict) and 'text' in item:
                        text_content.append(item['text'])
                    elif isinstance(item, str):
                        text_content.append(item)
                
                full_text = self.paragraph_separator.join(text_content)
            elif 'content' in doc and isinstance(doc['content'], str):
                # Handle plain text content
                full_text = doc['content']
            else:
                # Skip documents without proper content
                continue
            
            # Split the text
            chunks = self.split_text(full_text)
            
            # Create new document for each chunk
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    'content': chunk,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'chunk_index': i,
                        'chunk_count': len(chunks)
                    },
                    'document_type': doc.get('document_type', 'unknown')
                }
                chunked_documents.append(chunked_doc)
        
        return chunked_documents

# Note: A TokenTextSplitter would require a tokenizer library
# This is a placeholder for the implementation

"""
class TokenTextSplitter(TextSplitter):
    # Requires: tiktoken or another tokenizer
    # Implementation would split text based on token count rather than character count
    pass
"""
