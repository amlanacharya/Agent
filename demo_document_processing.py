"""
Demo script for document processing pipeline.

This script demonstrates how to use the document processing components
to load, process, and chunk documents for a RAG system.
"""

import os
import argparse
import sys
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Document loader implementation
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
    else:
        raise ValueError(f"No loader available for file type: {extension}")

# Text splitter implementation
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
                splits = [s for s in splits if s.strip()]  # Remove empty splits

                # If the text can't be split on this separator, try the next one
                if len(splits) <= 1:
                    continue

                # Merge splits into chunks of appropriate size
                chunks = []
                current_chunk = []
                current_length = 0

                for split in splits:
                    split_length = len(split)

                    # If this split alone exceeds chunk size, we need to use the next separator
                    if split_length > self.chunk_size:
                        # First add the current accumulated chunk if it exists
                        if current_chunk:
                            chunks.append(separator.join(current_chunk))
                            current_chunk = []
                            current_length = 0

                        # Try to split this large split with the next separator
                        next_separator_index = self.separators.index(separator) + 1
                        if next_separator_index < len(self.separators):
                            # Use the next separator in the list
                            next_separator = self.separators[next_separator_index]
                            if next_separator in split:
                                # If the next separator exists in this split, recursively split it
                                sub_splits = split.split(next_separator)
                                sub_splits = [s for s in sub_splits if s.strip()]

                                # Merge sub_splits into chunks
                                sub_current_chunk = []
                                sub_current_length = 0

                                for sub_split in sub_splits:
                                    sub_split_length = len(sub_split)

                                    # If adding this sub_split would exceed chunk size, finalize the current sub-chunk
                                    if sub_current_length + sub_split_length + len(next_separator) > self.chunk_size and sub_current_chunk:
                                        chunks.append(next_separator.join(sub_current_chunk))

                                        # Start a new sub-chunk with overlap
                                        overlap_start = max(0, len(sub_current_chunk) - 1)  # Just overlap by 1 item for simplicity
                                        sub_current_chunk = sub_current_chunk[overlap_start:]
                                        sub_current_length = sum(len(s) for s in sub_current_chunk) + len(next_separator) * (len(sub_current_chunk) - 1)

                                    # Add the current sub_split to the sub-chunk
                                    sub_current_chunk.append(sub_split)
                                    sub_current_length += sub_split_length + len(next_separator)

                                # Add the final sub-chunk if it's not empty
                                if sub_current_chunk:
                                    chunks.append(next_separator.join(sub_current_chunk))
                            else:
                                # If the next separator doesn't exist, use character splitting as a last resort
                                for i in range(0, split_length, self.chunk_size - self.chunk_overlap):
                                    chunks.append(split[i:i + self.chunk_size])
                        else:
                            # If there are no more separators, use character splitting
                            for i in range(0, split_length, self.chunk_size - self.chunk_overlap):
                                chunks.append(split[i:i + self.chunk_size])
                    else:
                        # If adding this split would exceed chunk size, finalize the current chunk
                        if current_length + split_length + len(separator) > self.chunk_size and current_chunk:
                            chunks.append(separator.join(current_chunk))

                            # Start a new chunk with overlap (if possible)
                            if self.chunk_overlap > 0 and len(current_chunk) > 1:
                                # Overlap by at most 1 item for simplicity and to avoid excessive recursion
                                current_chunk = current_chunk[-1:]
                                current_length = len(current_chunk[0])
                            else:
                                current_chunk = []
                                current_length = 0

                        # Add the current split to the chunk
                        current_chunk.append(split)
                        current_length += split_length + len(separator)

                # Add the final chunk if it's not empty
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                return chunks

        # If no separator was found or worked, return the text as a single chunk
        # This should rarely happen since we have the empty separator as a fallback
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

def process_document(file_path, chunk_size=1000, chunk_overlap=200, splitter_type='recursive'):
    """
    Process a document through the pipeline.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        splitter_type: Type of text splitter to use ('character', 'recursive', or 'semantic')

    Returns:
        A list of document chunks
    """
    print(f"Processing document: {file_path}")

    # Get the appropriate loader
    try:
        loader = get_loader_for_file(file_path)
        print(f"Using loader: {loader.__class__.__name__}")
    except ValueError as e:
        print(f"Error: {str(e)}")
        return []

    # Load the document
    try:
        document = loader.load(file_path)
        print(f"Document loaded successfully")
        print(f"Document type: {document['document_type']}")
        print(f"Metadata: {document['metadata']}")
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return []

    # Create the appropriate text splitter
    if splitter_type == 'character':
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"Using CharacterTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    elif splitter_type == 'recursive':
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"Using RecursiveCharacterTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    elif splitter_type == 'semantic':
        splitter = SemanticTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"Using SemanticTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    else:
        print(f"Unknown splitter type: {splitter_type}")
        return []

    # Split the document
    try:
        chunks = splitter.split_documents([document])
        print(f"Document split into {len(chunks)} chunks")

        # Print sample chunks
        if chunks:
            print("\nSample chunks:")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"\nChunk {i+1}:")
                print(f"Metadata: {chunk['metadata']}")
                content = chunk['content']
                if isinstance(content, str):
                    # Show first 100 characters
                    print(f"Content: {content[:100]}...")
                else:
                    print(f"Content: {content}")

            if len(chunks) > 3:
                print(f"\n... and {len(chunks) - 3} more chunks")

        return chunks
    except Exception as e:
        print(f"Error splitting document: {str(e)}")
        return []

def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description='Document Processing Demo')
    parser.add_argument('file_path', help='Path to the document file')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Maximum size of each chunk')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Number of characters to overlap between chunks')
    parser.add_argument('--splitter', choices=['character', 'recursive', 'semantic'], default='recursive',
                        help='Type of text splitter to use')

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return

    process_document(args.file_path, args.chunk_size, args.chunk_overlap, args.splitter)

if __name__ == "__main__":
    main()
