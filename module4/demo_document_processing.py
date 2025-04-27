"""
Demo script for document processing pipeline.

This script demonstrates how to use the document processing components
to load, process, and chunk documents for a RAG system.
"""

import os
import argparse
import sys

# Add the module directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module4.code.document_loaders import TextLoader, MarkdownLoader, get_loader_for_file
from module4.code.text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, SemanticTextSplitter

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
