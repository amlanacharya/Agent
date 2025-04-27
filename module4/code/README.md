# 🧩 Module 4: Code Examples

## 📚 Overview

This directory contains all the code examples and implementations for Module 4: Document Processing & RAG Foundations. These examples demonstrate the core concepts of document processing, chunking strategies, embedding generation, metadata extraction, and RAG system implementation.

## 🔍 File Descriptions

### Core Implementations
- **document_loaders.py**: Implementation of document loaders for various file formats (PDF, TXT, DOCX, etc.)
- **text_splitters.py**: Text splitting strategies for optimal chunking of documents
- **embedding_pipelines.py**: Embedding generation pipelines for different content types
- **metadata_extractors.py**: Metadata extraction systems for enhanced retrieval
- **rag_system.py**: Simple RAG system combining retrieval and generation
- **document_qa.py**: Complete document Q&A system implementation

### Test Scripts
- **test_document_loaders.py**: Tests for document loader implementations
- **test_text_splitters.py**: Tests for text splitting strategies
- **test_embedding_pipelines.py**: Tests for embedding generation pipelines
- **test_metadata_extractors.py**: Tests for metadata extraction systems
- **test_rag_system.py**: Tests for the RAG system implementation
- **test_document_qa.py**: Tests for the document Q&A system

## 🧠 Implementation Details

### Document Loaders
- The document loaders support various file formats including PDF, TXT, DOCX, HTML, and Markdown
- Each loader handles format-specific challenges like preserving structure, extracting tables, and managing images
- The implementation includes error handling for corrupted files and unsupported formats
- Loaders return standardized document objects with content and metadata

### Text Splitters
- Multiple chunking strategies are implemented for different document types and use cases
- Size-based splitters divide text based on character or token count
- Semantic splitters preserve meaning by respecting paragraph and section boundaries
- Recursive splitters handle nested document structures
- Token-aware splitters ensure chunks don't exceed model token limits

### Embedding Pipelines
- The embedding pipelines support multiple embedding models (Sentence Transformers, OpenAI, etc.)
- Implementation includes preprocessing steps like normalization and cleaning
- Batching mechanisms optimize throughput for large document collections
- Caching systems reduce redundant embedding generation
- Model selection helpers guide users to appropriate embeddings for their content

### Metadata Extractors
- Automatic metadata extraction from document properties (author, date, title, etc.)
- Content-based metadata generation (topics, entities, keywords)
- Custom metadata tagging systems for user-defined attributes
- Structured metadata schemas for consistent organization
- Metadata filtering mechanisms for retrieval refinement

### RAG System
- Vector database integration with FAISS or ChromaDB
- Retrieval mechanisms based on semantic similarity
- Context augmentation for query enhancement
- Generation with retrieved context using LLMs
- Hybrid retrieval combining keyword and semantic search

### Document Q&A System
- Complete implementation combining all components
- Question processing and reformulation
- Multi-document retrieval and synthesis
- Answer generation with source attribution
- Confidence scoring for responses
- Handling of metadata-specific queries

## 🔄 Integration with Previous Modules

The implementations in this module build upon concepts from previous modules:
- **Module 1**: Uses the sense-think-act loop for processing queries and generating responses
- **Module 2**: Incorporates vector databases and retrieval patterns from memory systems
- **Module 3**: Applies data validation with Pydantic for structured document representation

## 🧪 Example Usage

Here's a simple example of how to use the document Q&A system:

```python
from document_loaders import PDFLoader, TextLoader
from text_splitters import RecursiveCharacterTextSplitter
from embedding_pipelines import SentenceTransformerEmbeddings
from rag_system import SimpleRAGSystem
from document_qa import DocumentQASystem

# Load documents
pdf_loader = PDFLoader("path/to/document.pdf")
text_loader = TextLoader("path/to/document.txt")
documents = pdf_loader.load() + text_loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create embeddings
embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(chunks)

# Initialize RAG system
rag_system = SimpleRAGSystem(
    documents=chunks,
    embeddings=embeddings,
    vector_store_type="faiss"
)

# Create Q&A system
qa_system = DocumentQASystem(rag_system=rag_system)

# Ask questions
response = qa_system.answer_question(
    question="What are the main points in the document?",
    k=3  # Number of chunks to retrieve
)

print(response.answer)
print("Sources:", response.sources)
```

## 🛠️ Advanced Features

The code examples in this module also demonstrate several advanced features:

- **Streaming Processing**: Handle large documents with limited memory through streaming
- **Parallel Processing**: Speed up document processing with parallel execution
- **Custom Chunking Rules**: Define domain-specific rules for optimal chunking
- **Embedding Caching**: Reduce computation by caching embeddings
- **Hybrid Search**: Combine keyword and semantic search for better results
- **Metadata Filtering**: Refine retrieval based on document metadata
- **Multi-Modal Support**: Handle text, tables, and images in documents
- **Confidence Scoring**: Assess the reliability of generated answers

## 🔍 Implementation Notes

- All implementations follow clean code principles with proper documentation
- Error handling is robust with appropriate exception types
- Logging is comprehensive for debugging and monitoring
- Performance considerations are addressed with efficient algorithms
- Testing covers both unit tests and integration tests
