# 📚 Module 4: Document Processing & RAG Foundations

## 📚 Overview

Welcome to Module 4 of the Accelerated Agentic AI Mastery course! This module explores the foundations of document processing and Retrieval-Augmented Generation (RAG), focusing on how to effectively process, chunk, embed, and retrieve information from various document types to enhance your AI agents with knowledge from external sources.

> 💡 **Note**: Module 4 builds on the foundations established in previous modules, applying agent fundamentals, memory systems, and data validation to create robust document processing pipelines and RAG systems. We'll explore how to transform raw documents into structured knowledge that agents can effectively utilize.

## 🆕 New Features

We've recently added several exciting new features to this module:

1. **Streamlit-based Document Q&A System**: A complete, interactive web application that demonstrates all the concepts covered in this module. The app allows you to:
   - Upload and process documents in various formats (PDF, TXT, DOCX, CSV)
   - Ask questions and get answers with source attribution
   - Explore document metadata and content
   - Visualize document statistics and analysis
   - Export conversation history and document collections

2. **Enhanced RAG Components**: Improved implementation of RAG system components with:
   - Robust error handling for document processing
   - Fallback mechanisms for missing dependencies
   - Hybrid search combining semantic and keyword-based retrieval
   - Confidence scoring and uncertainty handling
   - Metadata extraction and querying

3. **Progressive Journey Diagrams**: New diagrams that illustrate the step-by-step development of:
   - Document processing pipelines
   - Chunking strategies
   - Embedding generation
   - RAG system architecture
   - Streamlit app architecture

## 📂 Module Structure

```
module4/
├── lessons/                  # Lesson content in markdown format
│   ├── lesson1.md            # Lesson 1: Document Processing Fundamentals
│   ├── lesson2.md            # Lesson 2: Chunking Strategies for Optimal Retrieval
│   ├── lesson3.md            # Lesson 3: Embedding Selection & Generation
│   ├── lesson4.md            # Lesson 4: Metadata Extraction & Management
│   ├── lesson5.md            # Lesson 5: Building a Document Q&A System
│   └── module4_diagrams.md   # Diagrams for Module 4 concepts
├── code/                     # Code examples and implementations
│   ├── README.md             # Code directory documentation
│   ├── document_loaders.py   # Document loader implementations for various formats
│   ├── test_document_loaders.py # Test script for document loaders
│   ├── text_splitters.py     # Text splitting strategies implementation
│   ├── test_text_splitters.py # Test script for text splitters
│   ├── embedding_pipelines.py # Embedding generation pipelines
│   ├── test_embedding_pipelines.py # Test script for embedding pipelines
│   ├── metadata_extractors.py # Metadata extraction systems
│   ├── test_metadata_extractors.py # Test script for metadata extractors
│   ├── rag_system.py         # Simple RAG system implementation
│   ├── test_rag_system.py    # Test script for RAG system
│   ├── document_qa.py        # Document Q&A system implementation
│   └── test_document_qa.py   # Test script for document Q&A system
├── exercises/                # Practice exercises and solutions
│   ├── README.md             # Exercises directory documentation
│   ├── lesson1_exercises.py  # Solutions for lesson 1 exercises
│   ├── test_lesson1_exercises.py # Tests for lesson 1 solutions
│   ├── lesson2_exercises.py  # Solutions for lesson 2 exercises
│   ├── test_lesson2_exercises.py # Tests for lesson 2 solutions
│   ├── lesson3_exercises.py  # Solutions for lesson 3 exercises
│   ├── test_lesson3_exercises.py # Tests for lesson 3 solutions
│   ├── lesson4_exercises.py  # Solutions for lesson 4 exercises
│   ├── test_lesson4_exercises.py # Tests for lesson 4 solutions
│   ├── lesson5_exercises.py  # Solutions for lesson 5 exercises
│   └── test_lesson5_exercises.py # Tests for lesson 5 solutions
├── streamlit_app/            # Interactive Streamlit application
│   ├── app.py                # Main Streamlit application file
│   ├── rag_components.py     # RAG system components for the app
│   ├── requirements.txt      # Dependencies for the Streamlit app
│   ├── README.md             # Documentation for the Streamlit app
│   └── .streamlit/           # Streamlit configuration
│       └── config.toml       # Streamlit theme and settings
├── diagrams/                 # Additional diagrams and visualizations
│   ├── README.md             # Diagrams directory documentation
│   └── module4_progressive_journey.md # Progressive journey through module concepts
├── demo_document_processing.py # Demo script for document processing
└── demo_document_qa.py       # Demo script for the document Q&A system
```

## 🎯 Learning Objectives

By the end of this module, you will:
- 📄 Understand document processing pipelines for various file formats
- ✂️ Master chunking strategies for optimal retrieval
- 🔢 Learn embedding selection and generation for different content types
- 🏷️ Implement metadata extraction systems for improved retrieval
- 🔄 Build a simple RAG system combining retrieval and generation
- 💬 Create a complete document Q&A system that can answer questions from multiple documents

## 🚀 Getting Started

1. Start by reading through the lessons in order:
   - **lessons/lesson1.md**: Document Processing Fundamentals
   - **lessons/lesson2.md**: Chunking Strategies for Optimal Retrieval
   - **lessons/lesson3.md**: Embedding Selection & Generation
   - **lessons/lesson4.md**: Metadata Extraction & Management
   - **lessons/lesson5.md**: Building a Document Q&A System

2. Examine the code examples for each lesson:
   - Lesson 1: **code/document_loaders.py**
   - Lesson 2: **code/text_splitters.py**
   - Lesson 3: **code/embedding_pipelines.py**
   - Lesson 4: **code/metadata_extractors.py**
   - Lesson 5: **code/rag_system.py** and **code/document_qa.py**

3. Complete the practice exercises to reinforce your learning:
   - **exercises/lesson1_exercises.py**: Document processing exercises
   - **exercises/lesson2_exercises.py**: Text splitting exercises
   - **exercises/lesson3_exercises.py**: Embedding generation exercises
   - **exercises/lesson4_exercises.py**: Metadata extraction exercises
   - **exercises/lesson5_exercises.py**: RAG system exercises

4. Run the demo scripts to see document processing and Q&A in action:
   - **demo_document_processing.py**: Demonstrates document loading and processing
   - **demo_document_qa.py**: Shows a simple document Q&A system

5. Try the interactive Streamlit application:
   - Navigate to the **streamlit_app** directory
   - Install the required dependencies: `pip install -r requirements.txt`
   - Run the app: `streamlit run app.py`
   - Upload documents, ask questions, and explore the full RAG system

6. Explore the progressive journey diagrams to understand how the components fit together:
   - **diagrams/module4_progressive_journey.md**: Step-by-step diagrams of the system architecture

## 🧠 Key Concepts

### Document Processing
- Document loaders for various file formats (PDF, TXT, DOCX, etc.)
- Text extraction and normalization
- Document structure preservation
- Handling different document types (articles, books, code, etc.)

### Chunking Strategies
- Size-based chunking
- Semantic chunking
- Recursive chunking
- Token-based chunking
- Content-aware chunking

### Embedding Selection & Generation
- Embedding models comparison
- Domain-specific embeddings
- Multi-modal embeddings
- Embedding pipelines
- Dimensionality considerations

### Metadata Extraction & Management
- Automatic metadata extraction
- Custom metadata tagging
- Metadata filtering for retrieval
- Structured metadata schemas
- Using metadata to improve relevance

### RAG Systems
- Vector database integration
- Retrieval mechanisms
- Context augmentation
- Generation with retrieved context
- Hybrid retrieval approaches

## 🔍 Mini-Project: Document Q&A System

The culminating project for this module is building a Document Q&A System that can:
- Process multiple documents in different formats
- Create an optimized vector index of document content
- Answer questions with direct references to document sections
- Combine information from multiple documents when needed
- Handle queries about document metadata (author, date, etc.)

This project has been implemented in two ways:

1. **Core Implementation**: The `code/rag_system.py` and `code/document_qa.py` files provide a basic implementation of the Document Q&A System that demonstrates the core concepts.

2. **Interactive Streamlit App**: The `streamlit_app` directory contains a complete, interactive web application that brings the Document Q&A System to life with a user-friendly interface. The app includes:
   - Document upload and processing for multiple file formats
   - Question answering with source attribution
   - Document analysis and visualization
   - Metadata exploration
   - Document summarization
   - Export functionality for conversation history and document collection

To run the Streamlit app:
```bash
cd module4/streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

These implementations demonstrate your understanding of document processing and RAG systems, providing a foundation for building more advanced knowledge-based applications.

## 📚 Resources

### Document Processing
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/en/latest/)
- [Python-DOCX Documentation](https://python-docx.readthedocs.io/en/latest/)

### Chunking Strategies
- [Text Splitting Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Recursive Character Text Splitting](https://www.pinecone.io/learn/recursive-character-text-splitting/)

### Embeddings
- [Sentence Transformers Documentation](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Hugging Face Embeddings](https://huggingface.co/blog/embeddings)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

### RAG Systems
- [RAG Pattern Overview](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Database Fundamentals](https://www.pinecone.io/learn/vector-database/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Components](https://docs.streamlit.io/library/components)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud/get-started)
