"""
Demo Script for Document Q&A System
---------------------------------
This script demonstrates the functionality of the Document Q&A system
by loading sample documents, creating a RAG system, and answering questions.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from code.embedding_pipelines import SentenceTransformerEmbeddings, HashEmbeddings
    from code.rag_system import SimpleRAGSystem, SimpleLLMClient
    from code.document_qa import DocumentQASystem
except ImportError:
    logger.error("Failed to import required modules. Make sure you're running from the correct directory.")
    sys.exit(1)

# Try to import the GroqClient for LLM integration
try:
    # Try module3 path first
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from module3.code.groq_client import GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    try:
        # Try module2-llm path
        from module2_llm.code.groq_client import GroqClient
        GROQ_AVAILABLE = True
    except ImportError:
        GROQ_AVAILABLE = False
        logger.warning("GroqClient not available. Using simulated responses.")


def load_sample_documents():
    """Load sample documents for demonstration."""
    return [
        {
            "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) with external knowledge. In a standard RAG system, the retrieval component searches through a knowledge base to find relevant information, which is then added to the prompt before the LLM generates a response. This approach bridges the gap between the parametric knowledge stored in an LLM's weights and the ability to access and use up-to-date, specific information from external sources.",
            "metadata": {
                "source": "RAG Overview",
                "author": "AI Research Team",
                "date": "2023-05-10",
                "section": "Introduction"
            }
        },
        {
            "content": "Vector databases are specialized databases designed to store and search vector embeddings efficiently. They use algorithms like Approximate Nearest Neighbor (ANN) search to quickly find similar vectors. Popular vector database implementations include FAISS (Facebook AI Similarity Search) and ChromaDB. FAISS is optimized for high-dimensional vectors and supports both CPU and GPU acceleration, while ChromaDB offers persistence and metadata filtering capabilities.",
            "metadata": {
                "source": "Vector Database Guide",
                "author": "Database Experts",
                "date": "2022-11-15",
                "section": "Vector Databases"
            }
        },
        {
            "content": "Embedding models convert text into numerical vector representations that capture semantic meaning. These models are trained on large text corpora to learn the relationships between words and concepts. Popular embedding models include SentenceTransformers, which are based on transformer architectures like BERT, and OpenAI's text-embedding models. The quality of embeddings significantly impacts retrieval performance in RAG systems.",
            "metadata": {
                "source": "Embedding Models Overview",
                "author": "NLP Research Group",
                "date": "2023-02-28",
                "section": "Text Embeddings"
            }
        },
        {
            "content": "Document chunking is the process of splitting documents into smaller, manageable pieces for embedding and retrieval. Effective chunking strategies balance chunk size (typically 256-1024 tokens) with semantic coherence. Common approaches include splitting by fixed token count, by paragraph, or recursively by content structure. Chunk overlap (typically 10-20%) helps maintain context across chunk boundaries.",
            "metadata": {
                "source": "Document Processing Techniques",
                "author": "Content Processing Team",
                "date": "2023-03-15",
                "section": "Chunking Strategies"
            }
        },
        {
            "content": "Metadata extraction enhances RAG systems by capturing structured information about documents. This includes bibliographic data (author, title, date), content-based metadata (topics, entities, sentiment), and structural metadata (sections, headings, formatting). Metadata can be used for filtering, boosting relevance, and providing additional context during retrieval and generation.",
            "metadata": {
                "source": "Metadata in RAG Systems",
                "author": "Information Retrieval Lab",
                "date": "2023-04-22",
                "section": "Metadata Extraction"
            }
        }
    ]


def create_embeddings(documents):
    """Create embeddings for the documents."""
    # Try to use SentenceTransformers if available
    try:
        embedding_model = SentenceTransformerEmbeddings()
        logger.info("Using SentenceTransformer for embeddings.")
    except Exception:
        embedding_model = HashEmbeddings()
        logger.info("Using HashEmbeddings as fallback.")
    
    # Extract content for embedding
    contents = [doc["content"] for doc in documents]
    
    # Generate embeddings
    embeddings = embedding_model.embed_documents(contents)
    logger.info(f"Generated {len(embeddings)} embeddings.")
    
    return embedding_model, embeddings


def main():
    """Main function to demonstrate the Document Q&A system."""
    print("=" * 80)
    print("Document Q&A System Demo")
    print("=" * 80)
    
    # Load sample documents
    print("\nLoading sample documents...")
    documents = load_sample_documents()
    print(f"Loaded {len(documents)} documents.")
    
    # Create embeddings
    print("\nGenerating embeddings...")
    embedding_model, embeddings = create_embeddings(documents)
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    try:
        rag_system = SimpleRAGSystem(documents, embeddings, vector_store_type="faiss")
        print("Using FAISS for vector search.")
    except ImportError:
        try:
            rag_system = SimpleRAGSystem(documents, embeddings, vector_store_type="chroma")
            print("Using ChromaDB for vector search.")
        except ImportError:
            print("Error: Neither FAISS nor ChromaDB is available.")
            print("Please install one of them: 'pip install faiss-cpu' or 'pip install chromadb'")
            return
    
    # Initialize LLM client
    print("\nInitializing LLM client...")
    if GROQ_AVAILABLE:
        llm_client = GroqClient()
        print("Using GroqClient for text generation.")
    else:
        llm_client = SimpleLLMClient()
        print("Using SimpleLLMClient for text generation (simulated responses).")
    
    # Create Document Q&A system
    print("\nCreating Document Q&A system...")
    qa_system = DocumentQASystem(
        rag_system=rag_system,
        embedding_model=embedding_model,
        llm_client=llm_client
    )
    
    # Demo questions
    questions = [
        "What is RAG and how does it work?",
        "Compare FAISS and ChromaDB.",
        "What are embedding models used for?",
        "Who wrote the guide on vector databases?",
        "What are the best practices for document chunking?"
    ]
    
    # Answer questions
    print("\nAnswering questions...")
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        
        # Get answer
        response = qa_system.answer_question(question, k=3)
        
        # Print answer
        print(f"\nAnswer: {response['answer']}")
        print(f"Confidence: {response['confidence']:.2f}")
        
        # Print sources
        print("\nSources:")
        for j, source in enumerate(response['sources']):
            if isinstance(source, dict) and 'source' in source:
                print(f"  {j+1}. {source.get('source', 'Unknown')}")
        
        print("-" * 80)
    
    # Demo metadata query
    print("\nMetadata Query Example:")
    metadata_question = "Who wrote the document about metadata extraction?"
    print(f"Question: {metadata_question}")
    
    # Get answer
    metadata_response = qa_system.answer_question(metadata_question)
    
    # Print answer
    print(f"\nAnswer: {metadata_response['answer']}")
    print(f"Is metadata query: {metadata_response['is_metadata_query']}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
