"""
Test script for the RAG evaluation framework.

This script creates a simple RAG system and evaluates it using the
evaluation framework implemented in evaluation.py.
"""

import os
import sys
from typing import List, Dict, Any

# Add parent directory to path to import evaluation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation framework
from evaluation import (
    RetrievalEvaluator,
    GenerationEvaluator,
    RAGEvaluator,
    EvaluationVisualizer,
    example_usage
)

# Import required packages
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


def create_test_documents() -> List[Document]:
    """Create a set of test documents."""
    return [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval and generation for improved LLM responses.",
            metadata={"source": "document1.txt", "page": 1, "topic": "RAG"}
        ),
        Document(
            page_content="Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant information before generating responses.",
            metadata={"source": "document2.txt", "page": 5, "topic": "RAG"}
        ),
        Document(
            page_content="RAG systems use vector databases to store and retrieve relevant context for language model queries.",
            metadata={"source": "document3.txt", "page": 12, "topic": "RAG"}
        ),
        Document(
            page_content="Vector databases like FAISS and Chroma are commonly used in RAG systems for efficient similarity search.",
            metadata={"source": "document4.txt", "page": 8, "topic": "Vector Databases"}
        ),
        Document(
            page_content="Embedding models convert text into vector representations that capture semantic meaning.",
            metadata={"source": "document5.txt", "page": 3, "topic": "Embeddings"}
        ),
        Document(
            page_content="Text chunking strategies are important for effective retrieval in RAG systems.",
            metadata={"source": "document6.txt", "page": 7, "topic": "Chunking"}
        ),
        Document(
            page_content="Reranking improves retrieval quality by applying more sophisticated models to initial search results.",
            metadata={"source": "document7.txt", "page": 15, "topic": "Reranking"}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by language models, including RAG systems.",
            metadata={"source": "document8.txt", "page": 2, "topic": "LangChain"}
        )
    ]


class SimpleRAGSystem:
    """A simple RAG system for testing the evaluation framework."""

    def __init__(self, documents: List[Document]):
        """Initialize the RAG system with documents."""
        # Initialize embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except:
            # Mock embedding model if HuggingFace is not available
            class MockEmbeddings:
                def embed_documents(self, texts):
                    return [[0.1] * 384 for _ in texts]

                def embed_query(self, text):
                    return [0.1] * 384

            self.embedding_model = MockEmbeddings()

        # Create vector store
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        except:
            # Mock vector store if FAISS is not available
            class MockVectorStore:
                def __init__(self, docs):
                    self.docs = docs

                def similarity_search(self, query, k=4):
                    import random
                    return random.sample(self.docs, min(k, len(self.docs)))

            self.vectorstore = MockVectorStore(documents)

        # Store documents
        self.documents = documents

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query."""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except:
            # Fallback to simple keyword matching
            query_keywords = set(query.lower().split())
            scored_docs = []

            for doc in self.documents:
                doc_words = set(doc.page_content.lower().split())
                score = len(query_keywords.intersection(doc_words)) / len(query_keywords) if query_keywords else 0
                scored_docs.append((doc, score))

            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:k]]

    def generate(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generate a response based on the query and retrieved documents."""
        # Simple response generation by combining document content
        if not retrieved_docs:
            return "I don't have enough information to answer that question."

        # Extract relevant sentences from documents
        relevant_content = []
        for doc in retrieved_docs:
            sentences = doc.page_content.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and any(keyword in sentence.lower() for keyword in query.lower().split()):
                    relevant_content.append(sentence)

        # Combine relevant content
        if relevant_content:
            response = " ".join(relevant_content) + "."
        else:
            # Use the first document if no relevant sentences found
            response = retrieved_docs[0].page_content

        return response

    def invoke(self, query: str) -> Dict[str, Any]:
        """Process a query and return response with source documents."""
        # Retrieve documents
        retrieved_docs = self.retrieve(query)

        # Generate response
        response = self.generate(query, retrieved_docs)

        # Return response and source documents
        return {
            "query": query,
            "response": response,
            "source_documents": retrieved_docs
        }


def test_retrieval_evaluation():
    """Test the retrieval evaluation functionality."""
    print("\n=== Testing Retrieval Evaluation ===")

    # Create test documents
    documents = create_test_documents()

    # Create RAG system
    rag_system = SimpleRAGSystem(documents)

    # Define test queries
    queries = [
        "What is RAG?",
        "How do vector databases work?",
        "What are embedding models?",
        "Explain text chunking strategies"
    ]

    # Define relevant documents for each query
    relevant_docs_map = {
        "What is RAG?": [documents[0], documents[1], documents[2]],
        "How do vector databases work?": [documents[3]],
        "What are embedding models?": [documents[4]],
        "Explain text chunking strategies": [documents[5]]
    }

    # Test retrieval for each query
    for query in queries:
        print(f"\nQuery: {query}")

        # Get relevant documents
        relevant_docs = relevant_docs_map.get(query, [])

        # Create evaluator
        evaluator = RetrievalEvaluator(relevant_docs)

        # Retrieve documents
        retrieved_docs = rag_system.retrieve(query)

        # Print retrieved documents
        print("Retrieved documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc.page_content[:100]}...")

        # Evaluate retrieval
        precision = evaluator.precision_at_k(retrieved_docs, k=4)
        recall = evaluator.recall_at_k(retrieved_docs, k=4)
        mrr = evaluator.mean_reciprocal_rank(retrieved_docs)

        # Print metrics
        print("Retrieval metrics:")
        print(f"  Precision@4: {precision:.4f}")
        print(f"  Recall@4: {recall:.4f}")
        print(f"  MRR: {mrr:.4f}")


def test_generation_evaluation():
    """Test the generation evaluation functionality."""
    print("\n=== Testing Generation Evaluation ===")

    # Create test documents
    documents = create_test_documents()

    # Create RAG system
    rag_system = SimpleRAGSystem(documents)

    # Define test query
    query = "What is RAG and how does it work?"

    # Process query
    result = rag_system.invoke(query)
    response = result["response"]
    retrieved_docs = result["source_documents"]

    # Print query and response
    print(f"\nQuery: {query}")
    print(f"Response: {response}")

    # Create evaluator
    evaluator = GenerationEvaluator()

    # Evaluate generation
    relevance = evaluator.evaluate_relevance(query, response, retrieved_docs)
    faithfulness = evaluator.evaluate_faithfulness(response, retrieved_docs)
    coherence = evaluator.evaluate_coherence(response)

    # Print metrics
    print("Generation metrics:")
    print(f"  Relevance: {relevance:.4f}")
    print(f"  Faithfulness: {faithfulness:.4f}")
    print(f"  Coherence: {coherence:.4f}")


def test_rag_evaluation():
    """Test the end-to-end RAG evaluation functionality."""
    print("\n=== Testing End-to-End RAG Evaluation ===")

    # Create test documents
    documents = create_test_documents()

    # Create RAG system
    rag_system = SimpleRAGSystem(documents)

    # Define test query
    query = "What is RAG and how does it work?"

    # Process query
    result = rag_system.invoke(query)
    response = result["response"]
    retrieved_docs = result["source_documents"]

    # Print query and response
    print(f"\nQuery: {query}")
    print(f"Response: {response}")

    # Define relevant documents
    relevant_docs = [documents[0], documents[1], documents[2]]

    # Create evaluator
    evaluator = RAGEvaluator(relevant_docs=relevant_docs)

    # Evaluate RAG system
    metrics = evaluator.evaluate(query, response, retrieved_docs)

    # Print metrics
    print("RAG evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Try to visualize results
    try:
        EvaluationVisualizer.plot_metrics(metrics)
    except Exception as e:
        print(f"Visualization error: {e}")


def main():
    """Run all tests."""
    # Test retrieval evaluation
    test_retrieval_evaluation()

    # Test generation evaluation
    test_generation_evaluation()

    # Test RAG evaluation
    test_rag_evaluation()

    # Run example usage from evaluation.py
    print("\n=== Running Example Usage ===")
    try:
        example_usage()
    except Exception as e:
        print(f"Example usage error: {e}")


if __name__ == "__main__":
    main()
