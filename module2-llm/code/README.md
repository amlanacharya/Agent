# 🧩 Module 2-LLM: Code Examples

## 📚 Overview

This directory contains all the code examples and implementations for Module 2-LLM: Memory Systems with Groq API. These examples demonstrate different memory types, vector database fundamentals, and retrieval patterns for contextual memory, all enhanced with real LLM integration.

## 🔍 File Descriptions

### Core Implementations
- **groq_client.py**: Utilities for interacting with the Groq API for text generation and embeddings
- **memory_types.py**: Implementation of different memory types enhanced with LLM capabilities
- **vector_store.py**: Vector database implementation with real embeddings from SentenceTransformers
- **retrieval_agent.py**: Agent with LLM-enhanced retrieval capabilities for contextual memory
- **knowledge_base.py**: Knowledge base implementation with LLM integration for storage and retrieval
- **kb_agent.py**: Complete knowledge base assistant implementation powered by Groq

### Test Scripts
- **test_memory_types.py**: Tests for the LLM-enhanced memory implementations
- **test_vector_store.py**: Tests for the vector database with real embeddings
- **test_retrieval_agent.py**: Tests for the LLM-enhanced retrieval agent
- **test_kb_agent.py**: Tests for the knowledge base assistant with Groq

## 🚀 Running the Examples

You can run any of the examples directly from the command line:

```bash
# Run from the project root
python -m module2-llm.code.memory_types
python -m module2-llm.code.vector_store
python -m module2-llm.code.retrieval_agent
python -m module2-llm.code.kb_agent
```

To run the tests:

```bash
# Run from the project root
python -m module2-llm.code.test_memory_types
python -m module2-llm.code.test_vector_store
python -m module2-llm.code.test_retrieval_agent
python -m module2-llm.code.test_kb_agent
```

## 📋 Implementation Notes

- The implementations require a valid Groq API key set in your environment
- The memory implementations use LLM capabilities for summarization and prioritization
- The vector store uses SentenceTransformers for generating real embeddings
- The retrieval agent uses LLM to enhance queries and improve retrieval quality
- The knowledge base assistant combines all components into a complete system
- All implementations are designed to handle API errors gracefully

## 🔄 LLM Integration

> 💡 **Note**: This module uses real LLM integration through the Groq API for both text generation and embeddings. This provides more sophisticated capabilities but requires API keys and internet connectivity.

## 🧪 Example Usage

Here's a simple example of how to use the Groq client:

```python
# Example code snippet showing basic usage
from module2-llm.code.groq_client import GroqClient

# Create a client instance
client = GroqClient()

# Generate text
response = client.generate_text("Explain the concept of vector embeddings")
print(response)

# Generate embeddings
embedding = client.generate_embedding("This is a sample text for embedding")
print(f"Embedding dimension: {len(embedding)}")
```

And here's how to use the knowledge base assistant:

```python
# Example code snippet showing knowledge base usage
from module2-llm.code.kb_agent import KnowledgeBaseAgent

# Create an instance
kb_agent = KnowledgeBaseAgent()

# Add knowledge
response = kb_agent.process_input("Remember that Paris is the capital of France")
print(response)

# Query knowledge
response = kb_agent.process_input("What is the capital of France?")
print(response)
```

## 🛠️ Extending the Code

Here are some ideas for extending or customizing the implementations:

1. Integrate with different LLM providers (OpenAI, Anthropic, etc.)
2. Add more sophisticated retrieval strategies like re-ranking with LLMs
3. Implement active learning mechanisms to improve the knowledge base over time
4. Add persistence to save the knowledge base between sessions
5. Create specialized knowledge bases for different domains with domain-specific embeddings

## 📚 Related Resources

- [Groq API Documentation](https://console.groq.com/docs/quickstart)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [Embeddings Explained](https://platform.openai.com/docs/guides/embeddings)
