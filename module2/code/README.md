# 🧩 Module 2: Code Examples

## 📚 Overview

This directory contains all the code examples and implementations for Module 2: Memory Systems. These examples demonstrate different memory types, vector database fundamentals, and retrieval patterns for contextual memory.

## 🔍 File Descriptions

### Core Implementations
- **memory_types.py**: Implementation of different memory types (working, short-term, long-term)
- **vector_store.py**: Simple vector database implementation with similarity search
- **retrieval_agent.py**: Agent with retrieval capabilities for contextual memory
- **knowledge_base.py**: Knowledge base implementation for storing and retrieving information
- **kb_agent.py**: Complete knowledge base assistant implementation

### Test Scripts
- **test_memory_types.py**: Tests for the memory type implementations
- **test_vector_store.py**: Tests for the vector database implementation
- **test_retrieval_agent.py**: Tests for the retrieval agent
- **test_kb_agent.py**: Tests for the knowledge base assistant

## 🚀 Running the Examples

You can run any of the examples directly from the command line:

```bash
# Run from the project root
python -m module2.code.memory_types
python -m module2.code.vector_store
python -m module2.code.retrieval_agent
python -m module2.code.kb_agent
```

To run the tests:

```bash
# Run from the project root
python -m module2.code.test_memory_types
python -m module2.code.test_vector_store
python -m module2.code.test_retrieval_agent
python -m module2.code.test_kb_agent
```

## 📋 Implementation Notes

- The memory implementations follow a consistent interface for easy interchangeability
- The vector store uses a simplified embedding approach for demonstration purposes
- The retrieval agent demonstrates how to integrate memory systems with the agent loop
- The knowledge base assistant builds on all previous components to create a complete system
- All implementations are designed to be modular and extensible

## 🔄 LLM Integration

> 💡 **Note**: Module 2 primarily uses simulated LLM responses, but introduces vector databases that can be used with real embeddings. The code is designed to work with either simulated or real embeddings.

## 🧪 Example Usage

Here's a simple example of how to use the memory types:

```python
# Example code snippet showing basic usage
from module2.code.memory_types import WorkingMemory, LongTermMemory

# Create memory instances
working_memory = WorkingMemory(capacity=5)
long_term_memory = LongTermMemory()

# Store information
working_memory.add("The weather today is sunny")
long_term_memory.add("Python was created by Guido van Rossum")

# Retrieve information
print(working_memory.get_all())
print(long_term_memory.search("Python"))
```

And here's how to use the knowledge base assistant:

```python
# Example code snippet showing knowledge base usage
from module2.code.kb_agent import KnowledgeBaseAgent

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

1. Integrate with a real vector database like FAISS or ChromaDB
2. Add more sophisticated retrieval strategies like re-ranking
3. Implement forgetting mechanisms for the memory systems
4. Add persistence to save the knowledge base between sessions
5. Create specialized knowledge bases for different domains

## 📚 Related Resources

- [Vector Database Fundamentals](https://www.pinecone.io/learn/vector-database/)
- [Semantic Search Tutorial](https://www.sbert.net/examples/applications/semantic-search/README.html)
- [Memory Systems in LangChain](https://python.langchain.com/docs/modules/memory/)
