# 🧠 Module 2-LLM: Memory Systems with Groq API

## 📚 Overview

Welcome to the LLM-integrated version of Module 2 of the Accelerated Agentic AI Mastery course! This module covers memory systems for AI agents, focusing on different memory types (working, short-term, long-term), vector database fundamentals, and retrieval patterns for contextual memory - all powered by real LLM integration using the Groq API.

> 💡 **Note**: Unlike Module 1 and the standard Module 2 which use simulated responses, this version integrates with actual Large Language Models (LLMs) through the Groq API for text generation and SentenceTransformers for embeddings. This allows you to experience the full power of LLMs in memory systems while learning the fundamentals of agent memory architecture.

## 📂 Module Structure

```
module2-llm/
├── lessons/                  # Lesson content in markdown format
│   ├── lesson1.md            # Lesson 1: Memory Types for AI Agents with LLM
│   ├── lesson2.md            # Lesson 2: Vector Database Fundamentals with Embeddings
│   ├── lesson3.md            # Lesson 3: Retrieval Patterns with LLM Enhancement
│   ├── lesson4.md            # Lesson 4: Building the Knowledge Base Assistant with Groq
│   └── module2-llm_diagrams.md # Diagrams for Module 2-LLM concepts
├── code/                     # Code examples and implementations
│   ├── README.md             # Code directory documentation
│   ├── groq_client.py        # Groq API integration utilities
│   ├── memory_types.py       # LLM-powered memory implementations
│   ├── test_memory_types.py  # Test script for memory implementations
│   ├── vector_store.py       # Vector database with real embeddings
│   ├── test_vector_store.py  # Test script for vector store
│   ├── retrieval_agent.py    # LLM-enhanced retrieval agent
│   ├── test_retrieval_agent.py # Test script for retrieval agent
│   ├── knowledge_base.py     # Knowledge base with LLM integration
│   ├── kb_agent.py           # Knowledge base assistant with Groq
│   └── test_kb_agent.py      # Test script for the knowledge base assistant
├── exercises/                # Practice exercises and solutions
│   ├── README.md             # Exercises directory documentation
│   ├── memory_exercises.py   # Solutions for lesson 1 exercises
│   ├── test_memory_exercises.py # Tests for lesson 1 solutions
│   ├── vector_exercises.py   # Solutions for lesson 2 exercises
│   ├── test_vector_exercises.py # Tests for lesson 2 solutions
│   ├── retrieval_exercises.py # Solutions for lesson 3 exercises
│   └── test_retrieval_exercises.py # Tests for lesson 3 solutions
└── demo_memory_systems_llm.py # Interactive demonstration for all LLM-enhanced memory systems
```

## 🎯 Learning Objectives

By the end of this module, you will:
- 🧠 Understand different memory types for AI agents and how LLMs enhance them
- 🔢 Learn vector database fundamentals with real embeddings
- 🔍 Master retrieval patterns for contextual memory with LLM enhancement
- 📚 Build a knowledge base assistant powered by Groq LLMs
- 🔄 Gain practical experience with the Groq API for both text generation and embeddings

## 🚀 Getting Started

### Prerequisites

Before starting this module, you'll need:
1. A Groq API key (sign up at [https://console.groq.com/](https://console.groq.com/))
2. Python 3.8+ installed
3. Basic understanding of agent concepts from Module 1

### Setup

1. Set up your Groq API key:
   ```python
   # In your .env file
   GROQ_API_KEY=your_key_here
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individual packages:
   ```bash
   pip install requests numpy python-dotenv sentence-transformers scikit-learn
   ```

3. Start by reading through the lessons in order:
   - **lessons/lesson1.md**: Memory Types for AI Agents with LLM
   - **lessons/lesson2.md**: Vector Database Fundamentals with Embeddings
   - **lessons/lesson3.md**: Retrieval Patterns with LLM Enhancement
   - **lessons/lesson4.md**: Building the Knowledge Base Assistant with Groq

4. Examine the code examples for each lesson:
   - Lesson 1: **code/memory_types.py** and **code/groq_client.py**
   - Lesson 2: **code/vector_store.py**
   - Lesson 3: **code/retrieval_agent.py**
   - Lesson 4: **code/kb_agent.py** and **code/knowledge_base.py**

5. Run the test scripts to see the memory systems in action:
   ```
   python module2-llm/code/test_memory_types.py
   python module2-llm/code/test_vector_store.py
   python module2-llm/code/test_retrieval_agent.py
   python module2-llm/code/test_kb_agent.py
   ```

6. Try the interactive demos by running:
   ```
   # Run the unified demo with a menu of all LLM-enhanced memory systems
   python module2-llm/demo_memory_systems_llm.py

   # Or run a specific component demo directly
   python module2-llm/demo_memory_systems_llm.py groq       # Groq Client
   python module2-llm/demo_memory_systems_llm.py working    # LLM-Enhanced Working Memory
   python module2-llm/demo_memory_systems_llm.py short      # LLM-Enhanced Short-Term Memory
   python module2-llm/demo_memory_systems_llm.py long       # LLM-Enhanced Long-Term Memory
   python module2-llm/demo_memory_systems_llm.py episodic   # LLM-Enhanced Episodic Memory
   python module2-llm/demo_memory_systems_llm.py system     # LLM-Enhanced Agent Memory System
   python module2-llm/demo_memory_systems_llm.py vector     # LLM-Enhanced Vector Store
   python module2-llm/demo_memory_systems_llm.py retrieval  # LLM-Enhanced Retrieval Agent
   python module2-llm/demo_memory_systems_llm.py kb         # LLM-Enhanced Knowledge Base
   python module2-llm/demo_memory_systems_llm.py assistant  # LLM-Enhanced Knowledge Base Assistant
   ```

## 🧪 Practice Exercises

The lessons include several practice exercises to help reinforce your learning:
1. 🧠 Implementing LLM-enhanced memory types with summarization and prioritization
2. 🔢 Creating a vector database with real embeddings from SentenceTransformers
3. 🔍 Designing effective retrieval patterns with LLM query enhancement
4. 📚 Building components for the knowledge base assistant with Groq integration

## 📝 Mini-Project: Knowledge Base Assistant with Groq

Throughout this module, you'll be building a Knowledge Base Assistant that can:
- 📄 Store and retrieve information using vector embeddings
- 🧩 Generate natural language responses using Groq LLMs
- ✅ Learn from conversations and improve over time
- 🔍 Provide citations and sources for its answers
- 📊 Handle uncertainty appropriately

## 🔧 Tools & Technologies

- Groq API for LLM text generation
- SentenceTransformers for generating embeddings
- Python for implementing memory systems
- Vector representations for semantic similarity
- Simple vector databases for information storage
- Retrieval algorithms with LLM enhancement

## 🧠 Skills You'll Develop

- LLM integration for agent systems
- Embedding generation and management
- Vector database implementation with real embeddings
- Retrieval-augmented generation techniques
- API integration and error handling
- Prompt engineering for memory systems

## 🔄 Comparing Simulated vs. Real LLM Approaches

Throughout this module, we'll highlight the differences between:
- The simulated approach used in Module 1 and standard Module 2
- The real LLM integration approach used in this version

This comparison will help you understand both the conceptual foundations and the practical implementation of LLM-powered agent systems.

## 📚 Additional Resources

- [Groq API Documentation](https://console.groq.com/docs/quickstart)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [Embeddings Explained](https://platform.openai.com/docs/guides/embeddings)

## 🤔 Need Help?

If you get stuck or have questions:
- Review the lesson material again
- Check the example solutions
- Experiment with different approaches
- Discuss with fellow students
- Check the Groq API documentation for specific API issues

Happy learning! 🚀
