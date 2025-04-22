# 🧠 Module 2 LLM Version: Memory Systems with Groq API

![Memory Banner](https://media.giphy.com/media/l0HlQXlQ3nHyLMvte/giphy.gif)

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
│   └── lesson4.md            # Lesson 4: Building the Knowledge Base Assistant with Groq
├── code/                     # Code examples and implementations
│   ├── groq_client.py        # Groq API integration utilities
│   ├── memory_types.py       # LLM-powered memory implementations
│   ├── test_memory_types.py  # Test script for memory implementations
│   ├── vector_store.py       # Vector database with real embeddings
│   ├── retrieval_agent.py    # LLM-enhanced retrieval agent
│   ├── knowledge_base.py     # Knowledge base with LLM integration
│   ├── kb_agent.py           # Knowledge base assistant with Groq
│   └── test_kb_agent.py      # Test script for the knowledge base assistant
└── exercises/                # Practice exercises and solutions
    ├── memory_exercises.py   # Solutions for lesson 1 exercises
    ├── vector_exercises.py   # Solutions for lesson 2 exercises
    └── retrieval_exercises.py # Solutions for lesson 3 exercises
```

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
   - Lesson 1: **code/memory_types.py**
   - Lesson 2: **code/vector_store.py**
   - Lesson 3: **code/retrieval_agent.py**
   - Lesson 4: **code/kb_agent.py**

5. Run the test scripts to see the memory systems in action:
   ```
   python module2-llm/code/test_memory_types.py
   python module2-llm/code/test_vector_store.py
   python module2-llm/code/test_retrieval_agent.py
   python module2-llm/code/test_kb_agent.py
   ```

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand different memory types for AI agents and how LLMs enhance them
- Learn vector database fundamentals with real embeddings
- Master retrieval patterns for contextual memory with LLM enhancement
- Build a knowledge base assistant powered by Groq LLMs
- Gain practical experience with the Groq API for both text generation and embeddings

## 🧪 Practice Exercises

The lessons include several practice exercises to help reinforce your learning:
1. Implementing LLM-enhanced memory types
2. Creating a vector database with real embeddings
3. Designing effective retrieval patterns with LLM query enhancement

## 📝 Mini-Project: Knowledge Base Assistant with Groq

The culminating project for this module is building a knowledge base assistant that can:
- Store and retrieve information using vector embeddings
- Generate natural language responses using Groq LLMs
- Learn from conversations and improve over time
- Provide citations and sources for its answers
- Handle uncertainty appropriately

## 🔄 Comparing Simulated vs. Real LLM Approaches

Throughout this module, we'll highlight the differences between:
- The simulated approach used in Module 1 and standard Module 2
- The real LLM integration approach used in this version

This comparison will help you understand both the conceptual foundations and the practical implementation of LLM-powered agent systems.

## 📚 Resources

- [Groq API Documentation](https://console.groq.com/docs/quickstart)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)
- [Embeddings Explained](https://platform.openai.com/docs/guides/embeddings)
