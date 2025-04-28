# 🧩 Module 5: Code Examples

## 📚 Overview

This directory contains all the code examples and implementations for Module 5: Advanced RAG Systems. These examples demonstrate advanced retrieval strategies, query transformation techniques, reranking systems, adaptive RAG, and a complete Research Literature Assistant implementation.

## 🔍 File Descriptions

### Core Implementations
- **advanced_retrieval.py**: Advanced retrieval strategies beyond basic vector search
- **query_transformation.py**: Query expansion and reformulation techniques
- **reranking.py**: Reranking systems for result optimization
- **adaptive_rag.py**: Self-querying and adaptive RAG implementations
- **research_assistant.py**: Complete Research Literature Assistant implementation
- **evaluation.py**: RAG evaluation frameworks and metrics

## 🛠️ Key Components

### Advanced Retrieval Strategies
- Hybrid search (semantic + keyword)
- Multi-index retrieval
- Parent document retrieval
- Contextual compression

### Query Transformation
- Query expansion
- LLM-based query reformulation
- Multi-query retrieval
- Hypothetical Document Embeddings (HyDE)

### Reranking Systems
- Cross-encoder rerankers
- Reciprocal rank fusion
- Maximal marginal relevance
- Source attribution

### Adaptive RAG
- Self-querying retrieval
- Query routing
- Controlled generation (C-RAG)
- Multi-hop reasoning

### Research Assistant
- Academic paper processing
- Citation tracking
- Research question analysis
- Literature review generation

## 🔗 LCEL Implementation

All components are implemented using LangChain Expression Language (LCEL), which provides:
- Improved readability through the pipe operator (`|`)
- Better composability of components
- Enhanced debugging capabilities
- Native support for streaming
- Automatic parallelization of independent operations

## 🧪 Usage Examples

Each file contains detailed examples of how to use the implemented components. Here's a simple example of using the hybrid retriever:

```python
from advanced_retrieval import HybridRetriever
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create hybrid retriever
hybrid_retriever = HybridRetriever(
    documents=documents,
    embedding_model=embedding_model,
    keyword_weight=0.3
)

# Retrieve documents
results = hybrid_retriever.retrieve("quantum computing applications")
```

## 🛠️ Extending the Code

Here are some ideas for extending or customizing the implementations:

1. Add more specialized retrievers for specific domains
2. Implement additional query transformation techniques
3. Create domain-specific rerankers
4. Build custom evaluation metrics for your specific use case
5. Extend the Research Assistant with domain-specific capabilities

## 📚 Related Resources

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangChain Expression Language Guide](https://python.langchain.com/docs/expression_language/)
- [LCEL Cookbook](https://python.langchain.com/docs/expression_language/cookbook/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/en/latest/)
- [Hugging Face Rerankers](https://huggingface.co/models?pipeline_tag=text-to-text-generation&sort=downloads)
