# 🔍 Module 5: Advanced RAG Systems - Lesson 1: Advanced Retrieval Strategies 🚀

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧩 Understand the limitations of basic vector search
- 🔄 Implement hybrid search combining semantic and keyword approaches
- 📚 Build multi-index retrieval systems for diverse content
- 🌳 Create parent document retrieval mechanisms
- 📏 Develop contextual compression techniques
- 🔗 Implement these strategies using LCEL

---

## 🧩 Beyond Basic Vector Search

<img src="https://media.giphy.com/media/l0HlQXlQ3nHyLMvte/giphy.gif" width="50%" height="50%"/>

### The Limitations of Simple Vector Search

While basic vector search (as covered in Module 4) provides a solid foundation for retrieval, it has several limitations:

1. **Semantic-Only Retrieval**: Pure vector search relies entirely on semantic similarity, missing exact keyword matches
2. **Fixed Context Window**: Retrieved chunks have a fixed size, regardless of relevance
3. **Chunk Isolation**: Each chunk is treated independently, losing document-level context
4. **One-Size-Fits-All**: The same retrieval strategy is used for all queries, regardless of type
5. **Limited Information**: Only the content of chunks is considered, ignoring metadata

### Advanced Retrieval Landscape

To overcome these limitations, we can implement several advanced retrieval strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| Hybrid Search | Combines semantic and keyword search | Queries with specific terms |
| Multi-Index Retrieval | Uses multiple vector indices | Diverse content types |
| Parent Document Retrieval | Retrieves parent documents of chunks | Maintaining broader context |
| Contextual Compression | Filters irrelevant content from chunks | Maximizing relevant context |
| Self-Querying Retrieval | Generates structured filters from queries | Metadata-aware queries |

---

## 🔄 Hybrid Search: Combining Semantic and Keyword Approaches

Hybrid search combines the strengths of semantic (vector) search with traditional keyword search:

### How Hybrid Search Works

1. **Semantic Search**: Find documents similar in meaning using embeddings
2. **Keyword Search**: Find documents containing exact query terms
3. **Result Fusion**: Combine results using weighted scoring or rank fusion

### Implementing Hybrid Search with LCEL

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough

# Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store and retriever
vectorstore = FAISS.from_documents(documents, embedding_model)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create keyword retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# Create ensemble retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # Adjust weights based on your use case
)

# Create LCEL chain
hybrid_chain = (
    {"query": RunnablePassthrough()}
    | hybrid_retriever
)
```

### Custom Hybrid Search Implementation

For more control, you can implement a custom hybrid search:

```python
def hybrid_search(query, documents, embedding_model, keyword_weight=0.3):
    # Semantic search
    query_embedding = embedding_model.embed_query(query)
    semantic_results = semantic_search(query_embedding, documents)
    
    # Keyword search
    keyword_results = keyword_search(query, documents)
    
    # Combine results
    combined_results = {}
    
    # Add semantic search results
    for doc, score in semantic_results:
        combined_results[doc.id] = (1 - keyword_weight) * score
    
    # Add keyword search results
    for doc, score in keyword_results:
        if doc.id in combined_results:
            combined_results[doc.id] += keyword_weight * score
        else:
            combined_results[doc.id] = keyword_weight * score
    
    # Sort by combined score
    sorted_results = sorted(
        combined_results.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Return documents
    return [doc_id for doc_id, _ in sorted_results]
```

---

## 📚 Multi-Index Retrieval: Specialized Indices for Different Content

Multi-index retrieval uses multiple specialized indices for different types of content:

### Why Use Multiple Indices?

1. **Content Diversity**: Different content types benefit from different embedding models
2. **Specialized Retrieval**: Use specialized retrievers for different document types
3. **Performance**: Smaller, focused indices can be more efficient
4. **Flexibility**: Add or remove indices without rebuilding the entire system

### Implementing Multi-Index Retrieval with LCEL

```python
from langchain.retrievers import MergerRetriever
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

# Create different embedding models
general_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
specialized_embeddings = OpenAIEmbeddings()  # Or another domain-specific model

# Create different vector stores for different document types
technical_docs = [doc for doc in documents if doc.metadata.get("type") == "technical"]
general_docs = [doc for doc in documents if doc.metadata.get("type") == "general"]

technical_vectorstore = FAISS.from_documents(technical_docs, specialized_embeddings)
general_vectorstore = Chroma.from_documents(general_docs, general_embeddings)

# Create retrievers
technical_retriever = technical_vectorstore.as_retriever(search_kwargs={"k": 3})
general_retriever = general_vectorstore.as_retriever(search_kwargs={"k": 3})

# Create merger retriever
multi_index_retriever = MergerRetriever(
    retrievers=[technical_retriever, general_retriever]
)

# Create LCEL chain
multi_index_chain = (
    {"query": RunnablePassthrough()}
    | multi_index_retriever
)
```

---

## 🌳 Parent Document Retrieval: Maintaining Document Context

Parent document retrieval helps maintain broader context by retrieving the parent documents of matching chunks:

### The Parent Document Approach

1. **Chunk-Level Search**: First, find the most relevant chunks
2. **Parent Mapping**: Map chunks back to their parent documents
3. **Parent Retrieval**: Retrieve the full parent documents or larger sections
4. **Context Preservation**: Maintain the broader context of the information

### Implementing Parent Document Retrieval with LCEL

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# Create text splitters for different granularities
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Create vector store
vectorstore = FAISS.from_documents([], embedding_model)

# Create document store for parent documents
doc_store = InMemoryStore()

# Create parent document retriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=doc_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents
parent_retriever.add_documents(documents)

# Create LCEL chain
parent_chain = (
    {"query": RunnablePassthrough()}
    | parent_retriever
)
```

---

## 📏 Contextual Compression: Maximizing Relevant Context

Contextual compression filters irrelevant content from retrieved chunks to maximize the relevant context:

### How Contextual Compression Works

1. **Initial Retrieval**: Retrieve documents using standard methods
2. **Relevance Filtering**: Filter out irrelevant parts of each document
3. **Compression**: Compress the remaining content to focus on the most relevant information
4. **Context Optimization**: Maximize the amount of relevant context within the context window

### Implementing Contextual Compression with LCEL

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatGroq

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create LLM for compression
llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")

# Create document compressor
compressor = LLMChainExtractor.from_llm(llm)

# Create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Create LCEL chain
compression_chain = (
    {"query": RunnablePassthrough()}
    | compression_retriever
)
```

---

## 🔗 Combining Advanced Retrieval Strategies with LCEL

One of the strengths of LCEL is the ability to combine multiple retrieval strategies:

### Creating a Multi-Strategy Retriever

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Define a function to route queries to different retrievers
def route_query(query):
    if "technical" in query.lower():
        return "technical"
    elif "general" in query.lower():
        return "general"
    else:
        return "hybrid"

# Create a dictionary of retrievers
retrievers = {
    "technical": technical_retriever,
    "general": general_retriever,
    "hybrid": hybrid_retriever
}

# Create a routing chain
router_chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(lambda x: route_query(x["query"]))
)

# Create the final retrieval chain
retrieval_chain = (
    {"query": RunnablePassthrough(), "retriever": router_chain}
    | RunnableLambda(lambda x: retrievers[x["retriever"]].get_relevant_documents(x["query"]))
)
```

---

## 💪 Practice Exercises

1. **Implement a Hybrid Search System**: Create a hybrid search system that combines semantic and keyword search with adjustable weights.

2. **Build a Multi-Index Retriever**: Implement a multi-index retriever that uses different embedding models for different document types.

3. **Create a Parent Document Retrieval System**: Build a parent document retrieval system that maintains document context while still finding relevant information.

4. **Develop a Contextual Compression System**: Implement a contextual compression system that filters irrelevant content from retrieved chunks.

5. **Combine Multiple Retrieval Strategies**: Create an advanced retrieval system that combines multiple strategies based on query type.

---

## 🔍 Key Takeaways

1. **Beyond Vector Search**: Advanced retrieval goes beyond simple vector search to address its limitations.

2. **Hybrid Approaches**: Combining semantic and keyword search provides more robust retrieval.

3. **Multiple Indices**: Using specialized indices for different content types improves retrieval quality.

4. **Context Preservation**: Parent document retrieval helps maintain broader context.

5. **Relevance Filtering**: Contextual compression maximizes the amount of relevant context.

6. **LCEL Integration**: All these strategies can be elegantly implemented and combined using LCEL.

---

## 📚 Resources

- [LangChain Retrievers Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [BM25 Algorithm Explanation](https://en.wikipedia.org/wiki/Okapi_BM25)
- [LangChain Contextual Compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/)
- [LangChain Parent Document Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
- [LCEL Retriever Patterns](https://python.langchain.com/docs/expression_language/cookbook/retrieval)

---

## 🚀 Next Steps

In the next lesson, we'll explore query transformation techniques that can further improve retrieval quality by reformulating and expanding queries before retrieval.
