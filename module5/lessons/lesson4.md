# 🧠 Module 5: Advanced RAG Systems - Lesson 4: Self-Querying and Adaptive RAG 🚀

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔍 Understand how to implement self-querying retrieval for metadata filtering
- 🧩 Build query-dependent retrieval strategies
- 🔄 Create adaptive RAG systems that modify approaches based on query type
- 📊 Implement query routing for specialized retrievers
- 🔗 Integrate these techniques using LCEL

---

## 🧠 Beyond Static Retrieval: The Need for Adaptivity

<img src="https://media.giphy.com/media/l0HlQXlQ3nHyLMvte/giphy.gif" width="50%" height="50%"/>

### Why One-Size-Fits-All Retrieval Falls Short

Traditional RAG systems use the same retrieval strategy for all queries, but different queries have different needs:

1. **Varying Information Needs**: Some queries need precise facts, others need broad context
2. **Metadata Constraints**: Users often want to filter by source, date, author, etc.
3. **Query Complexity**: Simple vs. complex queries benefit from different approaches
4. **Domain Specificity**: Domain-specific queries may need specialized retrievers
5. **User Context**: User preferences and history should influence retrieval

### The Adaptive RAG Landscape

To address these limitations, we can implement several adaptive RAG strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| Self-Querying Retrieval | Extracts metadata filters from natural language | Queries with implicit filters |
| Query Classification | Categorizes queries to route to specialized retrievers | Diverse query types |
| Multi-Strategy Retrieval | Applies different strategies based on query type | Complex information needs |
| Contextual Routing | Routes queries based on user context and history | Personalized retrieval |
| Feedback-Based Adaptation | Adjusts strategies based on user feedback | Iterative search sessions |

---

## 🔍 Self-Querying Retrieval: Natural Language Filters

Self-querying retrieval extracts metadata filters from natural language queries:

### How Self-Querying Retrieval Works

1. **Query Analysis**: Parse the query to identify filter conditions
2. **Metadata Extraction**: Extract metadata constraints from the query
3. **Filter Generation**: Generate structured filters for the vector store
4. **Filtered Retrieval**: Retrieve documents matching both semantic similarity and filters

### Implementing Self-Querying Retrieval with LCEL

```python
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatGroq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough

# Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = Chroma.from_documents(documents, embedding_model)

# Define metadata schema
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source of the document, e.g., 'RAG Paper', 'Vector DB Guide'",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="The author of the document",
        type="string",
    ),
    AttributeInfo(
        name="date",
        description="The publication date of the document in YYYY-MM-DD format",
        type="string",
    ),
    AttributeInfo(
        name="topic",
        description="The main topic of the document",
        type="string",
    ),
]

# Create LLM for query construction
llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")

# Create self-query retriever
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Academic papers and technical documents about RAG systems",
    metadata_field_info=metadata_field_info,
    verbose=True
)

# Create LCEL chain
self_query_chain = (
    {"query": RunnablePassthrough()}
    | self_query_retriever
)
```

### Example Queries and Generated Filters

| Natural Language Query | Generated Filter |
|------------------------|------------------|
| "Papers about RAG from 2023" | `{"source": {"$eq": "RAG Paper"}, "date": {"$gte": "2023-01-01"}}` |
| "Documents by Meta AI about retrieval" | `{"author": {"$eq": "Meta AI"}, "topic": {"$eq": "retrieval"}}` |
| "Recent papers about vector databases" | `{"topic": {"$eq": "vector databases"}, "date": {"$gte": "2022-01-01"}}` |

---

## 🧩 Query Classification and Routing

Query classification routes queries to specialized retrievers:

### How Query Classification Works

1. **Query Analysis**: Analyze the query to determine its type and intent
2. **Classification**: Categorize the query into predefined types
3. **Retriever Selection**: Select the most appropriate retriever for the query type
4. **Specialized Retrieval**: Apply the selected retrieval strategy

### Implementing Query Classification with LCEL

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq

# Create LLM for classification
llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")

# Create classification prompt
classification_prompt = ChatPromptTemplate.from_template("""
Classify the following query into one of these categories:
- factual: Seeking specific facts or information
- conceptual: Seeking explanation of concepts
- procedural: Asking how to do something
- comparative: Comparing multiple things
- exploratory: Broad exploration of a topic

Query: {query}

Category:
""")

# Create classification chain
classification_chain = classification_prompt | llm | RunnableLambda(lambda x: x.content.strip().lower())

# Create specialized retrievers
factual_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

conceptual_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

procedural_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.8}
)

comparative_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20}
)

exploratory_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5}  # More diversity
)

# Create retriever mapping
retrievers = {
    "factual": factual_retriever,
    "conceptual": conceptual_retriever,
    "procedural": procedural_retriever,
    "comparative": comparative_retriever,
    "exploratory": exploratory_retriever
}

# Create routing chain
routing_chain = (
    {"query": RunnablePassthrough(), "category": RunnableLambda(lambda x: classification_chain.invoke(x))}
    | RunnableLambda(lambda x: retrievers[x["category"]].get_relevant_documents(x["query"]))
)
```

---

## 🔄 Multi-Strategy Retrieval

Multi-strategy retrieval applies different retrieval strategies based on query characteristics:

### How Multi-Strategy Retrieval Works

1. **Strategy Selection**: Determine which retrieval strategies to apply
2. **Parallel Execution**: Execute multiple retrieval strategies in parallel
3. **Result Combination**: Combine results using fusion techniques
4. **Adaptive Weighting**: Adjust strategy weights based on query type

### Implementing Multi-Strategy Retrieval with LCEL

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Create base retrievers
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
keyword_retriever = BM25Retriever.from_documents(documents, k=10)

# Create compression retriever
llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=semantic_retriever
)

# Create ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.7, 0.3]
)

# Create strategy selector
def select_strategy(query):
    query_lower = query.lower()
    if "explain" in query_lower or "what is" in query_lower:
        return "compression"  # For explanatory queries
    elif any(term in query_lower for term in ["find", "search", "locate"]):
        return "semantic"  # For search queries
    elif any(term in query_lower for term in ["compare", "difference", "versus"]):
        return "ensemble"  # For comparative queries
    else:
        return "semantic"  # Default

# Create strategy mapping
strategies = {
    "semantic": semantic_retriever,
    "keyword": keyword_retriever,
    "compression": compression_retriever,
    "ensemble": ensemble_retriever
}

# Create multi-strategy chain
multi_strategy_chain = (
    {"query": RunnablePassthrough(), "strategy": RunnableLambda(lambda x: select_strategy(x))}
    | RunnableLambda(lambda x: strategies[x["strategy"]].get_relevant_documents(x["query"]))
)
```

---

## 📊 Adaptive RAG: Putting It All Together

Adaptive RAG combines multiple adaptive techniques for a comprehensive system:

### Components of an Adaptive RAG System

1. **Query Understanding**: Analyze and classify the query
2. **Strategy Selection**: Choose appropriate retrieval strategies
3. **Metadata Filtering**: Apply self-querying for metadata constraints
4. **Dynamic Execution**: Execute the selected strategies
5. **Result Optimization**: Apply appropriate reranking techniques
6. **Feedback Integration**: Learn from user feedback

### Implementing Adaptive RAG with LCEL

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq

# Create LLM
llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")

# Create query analyzer
analyzer_prompt = ChatPromptTemplate.from_template("""
Analyze the following query and extract:
1. Query type (factual, conceptual, procedural, comparative, exploratory)
2. Metadata filters (source, author, date, topic)
3. Complexity level (simple, moderate, complex)

Query: {query}

Analysis:
""")

analyzer_chain = analyzer_prompt | llm | RunnableLambda(lambda x: parse_analysis(x.content))

def parse_analysis(analysis):
    """Parse the analysis output into structured data."""
    # Simple parsing logic (in practice, use more robust parsing)
    lines = analysis.strip().split("\n")
    result = {}
    
    for line in lines:
        if "Query type:" in line:
            result["query_type"] = line.split("Query type:")[1].strip().lower()
        elif "Metadata filters:" in line:
            filters_text = line.split("Metadata filters:")[1].strip()
            result["metadata_filters"] = parse_filters(filters_text)
        elif "Complexity level:" in line:
            result["complexity"] = line.split("Complexity level:")[1].strip().lower()
    
    return result

def parse_filters(filters_text):
    """Parse metadata filters text into structured filters."""
    # Simple parsing logic (in practice, use more robust parsing)
    filters = {}
    
    if "source:" in filters_text:
        source = filters_text.split("source:")[1].split(",")[0].strip()
        filters["source"] = source
    
    if "author:" in filters_text:
        author = filters_text.split("author:")[1].split(",")[0].strip()
        filters["author"] = author
    
    if "date:" in filters_text:
        date = filters_text.split("date:")[1].split(",")[0].strip()
        filters["date"] = date
    
    if "topic:" in filters_text:
        topic = filters_text.split("topic:")[1].split(",")[0].strip()
        filters["topic"] = topic
    
    return filters

# Create strategy selector
def select_retrieval_strategy(analysis):
    """Select retrieval strategy based on query analysis."""
    query_type = analysis.get("query_type", "factual")
    complexity = analysis.get("complexity", "simple")
    
    if query_type == "factual" and complexity == "simple":
        return "semantic"
    elif query_type == "conceptual":
        return "compression"
    elif query_type == "comparative":
        return "ensemble"
    elif query_type == "exploratory":
        return "mmr"
    elif complexity == "complex":
        return "multi_query"
    else:
        return "semantic"

# Create the adaptive RAG chain
adaptive_rag_chain = (
    {"query": RunnablePassthrough(), "analysis": RunnableLambda(lambda x: analyzer_chain.invoke(x))}
    | RunnableLambda(lambda x: {
        "query": x["query"],
        "strategy": select_retrieval_strategy(x["analysis"]),
        "filters": x["analysis"].get("metadata_filters", {})
    })
    | RunnableLambda(lambda x: retrieve_with_strategy(x["query"], x["strategy"], x["filters"]))
)

def retrieve_with_strategy(query, strategy, filters):
    """Retrieve documents using the selected strategy and filters."""
    # Apply metadata filters if available
    filtered_vectorstore = apply_filters(vectorstore, filters)
    
    # Get the appropriate retriever
    retriever = strategies[strategy]
    
    # Retrieve documents
    documents = retriever.get_relevant_documents(query)
    
    return documents
```

---

## 💪 Practice Exercises

1. **Implement Self-Querying Retrieval**: Create a self-querying retriever that extracts metadata filters from natural language queries.

2. **Build a Query Classification System**: Implement a query classifier that categorizes queries into different types.

3. **Create a Multi-Strategy Retrieval System**: Build a system that applies different retrieval strategies based on query characteristics.

4. **Develop a Query Router**: Implement a router that directs queries to specialized retrievers.

5. **Build an Adaptive RAG System**: Create a complete adaptive RAG system that combines multiple techniques.

6. **Implement Feedback Integration**: Add a mechanism to learn from user feedback to improve retrieval.

---

## 🔍 Key Takeaways

1. **Beyond Static Retrieval**: Adaptive RAG dynamically adjusts strategies based on query characteristics.

2. **Metadata Awareness**: Self-querying enables natural language filtering of metadata.

3. **Query Understanding**: Classification helps route queries to specialized retrievers.

4. **Strategy Selection**: Different query types benefit from different retrieval approaches.

5. **Comprehensive Adaptation**: Combining multiple adaptive techniques creates robust systems.

6. **LCEL Integration**: All these strategies can be elegantly implemented and combined using LCEL.

---

## 📚 Resources

- [LangChain Self-Query Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/)
- [Query Classification Techniques](https://arxiv.org/abs/2103.05633)
- [Adaptive Information Retrieval](https://dl.acm.org/doi/10.1145/3397271.3401419)
- [LCEL Routing Patterns](https://python.langchain.com/docs/expression_language/cookbook/router)
- [Metadata Filtering in Vector Databases](https://www.pinecone.io/learn/metadata-filtering/)

---

## 🚀 Next Steps

In the next lesson, we'll explore how to build a complete Research Literature Assistant that combines all the advanced RAG techniques we've learned to create a powerful tool for processing and querying academic papers.
