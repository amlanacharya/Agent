# 🔄 Module 5: Advanced RAG Systems - Lesson 2: Query Transformation Techniques 🚀

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🧠 Understand why query transformation is critical for effective retrieval
- 🔄 Implement query expansion techniques to broaden search scope
- 🔍 Build LLM-based query reformulation systems
- 🧩 Create multi-query retrieval for improved recall
- 🔮 Develop hypothetical document embeddings (HyDE)
- 🔙 Implement step-back prompting for complex queries
- 🔗 Integrate these techniques using LCEL

---

## 🧠 The Query Problem in RAG Systems

<img src="https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif" width="50%" height="50%"/>

### Why Queries Often Fail

Even with advanced retrieval strategies, RAG systems often struggle because:

1. **Vocabulary Mismatch**: Users use different terms than those in the documents
2. **Ambiguity**: Queries can have multiple interpretations
3. **Incompleteness**: Queries may lack important context
4. **Complexity**: Some questions require multi-step reasoning
5. **Abstraction Level**: Query may be more abstract than document content

### The Query Transformation Solution

Query transformation techniques address these issues by:

1. **Expanding Queries**: Adding synonyms and related terms
2. **Reformulating Queries**: Rephrasing for better matching
3. **Generating Multiple Queries**: Exploring different aspects
4. **Creating Hypothetical Answers**: Bridging the semantic gap
5. **Stepping Back**: Asking more general questions first

---

## 🔄 Query Expansion: Broadening the Search Scope

Query expansion adds related terms to the original query to improve recall:

### Types of Query Expansion

1. **Synonym Expansion**: Adding synonyms of key terms
2. **Entity Expansion**: Adding related entities
3. **Concept Expansion**: Adding broader or narrower concepts
4. **Domain-Specific Expansion**: Adding domain-specific related terms

### Implementing Query Expansion with LCEL

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatGroq

# Create LLM for expansion
llm = ChatGroq(temperature=0.2, model_name="llama2-70b-4096")

# Create expansion prompt
expansion_prompt = """
Expand the following search query by adding synonyms and related terms.
Format the output as a comma-separated list of terms.

Original query: {query}

Expanded terms:
"""

# Create expansion function
def expand_query(query):
    # Get expansion from LLM
    response = llm.invoke(expansion_prompt.format(query=query))
    expanded_terms = response.content.strip().split(", ")
    
    # Combine original query with expanded terms
    expanded_query = f"{query} {' '.join(expanded_terms)}"
    return expanded_query

# Create LCEL chain
expansion_chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(lambda x: expand_query(x["query"]))
    | RunnableLambda(lambda expanded_query: {"expanded_query": expanded_query, "original_query": expanded_query.split(" ")[0]})
)
```

### Simple Keyword-Based Expansion

For a simpler approach without an LLM, you can use predefined synonyms:

```python
import nltk
from nltk.corpus import wordnet

# Download WordNet if not already downloaded
nltk.download('wordnet')

def expand_query_with_wordnet(query):
    # Tokenize query
    tokens = query.lower().split()
    expanded_tokens = tokens.copy()
    
    # Add synonyms for each token
    for token in tokens:
        # Get synonyms from WordNet
        synsets = wordnet.synsets(token)
        
        # Add synonyms to expanded tokens
        for synset in synsets[:2]:  # Limit to top 2 synsets
            for lemma in synset.lemmas()[:3]:  # Limit to top 3 lemmas
                synonym = lemma.name().replace('_', ' ')
                if synonym != token and synonym not in expanded_tokens:
                    expanded_tokens.append(synonym)
    
    # Combine into expanded query
    expanded_query = ' '.join(expanded_tokens)
    return expanded_query
```

---

## 🔍 LLM-Based Query Reformulation

LLM-based query reformulation uses language models to rephrase queries for better retrieval:

### How Query Reformulation Works

1. **Query Analysis**: Analyze the original query to understand intent
2. **Knowledge Injection**: Add domain knowledge to the query
3. **Reformulation**: Rephrase the query to better match document language
4. **Specificity Adjustment**: Make vague queries more specific

### Implementing Query Reformulation with LCEL

```python
from langchain.prompts import ChatPromptTemplate

# Create reformulation prompt
reformulation_prompt = ChatPromptTemplate.from_template("""
You are an expert at reformulating search queries to improve retrieval results.
Your task is to reformulate the following query to make it more effective for retrieving relevant information.

Consider:
1. Using more specific terminology
2. Adding context if the query is ambiguous
3. Breaking down complex queries
4. Using domain-specific language

Original query: {query}

Reformulated query:
""")

# Create LCEL chain
reformulation_chain = (
    {"query": RunnablePassthrough()}
    | reformulation_prompt
    | llm
    | RunnableLambda(lambda x: x.content.strip())
)
```

### Domain-Specific Reformulation

For domain-specific applications, you can customize the reformulation:

```python
# Create domain-specific reformulation prompt
medical_reformulation_prompt = ChatPromptTemplate.from_template("""
You are a medical search expert. Reformulate the following query to use proper medical terminology
and make it more effective for retrieving relevant medical information.

Original query: {query}

Reformulated query:
""")

# Create LCEL chain for medical queries
medical_reformulation_chain = (
    {"query": RunnablePassthrough()}
    | medical_reformulation_prompt
    | llm
    | RunnableLambda(lambda x: x.content.strip())
)
```

---

## 🧩 Multi-Query Retrieval: Exploring Different Aspects

Multi-query retrieval generates multiple variations of a query to improve recall:

### How Multi-Query Retrieval Works

1. **Query Generation**: Generate multiple variations of the original query
2. **Parallel Retrieval**: Run each query variation through the retriever
3. **Result Fusion**: Combine and deduplicate results from all queries
4. **Reranking**: Rerank the combined results

### Implementing Multi-Query Retrieval with LCEL

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# Create multi-query retriever
def create_multi_query_retriever(base_retriever, llm):
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    return multi_query_retriever

# Create LCEL chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

multi_query_chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(lambda x: create_multi_query_retriever(base_retriever, llm).get_relevant_documents(x["query"]))
    | RunnableLambda(format_docs)
)
```

### Custom Multi-Query Implementation

For more control, you can implement a custom multi-query system:

```python
def generate_query_variations(query, llm, num_variations=3):
    # Create prompt for generating variations
    prompt = f"""
    Generate {num_variations} different variations of the following query.
    Each variation should explore a different aspect or use different terminology.
    Format each variation on a new line.
    
    Original query: {query}
    
    Variations:
    """
    
    # Get variations from LLM
    response = llm.invoke(prompt)
    variations = response.content.strip().split("\n")
    
    # Clean up variations
    cleaned_variations = [var.strip() for var in variations if var.strip()]
    
    # Add original query
    all_queries = [query] + cleaned_variations
    
    return all_queries

def multi_query_retrieval(query, retriever, llm, num_variations=3):
    # Generate query variations
    query_variations = generate_query_variations(query, llm, num_variations)
    
    # Retrieve documents for each variation
    all_docs = []
    for query_var in query_variations:
        docs = retriever.get_relevant_documents(query_var)
        all_docs.extend(docs)
    
    # Deduplicate documents
    unique_docs = []
    seen_contents = set()
    
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs
```

---

## 🔮 Hypothetical Document Embeddings (HyDE)

HyDE uses an LLM to generate a hypothetical document that would answer the query, then uses that document's embedding for retrieval:

### How HyDE Works

1. **Hypothetical Document Generation**: Generate a document that would answer the query
2. **Document Embedding**: Embed the hypothetical document
3. **Similarity Search**: Find real documents similar to the hypothetical one
4. **Result Retrieval**: Return the most similar real documents

### Implementing HyDE with LCEL

```python
from langchain.retrievers import HydeRetriever

# Create HyDE retriever
hyde_retriever = HydeRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    prompt_template="""
    Write a passage that answers the question.
    Question: {question}
    Passage:
    """
)

# Create LCEL chain
hyde_chain = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(lambda x: hyde_retriever.get_relevant_documents(x["question"]))
    | RunnableLambda(format_docs)
)
```

### Custom HyDE Implementation

For more control, you can implement a custom HyDE system:

```python
def generate_hypothetical_document(query, llm):
    # Create prompt for generating hypothetical document
    prompt = f"""
    Write a detailed passage that directly answers the following question.
    Make the passage informative and comprehensive, as if it were extracted from a document.
    
    Question: {query}
    
    Passage:
    """
    
    # Get hypothetical document from LLM
    response = llm.invoke(prompt)
    return response.content.strip()

def hyde_retrieval(query, vectorstore, llm, embedding_model):
    # Generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(query, llm)
    
    # Embed hypothetical document
    hyde_embedding = embedding_model.embed_query(hypothetical_doc)
    
    # Find similar documents
    similar_docs = vectorstore.similarity_search_by_vector(hyde_embedding, k=5)
    
    return similar_docs
```

---

## 🔙 Step-Back Prompting: From Specific to General

Step-back prompting addresses complex queries by first asking a more general question:

### How Step-Back Prompting Works

1. **Step Back**: Transform specific query into a more general one
2. **General Retrieval**: Retrieve information for the general query
3. **Context Building**: Use general information as context
4. **Specific Retrieval**: Retrieve information for the original query with added context

### Implementing Step-Back Prompting with LCEL

```python
from langchain.prompts import ChatPromptTemplate

# Create step-back prompt
step_back_prompt = ChatPromptTemplate.from_template("""
Given the following specific question, generate a more general question that would help provide context for answering the specific question.

Specific question: {query}

General question:
""")

# Create LCEL chain
def step_back_retrieval(query, retriever, llm):
    # Generate general question
    general_question_chain = (
        {"query": RunnablePassthrough()}
        | step_back_prompt
        | llm
        | RunnableLambda(lambda x: x.content.strip())
    )
    
    general_question = general_question_chain.invoke({"query": query})
    
    # Retrieve documents for general question
    general_docs = retriever.get_relevant_documents(general_question)
    
    # Retrieve documents for specific question
    specific_docs = retriever.get_relevant_documents(query)
    
    # Combine results
    combined_docs = general_docs + specific_docs
    
    # Deduplicate
    unique_docs = []
    seen_contents = set()
    
    for doc in combined_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs

# Create LCEL chain
step_back_chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(lambda x: step_back_retrieval(x["query"], retriever, llm))
    | RunnableLambda(format_docs)
)
```

---

## 🔗 Combining Query Transformation Techniques with LCEL

One of the strengths of LCEL is the ability to combine multiple query transformation techniques:

### Creating a Multi-Strategy Query Transformer

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch

# Define a function to route queries to different transformers
def route_query(query):
    query_lower = query.lower()
    
    if "how" in query_lower or "steps" in query_lower:
        return "step_back"
    elif any(term in query_lower for term in ["technical", "specific", "detailed"]):
        return "hyde"
    elif "compare" in query_lower or "difference" in query_lower:
        return "multi_query"
    else:
        return "reformulation"

# Create a dictionary of transformers
transformers = {
    "reformulation": reformulation_chain,
    "multi_query": lambda x: multi_query_retrieval(x, retriever, llm),
    "hyde": lambda x: hyde_retrieval(x, vectorstore, llm, embedding_model),
    "step_back": lambda x: step_back_retrieval(x, retriever, llm)
}

# Create a routing chain
router_chain = RunnableBranch(
    (lambda x: route_query(x["query"]) == "reformulation", transformers["reformulation"]),
    (lambda x: route_query(x["query"]) == "multi_query", transformers["multi_query"]),
    (lambda x: route_query(x["query"]) == "hyde", transformers["hyde"]),
    (lambda x: route_query(x["query"]) == "step_back", transformers["step_back"]),
    transformers["reformulation"]  # Default
)

# Create the final transformation chain
transformation_chain = (
    {"query": RunnablePassthrough()}
    | router_chain
)
```

---

## 💪 Practice Exercises

1. **Implement Query Expansion**: Create a query expansion system that adds synonyms and related terms to improve retrieval.

2. **Build an LLM-Based Query Reformulator**: Implement a system that uses an LLM to reformulate queries for better matching.

3. **Create a Multi-Query Retriever**: Build a multi-query retrieval system that generates and uses multiple query variations.

4. **Develop a HyDE Implementation**: Implement a Hypothetical Document Embeddings system for improved semantic retrieval.

5. **Implement Step-Back Prompting**: Create a step-back prompting system that handles complex queries by first retrieving general information.

6. **Combine Multiple Techniques**: Build an advanced query transformation system that combines multiple techniques based on query type.

---

## 🔍 Key Takeaways

1. **Beyond Basic Queries**: Query transformation addresses the limitations of basic retrieval by improving the queries themselves.

2. **Expansion Techniques**: Adding synonyms and related terms improves recall for keyword-based retrieval.

3. **LLM Reformulation**: Using LLMs to rephrase queries can bridge the gap between user language and document language.

4. **Multiple Perspectives**: Generating multiple query variations explores different aspects of the same question.

5. **Hypothetical Documents**: HyDE leverages the LLM's knowledge to create a bridge between queries and documents.

6. **Step-Back Approach**: Complex queries benefit from first retrieving general information before specific details.

7. **LCEL Integration**: All these techniques can be elegantly implemented and combined using LCEL.

---

## 📚 Resources

- [LangChain Multi-Query Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_query)
- [HyDE Paper](https://arxiv.org/abs/2212.10496)
- [Step-Back Prompting Paper](https://arxiv.org/abs/2310.06117)
- [Query Expansion Techniques](https://en.wikipedia.org/wiki/Query_expansion)
- [LangChain Query Transformation](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression)
- [LCEL Query Transformation Patterns](https://python.langchain.com/docs/expression_language/cookbook/retrieval)

---

## 🚀 Next Steps

In the next lesson, we'll explore reranking and result optimization techniques that can further improve retrieval quality by reordering and filtering results after the initial retrieval.
