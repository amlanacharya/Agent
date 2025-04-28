# 🔗 Lesson 5.5: LangChain Expression Language (LCEL) - The Modern Way to Build AI Chains

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
- Understand the core concepts of LangChain Expression Language (LCEL)
- Build chains using the pipe operator (`|`) for improved readability
- Implement functional programming patterns in AI applications
- Create reusable components for different chain architectures
- Refactor traditional chains to use LCEL syntax
- Implement advanced patterns like branching logic and parallel processing

## 📚 Introduction

LangChain Expression Language (LCEL) represents a paradigm shift in how we build AI applications. It introduces a declarative, functional approach to composing chains that makes your code more readable, maintainable, and efficient.

> 💡 **Note**: From Module 5 onwards, we'll be using LCEL as our primary approach for building chains. This lesson serves as a bridge to prepare you for that transition.

## 🧩 Core Concepts of LCEL

### The Pipe Operator (`|`)

At the heart of LCEL is the pipe operator (`|`), which allows you to compose components in a linear, readable fashion. This is similar to the pipe operator in Unix/Linux or functional programming languages.

```python
# Traditional approach (nested, hard to read)
final_result = component3(component2(component1(input_data)))

# LCEL approach (linear, easy to read)
chain = component1 | component2 | component3
final_result = chain.invoke(input_data)
```

### Runnables

LCEL is built around the concept of "runnables" - components that can process inputs and produce outputs. All LangChain components (prompts, LLMs, retrievers, etc.) implement the Runnable interface, making them compatible with LCEL.

Key runnable types include:
- `RunnablePassthrough`: Passes inputs directly through
- `RunnableLambda`: Applies a custom function to inputs
- `RunnableMap`: Processes inputs through multiple parallel paths

## 🔄 Traditional vs. LCEL Approach

Let's compare the traditional approach with LCEL for a simple RAG system:

### Traditional Approach

```python
def process_query(query, retriever, llm):
    # Retrieve documents
    documents = retriever.get_relevant_documents(query)
    
    # Format documents into context
    context = "\n\n".join(doc.page_content for doc in documents)
    
    # Create prompt
    prompt = f"""
    Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Generate answer
    response = llm.invoke(prompt)
    
    return response
```

### LCEL Approach

```python
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context:

Context:
{context}

Question: {question}

Answer:
""")

# Define document formatting function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chain with the pipe operator
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Use the chain
response = rag_chain.invoke(query)
```

## 🛠️ Building Blocks of LCEL

### 1. Input Processing

LCEL provides several ways to process inputs:

```python
# Pass input directly
chain1 = RunnablePassthrough() | llm

# Transform input with a function
chain2 = RunnableLambda(lambda x: f"Question: {x}") | llm

# Create a structured input
chain3 = (
    {"question": RunnablePassthrough(), "context": retriever | format_docs}
    | prompt
    | llm
)
```

### 2. Branching Logic

LCEL allows for conditional processing based on input:

```python
def route_by_type(input_text):
    if "how to" in input_text.lower():
        return "procedural"
    elif any(word in input_text.lower() for word in ["opinion", "think", "feel"]):
        return "opinion"
    else:
        return "factual"

factual_chain = factual_prompt | llm
opinion_chain = opinion_prompt | llm
procedural_chain = procedural_prompt | llm

def router(input_text):
    question_type = route_by_type(input_text)
    if question_type == "factual":
        return factual_chain.invoke(input_text)
    elif question_type == "opinion":
        return opinion_chain.invoke(input_text)
    else:
        return procedural_chain.invoke(input_text)

chain = RunnableLambda(router)
```

### 3. Parallel Processing

LCEL can execute multiple operations in parallel:

```python
from langchain.schema.runnable import RunnableParallel

# Run two retrievers in parallel
parallel_retriever = RunnableParallel(
    semantic=semantic_retriever,
    keyword=keyword_retriever
)

# Process results
chain = (
    parallel_retriever
    | RunnableLambda(lambda x: merge_results(x["semantic"], x["keyword"]))
    | format_docs
    | prompt
    | llm
)
```

## 🚀 Complete LCEL RAG Example

Here's a complete example of a RAG system using LCEL:

```python
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# 1. Load and process documents
loader = TextLoader("data.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

# 2. Create embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Create LLM
llm = ChatGroq(temperature=0, model_name="llama2-70b-4096")

# 4. Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

Context:
{context}

Question: {question}

Answer:
""")

# 5. Define document formatting function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Create the RAG chain with LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 7. Use the chain
query = "What is RAG?"
response = rag_chain.invoke(query)
print(response.content)
```

## 🔍 Advanced LCEL Patterns

### Memory Integration

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

def load_memory(_):
    return memory.load_memory_variables({})

def format_history(memory_variables):
    messages = memory_variables.get("history", [])
    return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

memory_chain = (
    RunnableLambda(load_memory)
    | RunnableLambda(format_history)
)

chain = (
    {
        "history": memory_chain,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)
```

### Streaming Responses

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create a streaming chain
streaming_chain = (
    prompt
    | llm.with_config({"callbacks": [StreamingStdOutCallbackHandler()]})
)

# Use the streaming chain
streaming_chain.invoke({"context": context, "question": query})
```

### Custom Runnables

```python
from langchain.schema.runnable import Runnable

class CustomRetriever(Runnable):
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def invoke(self, query, config=None):
        # Custom retrieval logic
        results = self.vectorstore.similarity_search(query, k=5)
        
        # Post-processing
        for doc in results:
            doc.metadata["score"] = 0.95  # Add custom metadata
            
        return results

custom_retriever = CustomRetriever(vectorstore)
chain = custom_retriever | format_docs | prompt | llm
```

## 💪 Practice Exercises

1. **Basic LCEL Chain**: Create a simple chain that takes a question and generates an answer using a prompt template and LLM.

2. **RAG with LCEL**: Implement a RAG system using LCEL that retrieves documents and generates answers with source attribution.

3. **Branching Logic**: Create a chain that routes questions to different sub-chains based on the question type.

4. **Memory Integration**: Build a chain that maintains conversation history and uses it to provide context-aware responses.

5. **Parallel Retrievers**: Implement a chain that uses multiple retrievers in parallel and combines their results.

## 🔑 Key Takeaways

1. **Readability**: LCEL makes chains more readable by using the pipe operator to show data flow clearly.

2. **Composability**: Components can be easily combined, reused, and rearranged.

3. **Functional Approach**: LCEL encourages a functional programming style that's easier to test and debug.

4. **Parallelism**: Operations can run in parallel for better performance.

5. **Streaming**: Native support for streaming responses.

6. **Maintainability**: Chains are easier to maintain and extend over time.

## 📚 Resources

- [LangChain Expression Language Guide](https://python.langchain.com/docs/expression_language/)
- [LCEL Cookbook](https://python.langchain.com/docs/expression_language/cookbook/)
- [Building RAG with LCEL](https://python.langchain.com/docs/use_cases/question_answering/quickstart)
- [LCEL Design Patterns](https://python.langchain.com/docs/expression_language/how_to/)
- [Streaming with LCEL](https://python.langchain.com/docs/expression_language/streaming)

## 🚀 Next Steps

In Module 5, we'll dive deeper into advanced RAG techniques using LCEL, including:
- Hybrid search strategies
- Contextual compression
- Reranking mechanisms
- Self-querying retrieval
- Multi-hop reasoning
- Evaluation frameworks

These techniques will help you build even more powerful and effective knowledge-based AI applications using modern LangChain patterns.
