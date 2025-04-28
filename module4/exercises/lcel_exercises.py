"""
LCEL Exercises
-------------
This module contains exercises for practicing with LangChain Expression Language (LCEL).
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import langchain
    from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install with 'pip install langchain' for LCEL functionality.")

# Try to import the GroqClient for LLM integration
try:
    # Try module3 path first
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    from module3.code.groq_client import GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    try:
        # Try module2-llm path
        from module2_llm.code.groq_client import GroqClient
        GROQ_AVAILABLE = True
    except ImportError:
        GROQ_AVAILABLE = False
        logger.warning("GroqClient not available. Some features will be limited.")


class SimpleLLMClient:
    """A simple LLM client for testing."""

    def generate_text(self, prompt, **kwargs):
        """Generate text from a prompt."""
        # This is a very simple simulation
        return {
            "choices": [
                {
                    "message": {
                        "content": f"This is a simulated response to: {prompt[:50]}..."
                    }
                }
            ]
        }

    def extract_text_from_response(self, response):
        """Extract text from a response."""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return response


# Exercise 1: Create a basic LCEL chain
def exercise1_basic_lcel_chain():
    """
    Exercise 1: Create a basic LCEL chain that takes a question, formats it into a prompt,
    and generates an answer using the LLM.

    Returns:
        A callable chain that takes a question and returns an answer
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for LCEL exercises. Install with 'pip install langchain'")

    # Initialize LLM client
    if GROQ_AVAILABLE:
        llm_client = GroqClient()
    else:
        llm_client = SimpleLLMClient()

    # Create a prompt template for answering questions
    prompt = PromptTemplate.from_template("""
    Please answer the following question clearly and concisely:

    Question: {question}

    Answer:
    """)

    # Create a chain that:
    # 1. Takes a question
    # 2. Formats it into a prompt using the template
    # 3. Passes the prompt to the LLM
    # 4. Extracts the text from the LLM response
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(lambda x: llm_client.generate_text(str(x)))
        | RunnableLambda(lambda x: llm_client.extract_text_from_response(x))
    )

    # Return the chain
    return chain


# Exercise 2: Create a RAG chain with LCEL
def exercise2_rag_chain(documents, embedding_model):
    """
    Exercise 2: Create a RAG chain that retrieves relevant documents and generates an answer.

    Args:
        documents: List of document chunks
        embedding_model: Model to generate embeddings

    Returns:
        A callable chain that takes a question and returns an answer with sources
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for LCEL exercises. Install with 'pip install langchain'")

    # Initialize LLM client
    if GROQ_AVAILABLE:
        llm_client = GroqClient()
    else:
        llm_client = SimpleLLMClient()

    # Create a function to retrieve documents based on a query
    def retrieve_docs(query):
        # Generate query embedding
        query_embedding = embedding_model.embed_text(query)

        # Simple similarity search (cosine similarity)
        results = []
        for doc in documents:
            # In a real implementation, we would use a vector database
            # Here we're just simulating retrieval
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Add to results (in a real implementation, we would sort by relevance)
            results.append({
                "content": content,
                "metadata": metadata
            })

        # Return top 2 results (in a real implementation, we would return top-k by relevance)
        return results[:2]

    # Create a function to format documents for the prompt
    def format_docs(docs):
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.get("metadata", {}).get("source", f"Source {i+1}")
            page = doc.get("metadata", {}).get("page", "")
            page_info = f", page {page}" if page else ""
            formatted_docs.append(f"[{i+1}] {doc['content']} (Source: {source}{page_info})")

        return "\n\n".join(formatted_docs)

    # Create a prompt template for RAG
    prompt = PromptTemplate.from_template("""
    Answer the following question based on the provided context.
    If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # Create a chain that:
    # 1. Takes a question
    # 2. Retrieves relevant documents
    # 3. Formats the documents into context
    # 4. Passes the context and question to the prompt
    # 5. Passes the prompt to the LLM
    # 6. Extracts the text from the LLM response
    chain = (
        RunnableLambda(lambda x: {"question": x, "docs": retrieve_docs(x)})
        | {
            "question": lambda x: x["question"],
            "context": lambda x: format_docs(x["docs"])
        }
        | prompt
        | RunnableLambda(lambda x: llm_client.generate_text(str(x)))
        | RunnableLambda(lambda x: llm_client.extract_text_from_response(x))
    )

    # Return the chain
    return chain


# Exercise 3: Create a chain with branching logic
def exercise3_branching_chain():
    """
    Exercise 3: Create a chain with branching logic that handles different types of questions.

    Returns:
        A callable chain that takes a question and routes it to the appropriate sub-chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for LCEL exercises. Install with 'pip install langchain'")

    # Initialize LLM client
    if GROQ_AVAILABLE:
        llm_client = GroqClient()
    else:
        llm_client = SimpleLLMClient()

    # Create a function to classify the question type
    def classify_question(question):
        question_lower = question.lower()

        # Check for procedural questions (how to)
        if any(kw in question_lower for kw in ["how do i", "how to", "steps to", "process for", "procedure"]):
            return "procedural"

        # Check for opinion questions
        if any(kw in question_lower for kw in ["opinion", "think", "feel", "believe", "better", "worst", "best"]):
            return "opinion"

        # Default to factual
        return "factual"

    # Create separate prompt templates for each question type
    factual_prompt = PromptTemplate.from_template("""
    Please provide a factual answer to the following question.
    Stick to verified information and cite sources when possible.

    Question: {question}

    Factual Answer:
    """)

    opinion_prompt = PromptTemplate.from_template("""
    Please provide a thoughtful opinion on the following question.
    Consider multiple perspectives and explain your reasoning.

    Question: {question}

    Opinion:
    """)

    procedural_prompt = PromptTemplate.from_template("""
    Please provide step-by-step instructions to address the following question.
    Be clear, concise, and thorough in your explanation.

    Question: {question}

    Step-by-Step Instructions:
    """)

    # Create separate chains for each question type
    def create_chain_for_type(prompt_template):
        return (
            {"question": RunnablePassthrough()}
            | prompt_template
            | RunnableLambda(lambda x: llm_client.generate_text(str(x)))
            | RunnableLambda(lambda x: llm_client.extract_text_from_response(x))
        )

    factual_chain = create_chain_for_type(factual_prompt)
    opinion_chain = create_chain_for_type(opinion_prompt)
    procedural_chain = create_chain_for_type(procedural_prompt)

    # Create a router function that selects the appropriate chain based on question type
    def route_question(question):
        question_type = classify_question(question)

        if question_type == "opinion":
            return opinion_chain.invoke(question)
        elif question_type == "procedural":
            return procedural_chain.invoke(question)
        else:  # factual
            return factual_chain.invoke(question)

    # Create the main chain that:
    # 1. Takes a question
    # 2. Classifies the question type
    # 3. Routes to the appropriate sub-chain
    # 4. Returns the result
    chain = RunnableLambda(route_question)

    # Return the chain
    return chain


# Exercise 4: Create a chain with memory
def exercise4_memory_chain():
    """
    Exercise 4: Create a chain that maintains conversation history.

    Returns:
        A callable chain that takes a question and conversation history,
        and returns an answer that considers the history
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for LCEL exercises. Install with 'pip install langchain'")

    # Initialize LLM client
    if GROQ_AVAILABLE:
        llm_client = GroqClient()
    else:
        llm_client = SimpleLLMClient()

    # Create a prompt template that includes conversation history
    prompt = PromptTemplate.from_template("""
    You are having a conversation with a human. Respond to their latest question or message.

    Conversation History:
    {history}

    Human: {question}

    AI:
    """)

    # Create a function to format conversation history
    def format_history(history):
        formatted_history = []
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")

            if role.lower() == "human":
                formatted_history.append(f"Human: {content}")
            elif role.lower() == "ai":
                formatted_history.append(f"AI: {content}")

        return "\n".join(formatted_history)

    # Create a chain that:
    # 1. Takes a dictionary with "question" and "history" keys
    # 2. Formats the history
    # 3. Passes the formatted history and question to the prompt
    # 4. Passes the prompt to the LLM
    # 5. Extracts the text from the LLM response
    chain = (
        {
            "question": lambda x: x["question"],
            "history": lambda x: format_history(x["history"])
        }
        | prompt
        | RunnableLambda(lambda x: llm_client.generate_text(str(x)))
        | RunnableLambda(lambda x: llm_client.extract_text_from_response(x))
    )

    # Return the chain
    return chain


# Exercise 5: Create a chain with parallel retrievers
def exercise5_parallel_retrievers(documents, embedding_model):
    """
    Exercise 5: Create a chain that uses multiple retrievers in parallel and combines the results.

    Args:
        documents: List of document chunks
        embedding_model: Model to generate embeddings

    Returns:
        A callable chain that takes a question and returns an answer with sources
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for LCEL exercises. Install with 'pip install langchain'")

    # Initialize LLM client
    if GROQ_AVAILABLE:
        llm_client = GroqClient()
    else:
        llm_client = SimpleLLMClient()

    # Create a semantic retriever function
    def semantic_retriever(query):
        # Generate query embedding
        query_embedding = embedding_model.embed_text(query)

        # Simple similarity search (cosine similarity)
        results = []
        for i, doc in enumerate(documents):
            # In a real implementation, we would use a vector database
            # Here we're just simulating retrieval
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Add to results with a simulated score
            results.append({
                "content": content,
                "metadata": metadata,
                "score": 0.9 - (i * 0.1),  # Simulated decreasing relevance
                "retriever": "semantic"
            })

        # Return top 2 results
        return results[:2]

    # Create a keyword retriever function
    def keyword_retriever(query):
        # Extract keywords from query (simple approach)
        keywords = set(query.lower().split())
        stop_words = {"what", "where", "when", "who", "how", "is", "are", "the", "a", "an", "in", "on", "at"}
        keywords = keywords - stop_words

        # Match documents containing keywords
        results = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "").lower()
            metadata = doc.get("metadata", {})

            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in content)

            if matches > 0:
                results.append({
                    "content": doc.get("content", ""),
                    "metadata": metadata,
                    "score": matches / len(keywords) if keywords else 0,
                    "retriever": "keyword"
                })

        # Sort by score and return top 2
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:2]

    # Create a function to merge results from both retrievers
    def merge_results(semantic_results, keyword_results):
        # Combine results
        all_results = semantic_results + keyword_results

        # Deduplicate by content
        unique_results = {}
        for result in all_results:
            content = result["content"]
            if content not in unique_results or result["score"] > unique_results[content]["score"]:
                unique_results[content] = result

        # Convert back to list and sort by score
        merged_results = list(unique_results.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)

        # Return top 3 results
        return merged_results[:3]

    # Create a function to format documents for the prompt
    def format_docs(docs):
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.get("metadata", {}).get("source", f"Source {i+1}")
            retriever = doc.get("retriever", "unknown")
            formatted_docs.append(f"[{i+1}] {doc['content']} (Source: {source}, Retriever: {retriever})")

        return "\n\n".join(formatted_docs)

    # Create a prompt template for RAG
    prompt = PromptTemplate.from_template("""
    Answer the following question based on the provided context.
    The context was retrieved using multiple retrieval methods (semantic and keyword).

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # Create a chain that:
    # 1. Takes a question
    # 2. Passes it to both retrievers in parallel
    # 3. Merges the results
    # 4. Formats the documents into context
    # 5. Passes the context and question to the prompt
    # 6. Passes the prompt to the LLM
    # 7. Extracts the text from the LLM response
    chain = (
        RunnableLambda(lambda x: {
            "question": x,
            "semantic_results": semantic_retriever(x),
            "keyword_results": keyword_retriever(x)
        })
        | {
            "question": lambda x: x["question"],
            "merged_results": lambda x: merge_results(x["semantic_results"], x["keyword_results"])
        }
        | {
            "question": lambda x: x["question"],
            "context": lambda x: format_docs(x["merged_results"])
        }
        | prompt
        | RunnableLambda(lambda x: llm_client.generate_text(str(x)))
        | RunnableLambda(lambda x: llm_client.extract_text_from_response(x))
    )

    # Return the chain
    return chain


# Example usage
if __name__ == "__main__":
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not available. Install with 'pip install langchain' to run these exercises.")
        sys.exit(1)

    # Exercise 1: Basic LCEL chain
    print("\n=== Exercise 1: Basic LCEL Chain ===")
    try:
        chain = exercise1_basic_lcel_chain()
        if chain:
            result = chain.invoke("What is the capital of France?")
            print(f"Result: {result}")
        else:
            print("Exercise 1 not implemented yet.")
    except Exception as e:
        print(f"Error in Exercise 1: {e}")

    # Sample documents for exercises 2 and 5
    documents = [
        {
            "content": "Paris is the capital of France.",
            "metadata": {"source": "Geography Book", "page": 42}
        },
        {
            "content": "The Eiffel Tower is located in Paris.",
            "metadata": {"source": "Travel Guide", "page": 15}
        }
    ]

    # Simple embedding model for testing
    class SimpleEmbedding:
        def embed_text(self, text):
            # Very simple embedding function for testing
            import hashlib
            return [float(b) / 255.0 for b in hashlib.md5(text.encode()).digest()[:4]]

    embedding_model = SimpleEmbedding()

    # Exercise 2: RAG chain
    print("\n=== Exercise 2: RAG Chain ===")
    try:
        chain = exercise2_rag_chain(documents, embedding_model)
        if chain:
            result = chain.invoke("What is the capital of France?")
            print(f"Result: {result}")
        else:
            print("Exercise 2 not implemented yet.")
    except Exception as e:
        print(f"Error in Exercise 2: {e}")

    # Exercise 3: Branching chain
    print("\n=== Exercise 3: Branching Chain ===")
    try:
        chain = exercise3_branching_chain()
        if chain:
            result = chain.invoke("What is the capital of France?")
            print(f"Result: {result}")

            result = chain.invoke("How do I make a cake?")
            print(f"Result: {result}")

            result = chain.invoke("Do you think AI will replace humans?")
            print(f"Result: {result}")
        else:
            print("Exercise 3 not implemented yet.")
    except Exception as e:
        print(f"Error in Exercise 3: {e}")

    # Exercise 4: Memory chain
    print("\n=== Exercise 4: Memory Chain ===")
    try:
        chain = exercise4_memory_chain()
        if chain:
            history = [
                {"role": "human", "content": "My name is Alice."},
                {"role": "ai", "content": "Hello Alice, nice to meet you!"}
            ]
            result = chain.invoke({"question": "What is my name?", "history": history})
            print(f"Result: {result}")
        else:
            print("Exercise 4 not implemented yet.")
    except Exception as e:
        print(f"Error in Exercise 4: {e}")

    # Exercise 5: Parallel retrievers
    print("\n=== Exercise 5: Parallel Retrievers ===")
    try:
        chain = exercise5_parallel_retrievers(documents, embedding_model)
        if chain:
            result = chain.invoke("Tell me about Paris.")
            print(f"Result: {result}")
        else:
            print("Exercise 5 not implemented yet.")
    except Exception as e:
        print(f"Error in Exercise 5: {e}")
