"""
Module 5 - Lesson 2 Exercises: Query Transformation Techniques

This module contains exercises for implementing query transformation techniques
including query expansion, LLM-based query reformulation, multi-query retrieval,
hypothetical document embeddings (HyDE), and step-back prompting.
"""

from typing import List, Any, Optional
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnableLambda

# Check if LangChain is available
try:
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Check if NLTK is available
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Check if Groq is available
try:
    from langchain.chat_models import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class SimpleLLMClient:
    """Simple LLM client for when Groq is not available."""

    def __init__(self):
        pass

    def invoke(self, prompt):
        return {"content": f"This is a simulated response to: {prompt}"}


# Exercise 1: Implement a query expansion system
def exercise1_query_expansion(query: str, llm: Optional[Any] = None) -> str:
    """
    Exercise 1: Implement a query expansion system that adds synonyms and related terms.

    Args:
        query: Original query
        llm: Optional language model for expansion

    Returns:
        Expanded query
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # Initialize LLM client if not provided
    if llm is None:
        if GROQ_AVAILABLE:
            llm = ChatGroq(temperature=0.2, model_name="llama2-70b-4096")
        else:
            llm = SimpleLLMClient()

    # Implement query expansion
    # 1. Use WordNet to find synonyms (if NLTK is available)
    # 2. Use the LLM to generate related terms
    # 3. Combine the original query with the expanded terms
    # 4. Return the expanded query

    expanded_query = query

    # Try WordNet expansion if available
    if NLTK_AVAILABLE:
        try:
            # Download WordNet if not already downloaded
            nltk.download('wordnet', quiet=True)

            # Tokenize query
            tokens = query.lower().split()
            expanded_tokens = tokens.copy()

            # Add synonyms for each token
            for token in tokens:
                # Skip short tokens and common words
                if len(token) <= 3 or token in {"the", "and", "or", "for", "to", "in", "on", "at", "by", "with"}:
                    continue

                # Get synonyms from WordNet
                synsets = wordnet.synsets(token)

                # Add synonyms to expanded tokens
                for synset in synsets[:2]:  # Limit to top 2 synsets
                    for lemma in synset.lemmas()[:2]:  # Limit to top 2 lemmas per synset
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != token and synonym not in expanded_tokens:
                            expanded_tokens.append(synonym)

            # Combine into expanded query
            expanded_query = ' '.join(expanded_tokens)
        except Exception as e:
            print(f"WordNet expansion failed: {e}")

    # Try LLM expansion
    try:
        # Create expansion prompt
        expansion_prompt = ChatPromptTemplate.from_template("""
        Expand the following search query by adding synonyms and related terms.
        Format the output as a comma-separated list of terms.

        Original query: {query}

        Expanded terms:
        """)

        # Get expansion from LLM
        response = llm.invoke(expansion_prompt.format(query=query))

        # Extract expanded terms
        expanded_terms = []
        if hasattr(response, 'content'):
            expanded_terms = response.content.strip().split(", ")
        else:
            # Handle different response formats
            content = response.get('content', '')
            if content:
                expanded_terms = content.strip().split(", ")

        # Filter out empty terms
        expanded_terms = [term for term in expanded_terms if term]

        if expanded_terms:
            # Combine original query with expanded terms
            expanded_query = f"{query} {' '.join(expanded_terms)}"
    except Exception as e:
        print(f"LLM expansion failed: {e}")

    return expanded_query


# Exercise 2: Build an LLM-based query reformulator
def exercise2_query_reformulation(query: str, llm: Any, domain: Optional[str] = None) -> str:
    """
    Exercise 2: Build an LLM-based query reformulator that rephrases queries for better matching.

    Args:
        query: Original query
        llm: Language model for reformulation
        domain: Optional domain specialization

    Returns:
        Reformulated query
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # TODO: Implement query reformulation
    # 1. Create a prompt for query reformulation
    # 2. If domain is provided, create a domain-specific prompt
    # 3. Use the LLM to reformulate the query
    # 4. Return the reformulated query

    # Create reformulation prompt based on domain
    if domain:
        prompt_template = _create_domain_prompt(domain)
    else:
        prompt_template = _create_general_prompt()

    # Placeholder implementation
    try:
        # Create prompt
        prompt = prompt_template.format(query=query)

        # Get reformulation from LLM
        response = llm.invoke(prompt)
        reformulated_query = response.content.strip()

        return reformulated_query
    except:
        # Return original query if reformulation fails
        return query


def _create_general_prompt() -> str:
    """Create a general reformulation prompt."""
    return """
    You are an expert at reformulating search queries to improve retrieval results.
    Your task is to reformulate the following query to make it more effective for retrieving relevant information.

    Consider:
    1. Using more specific terminology
    2. Adding context if the query is ambiguous
    3. Breaking down complex queries
    4. Using domain-specific language

    Original query: {query}

    Reformulated query:
    """


def _create_domain_prompt(domain: str) -> str:
    """Create a domain-specific reformulation prompt."""
    domain_prompts = {
        "medical": """
        You are a medical search expert. Reformulate the following query to use proper medical terminology
        and make it more effective for retrieving relevant medical information.

        Original query: {query}

        Reformulated query:
        """,
        "legal": """
        You are a legal search expert. Reformulate the following query to use proper legal terminology
        and make it more effective for retrieving relevant legal information.

        Original query: {query}

        Reformulated query:
        """,
        "technical": """
        You are a technical search expert. Reformulate the following query to use proper technical terminology
        and make it more effective for retrieving relevant technical information.

        Original query: {query}

        Reformulated query:
        """,
        "academic": """
        You are an academic search expert. Reformulate the following query to use proper academic terminology
        and make it more effective for retrieving relevant scholarly information.

        Original query: {query}

        Reformulated query:
        """
    }

    return domain_prompts.get(domain.lower(), _create_general_prompt())


# Exercise 3: Create a multi-query retriever
def exercise3_multi_query_retrieval(query: str, retriever: BaseRetriever, llm: Any, num_variations: int = 3) -> List[Document]:
    """
    Exercise 3: Create a multi-query retrieval system that generates and uses multiple query variations.

    Args:
        query: Original query
        retriever: Base retriever for document retrieval
        llm: Language model for generating variations
        num_variations: Number of variations to generate

    Returns:
        List of retrieved documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # TODO: Implement multi-query retrieval
    # 1. Generate multiple variations of the original query
    # 2. Retrieve documents for each variation
    # 3. Combine and deduplicate the results
    # 4. Return the combined results

    # Placeholder implementation
    # Generate query variations
    query_variations = [query]

    try:
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

        # Add variations to the list
        query_variations.extend(cleaned_variations)
    except:
        # Add some simple variations if LLM fails
        query_variations.append(f"information about {query}")
        query_variations.append(f"explain {query}")

    # Retrieve documents for each variation
    all_docs = []
    for query_var in query_variations:
        try:
            docs = retriever.get_relevant_documents(query_var)
            all_docs.extend(docs)
        except:
            continue

    # Deduplicate documents
    unique_docs = []
    seen_contents = set()

    for doc in all_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs


# Exercise 4: Develop a HyDE implementation
def exercise4_hyde(query: str, vectorstore: Any, llm: Any, embedding_model: Embeddings) -> List[Document]:
    """
    Exercise 4: Develop a Hypothetical Document Embeddings (HyDE) system for improved semantic retrieval.

    Args:
        query: Original query
        vectorstore: Vector store for similarity search
        llm: Language model for generating hypothetical documents
        embedding_model: Model to generate embeddings

    Returns:
        List of retrieved documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # Implement HyDE (Hypothetical Document Embeddings)
    # 1. Generate a hypothetical document that would answer the query
    # 2. Embed the hypothetical document
    # 3. Find real documents similar to the hypothetical one
    # 4. Return the most similar real documents

    try:
        # Generate hypothetical document using the LLM
        hypothetical_doc_prompt = ChatPromptTemplate.from_template("""
        Write a detailed passage that directly answers the following question.
        Make the passage informative and comprehensive, as if it were extracted from a document.

        Question: {query}

        Passage:
        """)

        # Get hypothetical document from LLM
        response = llm.invoke(hypothetical_doc_prompt.format(query=query))

        # Extract the content
        if hasattr(response, 'content'):
            hypothetical_doc = response.content.strip()
        else:
            hypothetical_doc = response.get('content', f"This document contains information about {query}.")

        # Embed the hypothetical document
        hyde_embedding = embedding_model.embed_query(hypothetical_doc)

        # Find similar documents using the hypothetical document embedding
        similar_docs = vectorstore.similarity_search_by_vector(
            embedding=hyde_embedding,
            k=5
        )

        # Add metadata to indicate these were retrieved via HyDE
        for doc in similar_docs:
            doc.metadata["retrieval_method"] = "hyde"

        return similar_docs
    except Exception as e:
        print(f"HyDE retrieval failed: {e}")
        # Fall back to regular search if HyDE fails
        return vectorstore.similarity_search(query, k=5)


# The hypothetical document generation is now implemented directly in the exercise4_hyde function


# Exercise 5: Implement step-back prompting
def exercise5_step_back_prompting(query: str, retriever: BaseRetriever, llm: Any) -> List[Document]:
    """
    Exercise 5: Implement a step-back prompting system that handles complex queries by first retrieving general information.

    Args:
        query: Original query
        retriever: Base retriever for document retrieval
        llm: Language model for generating general questions

    Returns:
        List of retrieved documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # Implement step-back prompting
    # 1. Generate a more general question from the specific query
    # 2. Retrieve documents for the general question
    # 3. Retrieve documents for the original query
    # 4. Combine and deduplicate the results
    # 5. Return the combined results

    try:
        # Generate a more general question
        general_question_prompt = ChatPromptTemplate.from_template("""
        Given the following specific question, generate a more general question that would help provide context for answering the specific question.

        Specific question: {query}

        General question:
        """)

        # Get general question from LLM
        response = llm.invoke(general_question_prompt.format(query=query))

        # Extract the general question
        if hasattr(response, 'content'):
            general_question = response.content.strip()
        else:
            general_question = response.get('content', f"What is {' '.join(query.split()[:3])}?")

        print(f"Generated general question: {general_question}")

        # Retrieve documents for the general question
        general_docs = retriever.get_relevant_documents(general_question)

        # Add metadata to indicate these were retrieved for the general question
        for doc in general_docs:
            doc.metadata["retrieval_for"] = "general_question"
            doc.metadata["general_question"] = general_question

        # Retrieve documents for the specific question
        specific_docs = retriever.get_relevant_documents(query)

        # Add metadata to indicate these were retrieved for the specific question
        for doc in specific_docs:
            doc.metadata["retrieval_for"] = "specific_question"

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
    except Exception as e:
        print(f"Step-back prompting failed: {e}")
        # Fall back to regular retrieval if step-back fails
        return retriever.get_relevant_documents(query)


# The general question generation is now implemented directly in the exercise5_step_back_prompting function


# Exercise 6: Combine multiple query transformation techniques
def exercise6_combined_query_transformation(
    query: str,
    retriever: BaseRetriever,
    llm: Any,
    embedding_model: Embeddings,
    vectorstore: Any
) -> List[Document]:
    """
    Exercise 6: Build an advanced query transformation system that combines multiple techniques based on query type.

    Args:
        query: Original query
        retriever: Base retriever for document retrieval
        llm: Language model for transformations
        embedding_model: Model to generate embeddings
        vectorstore: Vector store for similarity search

    Returns:
        List of retrieved documents
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # TODO: Implement combined query transformation
    # 1. Analyze the query to determine the appropriate transformation technique
    # 2. Apply the selected technique
    # 3. Return the results

    # Placeholder implementation
    query_lower = query.lower()

    # Route query to appropriate transformation technique
    if any(keyword in query_lower for keyword in ["how", "steps", "process", "procedure"]):
        # Use step-back prompting for procedural queries
        return exercise5_step_back_prompting(query, retriever, llm)
    elif any(keyword in query_lower for keyword in ["technical", "specific", "detailed"]):
        # Use HyDE for technical queries
        return exercise4_hyde(query, vectorstore, llm, embedding_model)
    elif any(keyword in query_lower for keyword in ["compare", "difference", "versus", "vs"]):
        # Use multi-query retrieval for comparison queries
        return exercise3_multi_query_retrieval(query, retriever, llm)
    elif any(keyword in query_lower for keyword in ["similar", "like", "related"]):
        # Use query expansion for similarity queries
        expanded_query = exercise1_query_expansion(query, llm)
        return retriever.get_relevant_documents(expanded_query)
    else:
        # Use query reformulation for general queries
        reformulated_query = exercise2_query_reformulation(query, llm)
        return retriever.get_relevant_documents(reformulated_query)


# Exercise 7: Implement an LCEL query transformation chain
def exercise7_lcel_query_transformation(
    retriever: BaseRetriever,
    llm: Any,
    embedding_model: Embeddings,
    vectorstore: Any
):
    """
    Exercise 7: Implement an LCEL query transformation chain that combines multiple techniques.

    Args:
        retriever: Base retriever for document retrieval
        llm: Language model for transformations
        embedding_model: Model to generate embeddings
        vectorstore: Vector store for similarity search

    Returns:
        An LCEL query transformation chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for these exercises. Install with 'pip install langchain'")

    # Implement LCEL query transformation chain
    # 1. Create LCEL chains for each transformation technique
    # 2. Create a router that selects the appropriate technique based on the query
    # 3. Return the combined LCEL chain

    # Create expansion chain
    def expand_query(query):
        return exercise1_query_expansion(query, llm)

    expansion_chain = (
        RunnableLambda(expand_query)
        | RunnableLambda(lambda x: retriever.get_relevant_documents(x))
    )

    # Create reformulation chain
    def reformulate_query(query):
        return exercise2_query_reformulation(query, llm)

    reformulation_chain = (
        RunnableLambda(reformulate_query)
        | RunnableLambda(lambda x: retriever.get_relevant_documents(x))
    )

    # Create multi-query chain
    def multi_query(query):
        return exercise3_multi_query_retrieval(query, retriever, llm)

    multi_query_chain = RunnableLambda(multi_query)

    # Create HyDE chain
    def hyde_query(query):
        return exercise4_hyde(query, vectorstore, llm, embedding_model)

    hyde_chain = RunnableLambda(hyde_query)

    # Create step-back chain
    def step_back(query):
        return exercise5_step_back_prompting(query, retriever, llm)

    step_back_chain = RunnableLambda(step_back)

    # We'll use inline lambda functions in the RunnableBranch instead of a separate router function

    # Create LCEL chain with RunnableBranch
    from langchain.schema.runnable import RunnableBranch

    # Create branch chain that routes based on query content
    branch_chain = RunnableBranch(
        (lambda x: "how" in x.lower() or "steps" in x.lower() or "process" in x.lower() or "procedure" in x.lower(),
         step_back_chain),
        (lambda x: "technical" in x.lower() or "specific" in x.lower() or "detailed" in x.lower(),
         hyde_chain),
        (lambda x: "compare" in x.lower() or "difference" in x.lower() or "versus" in x.lower() or "vs" in x.lower(),
         multi_query_chain),
        (lambda x: "similar" in x.lower() or "like" in x.lower() or "related" in x.lower(),
         expansion_chain),
        reformulation_chain  # Default
    )

    # Create the final transformation chain
    transformation_chain = branch_chain

    return transformation_chain
