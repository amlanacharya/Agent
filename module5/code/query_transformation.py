"""
Query Transformation Techniques for RAG Systems

This module implements various query transformation techniques that improve
retrieval quality by modifying the original query, including:
- Query expansion
- LLM-based query reformulation
- Multi-query retrieval
- Hypothetical Document Embeddings (HyDE)
- Step-back prompting

All implementations use LangChain Expression Language (LCEL) for improved
readability and composability.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import VectorStore

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers import HydeRetriever
    ADVANCED_RETRIEVERS_AVAILABLE = True
except ImportError:
    ADVANCED_RETRIEVERS_AVAILABLE = False


class QueryExpander:
    """
    A class for expanding queries with synonyms and related terms.
    
    Attributes:
        llm: Language model for generating expansions
        use_wordnet: Whether to use WordNet for synonym expansion
        expansion_factor: How much to expand the query (1-5)
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        use_wordnet: bool = True,
        expansion_factor: int = 2
    ):
        """Initialize the query expander."""
        self.llm = llm
        self.use_wordnet = use_wordnet and NLTK_AVAILABLE
        self.expansion_factor = min(max(expansion_factor, 1), 5)
        
        # Initialize WordNet if needed
        if self.use_wordnet:
            try:
                nltk.download('wordnet', quiet=True)
            except:
                self.use_wordnet = False
    
    def expand_with_wordnet(self, query: str) -> str:
        """Expand query using WordNet synonyms."""
        if not self.use_wordnet:
            return query
        
        # Tokenize query
        tokens = query.lower().split()
        expanded_tokens = tokens.copy()
        
        # Add synonyms for each token
        for token in tokens:
            # Skip short tokens and common words
            if len(token) <= 3 or token in {"the", "and", "or", "for", "to", "in", "on", "at", "by", "with"}:
                continue
            
            try:
                # Get synonyms from WordNet
                synsets = wordnet.synsets(token)
                
                # Add synonyms to expanded tokens
                for synset in synsets[:self.expansion_factor]:  # Limit to top N synsets
                    for lemma in synset.lemmas()[:2]:  # Limit to top 2 lemmas per synset
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != token and synonym not in expanded_tokens:
                            expanded_tokens.append(synonym)
            except:
                continue
        
        # Combine into expanded query
        expanded_query = ' '.join(expanded_tokens)
        return expanded_query
    
    def expand_with_llm(self, query: str) -> str:
        """Expand query using LLM-generated related terms."""
        if not self.llm:
            return query
        
        # Create expansion prompt
        expansion_prompt = f"""
        Expand the following search query by adding synonyms and related terms.
        Add {self.expansion_factor * 2} related terms that would help find relevant information.
        Format the output as a comma-separated list of terms.

        Original query: {query}

        Expanded terms:
        """
        
        try:
            # Get expansion from LLM
            response = self.llm.invoke(expansion_prompt)
            expanded_terms = response.content.strip().split(", ")
            
            # Combine original query with expanded terms
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            return expanded_query
        except:
            return query
    
    def expand(self, query: str) -> str:
        """Expand query using available methods."""
        # Try LLM expansion first if available
        if self.llm:
            try:
                return self.expand_with_llm(query)
            except:
                pass
        
        # Fall back to WordNet expansion
        if self.use_wordnet:
            try:
                return self.expand_with_wordnet(query)
            except:
                pass
        
        # Return original query if all methods fail
        return query
    
    def as_lcel_chain(self):
        """Return the expander as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.expand(x["query"]))
        )


class QueryReformulator:
    """
    A class for reformulating queries using an LLM.
    
    Attributes:
        llm: Language model for reformulation
        domain: Optional domain specialization
    """
    
    def __init__(
        self,
        llm: Any,
        domain: Optional[str] = None
    ):
        """Initialize the query reformulator."""
        self.llm = llm
        self.domain = domain
        
        # Create reformulation prompt based on domain
        if domain:
            self.prompt_template = self._create_domain_prompt(domain)
        else:
            self.prompt_template = self._create_general_prompt()
    
    def _create_general_prompt(self) -> str:
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
    
    def _create_domain_prompt(self, domain: str) -> str:
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
        
        return domain_prompts.get(domain.lower(), self._create_general_prompt())
    
    def reformulate(self, query: str) -> str:
        """Reformulate the query using the LLM."""
        try:
            # Create prompt
            prompt = self.prompt_template.format(query=query)
            
            # Get reformulation from LLM
            response = self.llm.invoke(prompt)
            reformulated_query = response.content.strip()
            
            return reformulated_query
        except:
            # Return original query if reformulation fails
            return query
    
    def as_lcel_chain(self):
        """Return the reformulator as an LCEL chain."""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        
        return (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | RunnableLambda(lambda x: x.content.strip())
        )


class MultiQueryGenerator:
    """
    A class for generating multiple query variations.
    
    Attributes:
        llm: Language model for generating variations
        num_variations: Number of variations to generate
    """
    
    def __init__(
        self,
        llm: Any,
        num_variations: int = 3
    ):
        """Initialize the multi-query generator."""
        self.llm = llm
        self.num_variations = min(max(num_variations, 1), 5)
    
    def generate_variations(self, query: str) -> List[str]:
        """Generate multiple variations of the query."""
        # Create prompt for generating variations
        prompt = f"""
        Generate {self.num_variations} different variations of the following query.
        Each variation should explore a different aspect or use different terminology.
        Format each variation on a new line.
        
        Original query: {query}
        
        Variations:
        """
        
        try:
            # Get variations from LLM
            response = self.llm.invoke(prompt)
            variations = response.content.strip().split("\n")
            
            # Clean up variations
            cleaned_variations = [var.strip() for var in variations if var.strip()]
            
            # Add original query
            all_queries = [query] + cleaned_variations
            
            return all_queries
        except:
            # Return just the original query if generation fails
            return [query]
    
    def retrieve_with_variations(self, query: str, retriever: BaseRetriever) -> List[Document]:
        """Retrieve documents using multiple query variations."""
        # Generate query variations
        query_variations = self.generate_variations(query)
        
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
    
    def as_lcel_chain(self, retriever: BaseRetriever):
        """Return the multi-query generator as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.retrieve_with_variations(x["query"], retriever))
        )
    
    def create_langchain_retriever(self, base_retriever: BaseRetriever) -> BaseRetriever:
        """Create a LangChain MultiQueryRetriever if available."""
        if ADVANCED_RETRIEVERS_AVAILABLE:
            try:
                return MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=self.llm,
                    parser_key="variations"
                )
            except:
                pass
        
        # Create a custom retriever if MultiQueryRetriever is not available
        class CustomMultiQueryRetriever(BaseRetriever):
            def __init__(self, generator, base_retriever):
                super().__init__()
                self.generator = generator
                self.base_retriever = base_retriever
            
            def _get_relevant_documents(self, query):
                return self.generator.retrieve_with_variations(query, self.base_retriever)
        
        return CustomMultiQueryRetriever(self, base_retriever)


class HyDEGenerator:
    """
    A class for implementing Hypothetical Document Embeddings (HyDE).
    
    Attributes:
        llm: Language model for generating hypothetical documents
        embedding_model: Model to generate embeddings
        vectorstore: Vector store for similarity search
    """
    
    def __init__(
        self,
        llm: Any,
        embedding_model: Embeddings,
        vectorstore: VectorStore
    ):
        """Initialize the HyDE generator."""
        self.llm = llm
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that answers the query."""
        # Create prompt for generating hypothetical document
        prompt = f"""
        Write a detailed passage that directly answers the following question.
        Make the passage informative and comprehensive, as if it were extracted from a document.
        
        Question: {query}
        
        Passage:
        """
        
        try:
            # Get hypothetical document from LLM
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            # Return a simple document if generation fails
            return f"This document contains information about {query}."
    
    def retrieve_with_hyde(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using HyDE."""
        try:
            # Generate hypothetical document
            hypothetical_doc = self.generate_hypothetical_document(query)
            
            # Embed hypothetical document
            hyde_embedding = self.embedding_model.embed_query(hypothetical_doc)
            
            # Find similar documents
            similar_docs = self.vectorstore.similarity_search_by_vector(hyde_embedding, k=top_k)
            
            return similar_docs
        except:
            # Fall back to regular search if HyDE fails
            return self.vectorstore.similarity_search(query, k=top_k)
    
    def as_lcel_chain(self):
        """Return the HyDE generator as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.retrieve_with_hyde(x["query"]))
        )
    
    def create_langchain_retriever(self) -> BaseRetriever:
        """Create a LangChain HydeRetriever if available."""
        if ADVANCED_RETRIEVERS_AVAILABLE:
            try:
                return HydeRetriever.from_llm(
                    vectorstore=self.vectorstore,
                    llm=self.llm,
                    prompt_template="""
                    Write a passage that answers the question.
                    Question: {question}
                    Passage:
                    """
                )
            except:
                pass
        
        # Create a custom retriever if HydeRetriever is not available
        class CustomHyDERetriever(BaseRetriever):
            def __init__(self, generator):
                super().__init__()
                self.generator = generator
            
            def _get_relevant_documents(self, query):
                return self.generator.retrieve_with_hyde(query)
        
        return CustomHyDERetriever(self)


class StepBackPrompting:
    """
    A class for implementing step-back prompting.
    
    Attributes:
        llm: Language model for generating general questions
        retriever: Retriever for finding documents
    """
    
    def __init__(
        self,
        llm: Any,
        retriever: BaseRetriever
    ):
        """Initialize the step-back prompting system."""
        self.llm = llm
        self.retriever = retriever
    
    def generate_general_question(self, query: str) -> str:
        """Generate a more general question from the specific query."""
        # Create prompt for generating general question
        prompt = f"""
        Given the following specific question, generate a more general question that would help provide context for answering the specific question.
        
        Specific question: {query}
        
        General question:
        """
        
        try:
            # Get general question from LLM
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except:
            # Return a simple general question if generation fails
            return f"What is {' '.join(query.split()[:3])}?"
    
    def retrieve_with_step_back(self, query: str) -> List[Document]:
        """Retrieve documents using step-back prompting."""
        try:
            # Generate general question
            general_question = self.generate_general_question(query)
            
            # Retrieve documents for general question
            general_docs = self.retriever.get_relevant_documents(general_question)
            
            # Retrieve documents for specific question
            specific_docs = self.retriever.get_relevant_documents(query)
            
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
        except:
            # Fall back to regular retrieval if step-back fails
            return self.retriever.get_relevant_documents(query)
    
    def as_lcel_chain(self):
        """Return the step-back prompting system as an LCEL chain."""
        # Create step-back prompt
        step_back_prompt = ChatPromptTemplate.from_template("""
        Given the following specific question, generate a more general question that would help provide context for answering the specific question.

        Specific question: {query}

        General question:
        """)
        
        # Create general question chain
        general_question_chain = (
            {"query": RunnablePassthrough()}
            | step_back_prompt
            | self.llm
            | RunnableLambda(lambda x: x.content.strip())
        )
        
        # Create retrieval chain
        retrieval_chain = (
            {"query": RunnablePassthrough(), "general_query": general_question_chain}
            | RunnableLambda(lambda x: {
                "general_docs": self.retriever.get_relevant_documents(x["general_query"]),
                "specific_docs": self.retriever.get_relevant_documents(x["query"])
            })
            | RunnableLambda(lambda x: x["general_docs"] + x["specific_docs"])
            | RunnableLambda(lambda docs: list({doc.page_content: doc for doc in docs}.values()))
        )
        
        return retrieval_chain


class QueryTransformationRouter:
    """
    A router that selects the appropriate query transformation technique.
    
    Attributes:
        transformers: Dictionary mapping transformation types to transformers
        router_function: Function to route queries to transformers
    """
    
    def __init__(
        self,
        transformers: Dict[str, Any],
        router_function: Optional[Callable[[str], str]] = None
    ):
        """Initialize the query transformation router."""
        self.transformers = transformers
        self.router_function = router_function or self._default_router
    
    def _default_router(self, query: str) -> str:
        """Default router function based on query keywords."""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["how", "steps", "process", "procedure"]):
            return "step_back"
        elif any(keyword in query_lower for keyword in ["technical", "specific", "detailed"]):
            return "hyde"
        elif any(keyword in query_lower for keyword in ["compare", "difference", "versus", "vs"]):
            return "multi_query"
        elif any(keyword in query_lower for keyword in ["similar", "like", "related"]):
            return "expansion"
        else:
            return "reformulation"
    
    def transform(self, query: str) -> Any:
        """Transform the query using the appropriate technique."""
        transformation_type = self.router_function(query)
        
        if transformation_type in self.transformers:
            transformer = self.transformers[transformation_type]
            
            # Handle different transformer types
            if hasattr(transformer, "expand"):
                return transformer.expand(query)
            elif hasattr(transformer, "reformulate"):
                return transformer.reformulate(query)
            elif hasattr(transformer, "retrieve_with_variations"):
                return transformer.retrieve_with_variations(query, self.transformers.get("retriever"))
            elif hasattr(transformer, "retrieve_with_hyde"):
                return transformer.retrieve_with_hyde(query)
            elif hasattr(transformer, "retrieve_with_step_back"):
                return transformer.retrieve_with_step_back(query)
            else:
                # Try to call the transformer directly
                try:
                    return transformer(query)
                except:
                    return query
        else:
            # Default to reformulation if available
            if "reformulation" in self.transformers:
                return self.transformers["reformulation"].reformulate(query)
            else:
                return query
    
    def as_lcel_chain(self):
        """Return the router as an LCEL chain."""
        # Create a branch for each transformer
        branches = []
        
        for transformation_type, transformer in self.transformers.items():
            if transformation_type != "retriever":  # Skip the retriever
                condition = lambda x, t=transformation_type: self.router_function(x["query"]) == t
                
                if hasattr(transformer, "as_lcel_chain"):
                    branch = transformer.as_lcel_chain()
                else:
                    # Create a simple chain for the transformer
                    branch = RunnableLambda(lambda x, t=transformer: t(x["query"]))
                
                branches.append((condition, branch))
        
        # Add default branch
        if "reformulation" in self.transformers and hasattr(self.transformers["reformulation"], "as_lcel_chain"):
            default_branch = self.transformers["reformulation"].as_lcel_chain()
        else:
            default_branch = RunnableLambda(lambda x: x["query"])
        
        # Create the branch chain
        branch_chain = RunnableBranch(*branches, default_branch)
        
        # Create the final chain
        return (
            {"query": RunnablePassthrough()}
            | branch_chain
        )


# Example usage
def create_query_transformation_system(
    llm: Any,
    embedding_model: Embeddings,
    vectorstore: VectorStore,
    retriever: BaseRetriever
) -> QueryTransformationRouter:
    """
    Create a complete query transformation system with multiple techniques.
    
    Args:
        llm: Language model for transformations
        embedding_model: Model to generate embeddings
        vectorstore: Vector store for similarity search
        retriever: Base retriever for document retrieval
        
    Returns:
        A query transformation router
    """
    # Create query expander
    query_expander = QueryExpander(llm=llm)
    
    # Create query reformulator
    query_reformulator = QueryReformulator(llm=llm)
    
    # Create multi-query generator
    multi_query_generator = MultiQueryGenerator(llm=llm)
    
    # Create HyDE generator
    hyde_generator = HyDEGenerator(llm=llm, embedding_model=embedding_model, vectorstore=vectorstore)
    
    # Create step-back prompting system
    step_back_system = StepBackPrompting(llm=llm, retriever=retriever)
    
    # Create query transformation router
    router = QueryTransformationRouter(
        transformers={
            "expansion": query_expander,
            "reformulation": query_reformulator,
            "multi_query": multi_query_generator,
            "hyde": hyde_generator,
            "step_back": step_back_system,
            "retriever": retriever  # Include retriever for multi-query
        }
    )
    
    return router
