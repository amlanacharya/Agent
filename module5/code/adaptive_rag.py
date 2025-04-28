"""
Self-Querying and Adaptive RAG Systems

This module implements various adaptive RAG techniques that dynamically adjust
retrieval strategies based on query characteristics, including:
- Self-querying retrieval for metadata filtering
- Query classification and routing
- Multi-strategy retrieval
- Adaptive RAG systems

All implementations use LangChain Expression Language (LCEL) for improved
readability and composability.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import re
import json
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.retrievers import SelfQueryRetriever, ContextualCompressionRetriever, EnsembleRetriever, BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq


class SelfQueryingSystem:
    """System for self-querying retrieval with metadata filtering."""
    
    def __init__(
        self,
        vectorstore: Any,
        llm: Any,
        metadata_field_info: List[AttributeInfo],
        document_content_description: str = "Documents about various topics",
        verbose: bool = False
    ):
        """Initialize the self-querying system.
        
        Args:
            vectorstore: Vector store for document retrieval
            llm: Language model for query construction
            metadata_field_info: Information about metadata fields
            document_content_description: Description of document contents
            verbose: Whether to print verbose output
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.metadata_field_info = metadata_field_info
        self.document_content_description = document_content_description
        self.verbose = verbose
        
        # Create self-query retriever
        self.retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=verbose
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents based on the query with metadata filtering.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # The SelfQueryRetriever automatically extracts metadata filters
        documents = self.retriever.get_relevant_documents(query)
        
        # Limit to top_k
        return documents[:top_k]
    
    def as_lcel_chain(self):
        """Return the self-querying system as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.retrieve(x["query"]))
        )


class QueryClassifier:
    """Classifier for categorizing queries into different types."""
    
    def __init__(
        self,
        llm: Any,
        query_types: List[str] = None,
        verbose: bool = False
    ):
        """Initialize the query classifier.
        
        Args:
            llm: Language model for classification
            query_types: List of query types to classify into
            verbose: Whether to print verbose output
        """
        self.llm = llm
        self.query_types = query_types or [
            "factual", "conceptual", "procedural", 
            "comparative", "exploratory"
        ]
        self.verbose = verbose
        
        # Create classification prompt
        self.prompt = ChatPromptTemplate.from_template("""
        Classify the following query into one of these categories:
        {query_types}
        
        Query: {query}
        
        Category:
        """)
    
    def classify(self, query: str) -> str:
        """Classify the query into a type.
        
        Args:
            query: User query
            
        Returns:
            Query type
        """
        # Format query types for the prompt
        query_types_str = "\n".join([f"- {qt}: {self._get_type_description(qt)}" 
                                    for qt in self.query_types])
        
        # Invoke the LLM
        response = self.llm.invoke(
            self.prompt.format(query_types=query_types_str, query=query)
        )
        
        # Extract the category
        category = response.content.strip().lower()
        
        # Ensure it's one of our query types
        if category not in self.query_types:
            # Try to extract just the category name
            for qt in self.query_types:
                if qt in category:
                    category = qt
                    break
            else:
                # Default to factual if we can't match
                category = "factual"
        
        if self.verbose:
            print(f"Query: {query}")
            print(f"Classified as: {category}")
        
        return category
    
    def _get_type_description(self, query_type: str) -> str:
        """Get a description for a query type.
        
        Args:
            query_type: Query type
            
        Returns:
            Description of the query type
        """
        descriptions = {
            "factual": "Seeking specific facts or information",
            "conceptual": "Seeking explanation of concepts",
            "procedural": "Asking how to do something",
            "comparative": "Comparing multiple things",
            "exploratory": "Broad exploration of a topic"
        }
        
        return descriptions.get(query_type, "")
    
    def as_lcel_chain(self):
        """Return the query classifier as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.classify(x["query"]))
        )


class QueryRouter:
    """Router for directing queries to specialized retrievers."""
    
    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        classifier: QueryClassifier = None,
        default_retriever_key: str = None,
        verbose: bool = False
    ):
        """Initialize the query router.
        
        Args:
            retrievers: Dictionary of retrievers
            classifier: Query classifier for routing
            default_retriever_key: Key of the default retriever
            verbose: Whether to print verbose output
        """
        self.retrievers = retrievers
        self.classifier = classifier
        self.default_retriever_key = default_retriever_key or next(iter(retrievers.keys()))
        self.verbose = verbose
    
    def route_and_retrieve(self, query: str) -> List[Document]:
        """Route the query to the appropriate retriever and retrieve documents.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved documents
        """
        # Classify the query if classifier is available
        if self.classifier:
            query_type = self.classifier.classify(query)
        else:
            # Use simple keyword-based classification
            query_type = self._simple_classify(query)
        
        # Get the appropriate retriever
        retriever = self.retrievers.get(query_type, self.retrievers[self.default_retriever_key])
        
        if self.verbose:
            print(f"Query: {query}")
            print(f"Routed to: {query_type}")
        
        # Retrieve documents
        return retriever.get_relevant_documents(query)
    
    def _simple_classify(self, query: str) -> str:
        """Simple keyword-based query classification.
        
        Args:
            query: User query
            
        Returns:
            Query type
        """
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["what is", "explain", "describe", "define"]):
            return "conceptual"
        elif any(term in query_lower for term in ["how to", "steps", "procedure", "process"]):
            return "procedural"
        elif any(term in query_lower for term in ["compare", "difference", "versus", "vs"]):
            return "comparative"
        elif any(term in query_lower for term in ["explore", "tell me about", "overview"]):
            return "exploratory"
        else:
            return "factual"
    
    def as_lcel_chain(self):
        """Return the query router as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.route_and_retrieve(x["query"]))
        )


class MultiStrategyRetrieval:
    """System for applying different retrieval strategies based on query characteristics."""
    
    def __init__(
        self,
        strategies: Dict[str, BaseRetriever],
        strategy_selector: Callable[[str], str] = None,
        default_strategy: str = None,
        verbose: bool = False
    ):
        """Initialize the multi-strategy retrieval system.
        
        Args:
            strategies: Dictionary of retrieval strategies
            strategy_selector: Function to select strategy based on query
            default_strategy: Default strategy to use
            verbose: Whether to print verbose output
        """
        self.strategies = strategies
        self.strategy_selector = strategy_selector or self._default_strategy_selector
        self.default_strategy = default_strategy or next(iter(strategies.keys()))
        self.verbose = verbose
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents using the appropriate strategy.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved documents
        """
        # Select strategy
        strategy = self.strategy_selector(query)
        
        # Get the appropriate retriever
        retriever = self.strategies.get(strategy, self.strategies[self.default_strategy])
        
        if self.verbose:
            print(f"Query: {query}")
            print(f"Selected strategy: {strategy}")
        
        # Retrieve documents
        return retriever.get_relevant_documents(query)
    
    def _default_strategy_selector(self, query: str) -> str:
        """Default strategy selector based on query keywords.
        
        Args:
            query: User query
            
        Returns:
            Selected strategy
        """
        query_lower = query.lower()
        
        if "explain" in query_lower or "what is" in query_lower:
            return "compression"  # For explanatory queries
        elif any(term in query_lower for term in ["find", "search", "locate"]):
            return "semantic"  # For search queries
        elif any(term in query_lower for term in ["compare", "difference", "versus"]):
            return "ensemble"  # For comparative queries
        else:
            return self.default_strategy
    
    def as_lcel_chain(self):
        """Return the multi-strategy retrieval system as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: {
                "query": x["query"],
                "strategy": self.strategy_selector(x["query"])
            })
            | RunnableLambda(lambda x: self.strategies.get(
                x["strategy"], 
                self.strategies[self.default_strategy]
            ).get_relevant_documents(x["query"]))
        )


class AdaptiveRAGSystem:
    """Comprehensive adaptive RAG system that combines multiple techniques."""
    
    def __init__(
        self,
        vectorstore: Any,
        llm: Any,
        embedding_model: Embeddings,
        documents: List[Document] = None,
        metadata_field_info: List[AttributeInfo] = None,
        verbose: bool = False
    ):
        """Initialize the adaptive RAG system.
        
        Args:
            vectorstore: Vector store for document retrieval
            llm: Language model for various tasks
            embedding_model: Model to generate embeddings
            documents: List of documents to index
            metadata_field_info: Information about metadata fields
            verbose: Whether to print verbose output
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.embedding_model = embedding_model
        self.documents = documents or []
        self.metadata_field_info = metadata_field_info or []
        self.verbose = verbose
        
        # Create base retrievers
        self.semantic_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        if documents:
            self.keyword_retriever = BM25Retriever.from_documents(documents, k=5)
        else:
            self.keyword_retriever = None
        
        # Create specialized retrievers
        self._create_specialized_retrievers()
        
        # Create query analyzer
        self._create_query_analyzer()
    
    def _create_specialized_retrievers(self):
        """Create specialized retrievers for different query types."""
        # Compression retriever for explanatory queries
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.semantic_retriever
        )
        
        # MMR retriever for diverse results
        self.mmr_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
        )
        
        # Ensemble retriever for combining semantic and keyword search
        if self.keyword_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.semantic_retriever, self.keyword_retriever],
                weights=[0.7, 0.3]
            )
        else:
            self.ensemble_retriever = self.semantic_retriever
        
        # Self-query retriever for metadata filtering
        if self.metadata_field_info:
            self.self_query_retriever = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                document_contents="Documents about various topics",
                metadata_field_info=self.metadata_field_info,
                verbose=self.verbose
            )
        else:
            self.self_query_retriever = self.semantic_retriever
        
        # Create strategy mapping
        self.strategies = {
            "semantic": self.semantic_retriever,
            "keyword": self.keyword_retriever if self.keyword_retriever else self.semantic_retriever,
            "compression": self.compression_retriever,
            "mmr": self.mmr_retriever,
            "ensemble": self.ensemble_retriever,
            "self_query": self.self_query_retriever
        }
    
    def _create_query_analyzer(self):
        """Create the query analyzer for understanding query characteristics."""
        self.analyzer_prompt = ChatPromptTemplate.from_template("""
        Analyze the following query and extract:
        1. Query type (factual, conceptual, procedural, comparative, exploratory)
        2. Metadata filters (if any)
        3. Complexity level (simple, moderate, complex)
        
        Query: {query}
        
        Analysis (JSON format):
        """)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine its characteristics.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query analysis
        """
        # Invoke the LLM
        response = self.llm.invoke(
            self.analyzer_prompt.format(query=query)
        )
        
        # Try to parse as JSON
        try:
            analysis = json.loads(response.content.strip())
        except json.JSONDecodeError:
            # Fallback to regex parsing
            analysis = self._parse_analysis_text(response.content)
        
        if self.verbose:
            print(f"Query: {query}")
            print(f"Analysis: {analysis}")
        
        return analysis
    
    def _parse_analysis_text(self, text: str) -> Dict[str, Any]:
        """Parse analysis text when JSON parsing fails.
        
        Args:
            text: Analysis text from LLM
            
        Returns:
            Structured analysis
        """
        analysis = {
            "query_type": "factual",  # Default
            "metadata_filters": {},
            "complexity": "simple"  # Default
        }
        
        # Extract query type
        query_type_match = re.search(r"Query type:?\s*(\w+)", text, re.IGNORECASE)
        if query_type_match:
            analysis["query_type"] = query_type_match.group(1).lower()
        
        # Extract complexity
        complexity_match = re.search(r"Complexity:?\s*(\w+)", text, re.IGNORECASE)
        if complexity_match:
            analysis["complexity"] = complexity_match.group(1).lower()
        
        # Extract metadata filters
        filters = {}
        filter_section = re.search(r"Metadata filters:?\s*(.+?)(?:\n\d\.|\Z)", text, re.DOTALL)
        if filter_section:
            filter_text = filter_section.group(1)
            
            # Look for common metadata fields
            for field in ["source", "author", "date", "topic"]:
                field_match = re.search(rf"{field}:?\s*([^,\n]+)", filter_text, re.IGNORECASE)
                if field_match:
                    filters[field] = field_match.group(1).strip()
        
        analysis["metadata_filters"] = filters
        
        return analysis
    
    def select_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select retrieval strategy based on query analysis.
        
        Args:
            analysis: Query analysis
            
        Returns:
            Selected strategy
        """
        query_type = analysis.get("query_type", "factual")
        complexity = analysis.get("complexity", "simple")
        has_filters = bool(analysis.get("metadata_filters", {}))
        
        # Select strategy based on query characteristics
        if has_filters:
            return "self_query"  # Use self-query for metadata filtering
        elif query_type == "conceptual":
            return "compression"  # Use compression for explanatory queries
        elif query_type == "comparative":
            return "ensemble"  # Use ensemble for comparative queries
        elif query_type == "exploratory":
            return "mmr"  # Use MMR for exploratory queries
        elif complexity == "complex":
            return "ensemble"  # Use ensemble for complex queries
        else:
            return "semantic"  # Default to semantic search
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents using adaptive strategies.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved documents
        """
        # Analyze the query
        analysis = self.analyze_query(query)
        
        # Select strategy
        strategy = self.select_strategy(analysis)
        
        if self.verbose:
            print(f"Selected strategy: {strategy}")
        
        # Get the appropriate retriever
        retriever = self.strategies.get(strategy, self.strategies["semantic"])
        
        # Apply metadata filters for self-query
        if strategy == "self_query":
            # Self-query retriever handles filters automatically
            return retriever.get_relevant_documents(query)
        else:
            # Regular retrieval
            return retriever.get_relevant_documents(query)
    
    def as_lcel_chain(self):
        """Return the adaptive RAG system as an LCEL chain."""
        return (
            {"query": RunnablePassthrough()}
            | RunnableLambda(lambda x: {
                "query": x["query"],
                "analysis": self.analyze_query(x["query"])
            })
            | RunnableLambda(lambda x: {
                "query": x["query"],
                "strategy": self.select_strategy(x["analysis"])
            })
            | RunnableLambda(lambda x: self.strategies.get(
                x["strategy"], 
                self.strategies["semantic"]
            ).get_relevant_documents(x["query"]))
        )


# Example usage
if __name__ == "__main__":
    print("Self-Querying and Adaptive RAG Systems")
    
    # This is just an example and won't run without the necessary dependencies
    # and document data. See the lesson4_exercises.py file for working examples.
