"""
RAG System Implementation
------------------------
This module implements a simple Retrieval-Augmented Generation (RAG) system
that combines vector database retrieval with LLM-based generation.
"""

import os
import sys
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with 'pip install faiss-cpu' for better vector search.")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with 'pip install chromadb' for persistent vector storage.")

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


class SimpleRAGSystem:
    """A simple RAG system with vector database integration."""
    
    def __init__(self, documents, embeddings, vector_store_type="faiss"):
        """
        Initialize the RAG system.
        
        Args:
            documents: List of document chunks
            embeddings: List of embedding vectors for the chunks
            vector_store_type: Type of vector database to use ("faiss" or "chroma")
        """
        self.documents = documents
        self.vector_store_type = vector_store_type
        
        # Initialize vector database
        if vector_store_type == "faiss":
            self.vector_db = self._init_faiss(documents, embeddings)
        elif vector_store_type == "chroma":
            self.vector_db = self._init_chroma(documents, embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
    
    def _init_faiss(self, documents, embeddings):
        """Initialize a FAISS vector database."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with 'pip install faiss-cpu'")
        
        import faiss
        import numpy as np
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        return {
            "index": index,
            "documents": documents,
            "embeddings": embeddings_array
        }
    
    def _init_chroma(self, documents, embeddings):
        """Initialize a ChromaDB vector database."""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with 'pip install chromadb'")
        
        import chromadb
        
        # Create ChromaDB client
        client = chromadb.Client()
        collection = client.create_collection("documents")
        
        # Add documents to collection
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            collection.add(
                ids=[f"doc_{i}"],
                embeddings=[embedding],
                metadatas=[doc.get("metadata", {})]
            )
        
        return {
            "collection": collection,
            "documents": documents
        }
    
    def _faiss_search(self, query_embedding, top_k=5):
        """Search using FAISS."""
        # Convert query embedding to numpy array
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.vector_db["index"].search(query_embedding, top_k)
        
        # Get results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.vector_db["documents"]):
                doc = self.vector_db["documents"][idx]
                results.append({
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(1.0 / (1.0 + dist))  # Convert distance to similarity score
                })
        
        return results
    
    def _chroma_search(self, query_embedding, top_k=5):
        """Search using ChromaDB."""
        # Search
        results = self.vector_db["collection"].query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Get results
        output = []
        for i, (id, distance, metadata) in enumerate(zip(
            results["ids"][0], results["distances"][0], results["metadatas"][0]
        )):
            # Get the original document
            idx = int(id.split("_")[1])
            doc = self.vector_db["documents"][idx]
            
            output.append({
                "content": doc.get("content", ""),
                "metadata": metadata,
                "score": float(1.0 / (1.0 + distance))  # Convert distance to similarity score
            })
        
        return output
    
    def _keyword_search(self, query, top_k=5):
        """Simple keyword-based search."""
        # Tokenize query
        query_terms = set(query.lower().split())
        
        # Calculate scores based on term frequency
        results = []
        for i, doc in enumerate(self.documents):
            content = doc.get("content", "").lower()
            score = sum(1 for term in query_terms if term in content)
            
            if score > 0:
                results.append({
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": score / len(query_terms)  # Normalize score
                })
        
        # Sort by score and limit to top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _merge_results(self, semantic_results, keyword_results, alpha=0.7):
        """Merge semantic and keyword search results."""
        # Create a dictionary to store merged results
        merged = {}
        
        # Add semantic results with weight alpha
        for result in semantic_results:
            content = result["content"]
            if content not in merged:
                merged[content] = {
                    "content": content,
                    "metadata": result["metadata"],
                    "score": alpha * result["score"],
                    "sources": ["semantic"]
                }
            else:
                merged[content]["score"] += alpha * result["score"]
                if "semantic" not in merged[content]["sources"]:
                    merged[content]["sources"].append("semantic")
        
        # Add keyword results with weight (1-alpha)
        for result in keyword_results:
            content = result["content"]
            if content not in merged:
                merged[content] = {
                    "content": content,
                    "metadata": result["metadata"],
                    "score": (1 - alpha) * result["score"],
                    "sources": ["keyword"]
                }
            else:
                merged[content]["score"] += (1 - alpha) * result["score"]
                if "keyword" not in merged[content]["sources"]:
                    merged[content]["sources"].append("keyword")
        
        # Convert to list and sort by score
        results = list(merged.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def retrieve(self, query, embedding_model, top_k=5, use_hybrid=False):
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User question
            embedding_model: Model to generate query embedding
            top_k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid retrieval (semantic + keyword)
            
        Returns:
            List of relevant document chunks
        """
        # Generate query embedding
        query_embedding = embedding_model.embed_text(query)
        
        # Semantic search
        if self.vector_store_type == "faiss":
            results = self._faiss_search(query_embedding, top_k)
        else:
            results = self._chroma_search(query_embedding, top_k)
        
        # Hybrid search (combine with keyword search)
        if use_hybrid:
            keyword_results = self._keyword_search(query, top_k)
            results = self._merge_results(results, keyword_results)
        
        return results[:top_k]
    
    def augment_query(self, query, context=None):
        """
        Augment the query with additional context.
        
        Args:
            query: Original user question
            context: Additional context (e.g., conversation history)
            
        Returns:
            Augmented query
        """
        if not context:
            return query
        
        # Simple augmentation: combine query with context
        augmented_query = f"Context: {context}\n\nQuestion: {query}"
        
        return augmented_query
    
    def generate_answer(self, query, retrieved_chunks, llm_client):
        """
        Generate an answer based on retrieved chunks.
        
        Args:
            query: User question
            retrieved_chunks: Relevant document chunks
            llm_client: LLM client for text generation
            
        Returns:
            Generated answer
        """
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["content"] for chunk in retrieved_chunks])
        
        # Create prompt for the LLM
        prompt = f"""
        Answer the following question based on the provided context.
        If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        # Generate answer
        response = llm_client.generate_text(prompt)
        answer = llm_client.extract_text_from_response(response)
        
        return answer
    
    def analyze_question(self, question):
        """
        Analyze a question to identify its type and key entities.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with question analysis
        """
        # Identify question type
        question_types = {
            "what": "definition",
            "who": "person",
            "when": "time",
            "where": "location",
            "why": "reason",
            "how": "process"
        }
        
        question_lower = question.lower()
        question_type = "general"
        
        for q_word, q_type in question_types.items():
            if question_lower.startswith(q_word):
                question_type = q_type
                break
        
        # Extract key entities (simplified)
        words = question_lower.split()
        stop_words = {"what", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an"}
        entities = [word for word in words if word not in stop_words and len(word) > 3]
        
        return {
            "type": question_type,
            "entities": entities,
            "is_metadata_query": self._is_metadata_query(question)
        }
    
    def expand_query(self, question, analysis):
        """
        Expand a query to improve retrieval.
        
        Args:
            question: Original question
            analysis: Question analysis
            
        Returns:
            List of expanded queries
        """
        expanded_queries = [question]  # Start with original question
        
        # Add variations based on question type
        if analysis["type"] == "definition":
            expanded_queries.append(f"definition of {' '.join(analysis['entities'])}")
            expanded_queries.append(f"what is {' '.join(analysis['entities'])}")
        elif analysis["type"] == "process":
            expanded_queries.append(f"steps for {' '.join(analysis['entities'])}")
            expanded_queries.append(f"process of {' '.join(analysis['entities'])}")
        
        # Add entity-focused queries
        for entity in analysis["entities"]:
            expanded_queries.append(entity)
        
        return expanded_queries
    
    def retrieve_from_multiple_documents(self, query, embedding_model, top_k=3, docs_per_source=2):
        """
        Retrieve information from multiple documents.
        
        Args:
            query: User question
            embedding_model: Model to generate query embedding
            top_k: Number of total chunks to retrieve
            docs_per_source: Maximum chunks per document source
            
        Returns:
            List of relevant chunks from different documents
        """
        # Get all relevant chunks
        all_chunks = self.retrieve(query, embedding_model, top_k=top_k*2)
        
        # Group by document source
        chunks_by_source = {}
        for chunk in all_chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        # Select top chunks from each source
        balanced_chunks = []
        for source, chunks in chunks_by_source.items():
            balanced_chunks.extend(chunks[:docs_per_source])
        
        # Sort by relevance and limit to top_k
        balanced_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        return balanced_chunks[:top_k]
    
    def synthesize_information(self, query, chunks, llm_client):
        """
        Synthesize information from multiple chunks.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            llm_client: LLM client for text generation
            
        Returns:
            Synthesized information
        """
        # Prepare context from chunks
        contexts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("metadata", {}).get("source", f"Source {i+1}")
            contexts.append(f"[{source}]: {chunk['content']}")
        
        context_text = "\n\n".join(contexts)
        
        # Create synthesis prompt
        prompt = f"""
        Synthesize information from the following sources to answer the question.
        If the sources contain conflicting information, acknowledge the differences.
        
        Question: {query}
        
        Sources:
        {context_text}
        
        Synthesized Answer:
        """
        
        # Generate synthesized answer
        response = llm_client.generate_text(prompt)
        answer = llm_client.extract_text_from_response(response)
        
        return answer
    
    def track_sources(self, chunks):
        """
        Track sources for retrieved chunks.
        
        Args:
            chunks: Retrieved document chunks
            
        Returns:
            Dictionary mapping content to sources
        """
        source_map = {}
        
        for chunk in chunks:
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            
            source = {
                "document": metadata.get("source", "Unknown"),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "score": chunk.get("score", 0)
            }
            
            if content in source_map:
                source_map[content].append(source)
            else:
                source_map[content] = [source]
        
        return source_map
    
    def generate_answer_with_citations(self, query, chunks, llm_client):
        """
        Generate an answer with citations.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            llm_client: LLM client for text generation
            
        Returns:
            Answer with citations
        """
        # Track sources
        source_map = self.track_sources(chunks)
        
        # Prepare context with source identifiers
        contexts = []
        for i, chunk in enumerate(chunks):
            contexts.append(f"[{i+1}] {chunk['content']}")
        
        context_text = "\n\n".join(contexts)
        
        # Create prompt for answer with citations
        prompt = f"""
        Answer the following question based on the provided sources.
        Use citation numbers [1], [2], etc. to indicate which source supports each part of your answer.
        
        Question: {query}
        
        Sources:
        {context_text}
        
        Answer with citations:
        """
        
        # Generate answer
        response = llm_client.generate_text(prompt)
        answer = llm_client.extract_text_from_response(response)
        
        # Add source details at the end
        sources_text = "\n\nSources:\n"
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "")
            page_info = f", page {page}" if page else ""
            sources_text += f"[{i+1}] {source}{page_info}\n"
        
        return answer + sources_text
    
    def _calculate_relevance(self, query, chunks):
        """Calculate relevance score based on semantic similarity."""
        if not chunks:
            return 0.0
        
        # Use the average score as the relevance measure
        avg_score = sum(chunk.get("score", 0) for chunk in chunks) / len(chunks)
        return min(1.0, avg_score)
    
    def _calculate_consistency(self, chunks):
        """Calculate consistency score based on agreement between chunks."""
        if not chunks or len(chunks) < 2:
            return 1.0  # Single source is consistent with itself
        
        # Simple heuristic: check if chunks are from the same source
        sources = [chunk.get("metadata", {}).get("source", "") for chunk in chunks]
        unique_sources = len(set(sources))
        
        if unique_sources == 1:
            return 1.0  # All from same source
        
        # More sources means potentially less consistency
        return max(0.5, 1.0 - (unique_sources - 1) / len(chunks))
    
    def _calculate_coverage(self, query, chunks):
        """Calculate coverage score based on how well chunks cover the query terms."""
        if not chunks:
            return 0.0
        
        # Extract query terms
        query_terms = set(term.lower() for term in query.split() if len(term) > 3)
        if not query_terms:
            return 0.5  # No significant terms to match
        
        # Count how many query terms appear in the chunks
        covered_terms = set()
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            for term in query_terms:
                if term in content:
                    covered_terms.add(term)
        
        # Calculate coverage ratio
        coverage_ratio = len(covered_terms) / len(query_terms)
        return coverage_ratio
    
    def _calculate_source_quality(self, chunks):
        """Calculate source quality score based on metadata."""
        if not chunks:
            return 0.0
        
        # Simple heuristic: prefer sources with more metadata
        quality_scores = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            
            # More metadata fields suggests higher quality
            metadata_count = len(metadata)
            
            # Specific quality indicators
            has_author = 1 if "author" in metadata else 0
            has_date = 1 if "date" in metadata else 0
            has_title = 1 if "title" in metadata else 0
            
            # Calculate quality score (0-1)
            quality = min(1.0, (metadata_count + has_author + has_date + has_title) / 10)
            quality_scores.append(quality)
        
        # Return average quality
        return sum(quality_scores) / len(quality_scores)
    
    def assess_confidence(self, query, chunks):
        """
        Assess confidence in the answer.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            
        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Factors affecting confidence
        factors = {
            "relevance": self._calculate_relevance(query, chunks),
            "consistency": self._calculate_consistency(chunks),
            "coverage": self._calculate_coverage(query, chunks),
            "source_quality": self._calculate_source_quality(chunks)
        }
        
        # Weighted average of factors
        weights = {
            "relevance": 0.4,
            "consistency": 0.2,
            "coverage": 0.3,
            "source_quality": 0.1
        }
        
        confidence = sum(score * weights[factor] for factor, score in factors.items())
        
        return min(1.0, max(0.0, confidence))
    
    def generate_response_with_uncertainty(self, query, chunks, confidence, llm_client):
        """
        Generate a response that communicates uncertainty appropriately.
        
        Args:
            query: User question
            chunks: Retrieved chunks
            confidence: Confidence score (0-1)
            llm_client: LLM client for text generation
            
        Returns:
            Response with appropriate uncertainty language
        """
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "high"
            prefix = "Based on the available information, "
        elif confidence >= 0.5:
            confidence_level = "medium"
            prefix = "The information suggests that "
        else:
            confidence_level = "low"
            prefix = "I'm not entirely certain, but based on limited information, "
        
        # Generate base answer
        base_answer = self.generate_answer(query, chunks, llm_client)
        
        # Add confidence prefix
        if confidence < 0.3 and "I don't have enough information" not in base_answer:
            response = f"{prefix}{base_answer}\n\nPlease note that the available information on this topic is limited."
        elif confidence < 0.8 and "I don't have enough information" not in base_answer:
            response = f"{prefix}{base_answer}"
        else:
            response = base_answer
        
        return response
    
    def _is_metadata_query(self, question):
        """
        Determine if a question is about document metadata.
        
        Args:
            question: User question
            
        Returns:
            Boolean indicating if this is a metadata query
        """
        metadata_keywords = {
            "author", "wrote", "written", "published", "publication", 
            "date", "year", "when was", "how old", "recent",
            "title", "called", "named", "file", "document",
            "type", "format", "source", "where from", "origin"
        }
        
        question_lower = question.lower()
        
        return any(keyword in question_lower for keyword in metadata_keywords)
    
    def retrieve_metadata(self, query, metadata_fields=None):
        """
        Retrieve document metadata based on a query.
        
        Args:
            query: User question about metadata
            metadata_fields: Specific metadata fields to search
            
        Returns:
            List of relevant metadata entries
        """
        # Default metadata fields to search
        if metadata_fields is None:
            metadata_fields = ["author", "title", "date", "source", "type"]
        
        # Extract key terms from query
        query_terms = set(query.lower().split())
        stop_words = {"what", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an"}
        query_terms = query_terms - stop_words
        
        # Search for documents with matching metadata
        results = []
        
        for i, doc in enumerate(self.documents):
            metadata = doc.get("metadata", {})
            
            # Calculate a simple relevance score
            score = 0
            matched_fields = []
            
            for field in metadata_fields:
                if field in metadata:
                    field_value = str(metadata[field]).lower()
                    
                    # Check if any query term is in the metadata value
                    for term in query_terms:
                        if term in field_value:
                            score += 1
                            matched_fields.append(field)
            
            if score > 0:
                results.append({
                    "document_id": i,
                    "metadata": metadata,
                    "score": score,
                    "matched_fields": matched_fields
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def answer_metadata_query(self, query, metadata_results, llm_client):
        """
        Generate an answer for a metadata query.
        
        Args:
            query: User question about metadata
            metadata_results: Retrieved metadata
            llm_client: LLM client for text generation
            
        Returns:
            Answer about document metadata
        """
        if not metadata_results:
            return "I couldn't find any documents with metadata matching your query."
        
        # Prepare metadata context
        metadata_context = []
        for i, result in enumerate(metadata_results[:5]):  # Limit to top 5
            metadata = result["metadata"]
            metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
            metadata_context.append(f"Document {i+1}: {metadata_str}")
        
        metadata_text = "\n".join(metadata_context)
        
        # Create prompt for metadata answer
        prompt = f"""
        Answer the following question about document metadata.
        
        Question: {query}
        
        Document Metadata:
        {metadata_text}
        
        Answer:
        """
        
        # Generate answer
        response = llm_client.generate_text(prompt)
        answer = llm_client.extract_text_from_response(response)
        
        return answer


# Simple LLM client for testing when GroqClient is not available
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
        return response["choices"][0]["message"]["content"]


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        {
            "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs with external knowledge.",
            "metadata": {
                "source": "RAG Paper",
                "author": "Meta AI",
                "date": "2023-01-01"
            }
        },
        {
            "content": "Vector databases store embeddings for efficient similarity search.",
            "metadata": {
                "source": "Vector DB Guide",
                "author": "Database Experts",
                "date": "2022-05-15"
            }
        }
    ]
    
    # Sample embeddings (simplified for demonstration)
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ]
    
    # Create RAG system
    try:
        rag_system = SimpleRAGSystem(documents, embeddings, vector_store_type="faiss")
        print("RAG system initialized successfully.")
        
        # Create LLM client
        if GROQ_AVAILABLE:
            llm_client = GroqClient()
            print("Using GroqClient for text generation.")
        else:
            llm_client = SimpleLLMClient()
            print("Using SimpleLLMClient for text generation (simulated responses).")
        
        # Simple embedding model for testing
        class SimpleEmbedding:
            def embed_text(self, text):
                # Very simple embedding function for testing
                return [0.1, 0.2, 0.3, 0.4]
        
        embedding_model = SimpleEmbedding()
        
        # Test retrieval
        query = "What is RAG?"
        results = rag_system.retrieve(query, embedding_model, top_k=1)
        print(f"\nQuery: {query}")
        print(f"Retrieved: {results[0]['content']}")
        
        # Test answer generation
        answer = rag_system.generate_answer(query, results, llm_client)
        print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"Error: {e}")
