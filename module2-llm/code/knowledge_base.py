"""
Knowledge Base Implementation with LLM Integration
----------------------------------------------
This file contains implementations of a knowledge base system that uses
vector embeddings for semantic search and retrieval, enhanced with
real LLM integration using the Groq API.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
import uuid

# Import our modules
try:
    # When running from the module2-llm/code directory
    from groq_client import GroqClient
    from vector_store import EnhancedVectorDB
except ImportError:
    # When running from the project root
    from module2_llm.code.groq_client import GroqClient
    from module2_llm.code.vector_store import EnhancedVectorDB


class KnowledgeBase:
    """
    A knowledge base system that uses vector embeddings for semantic search
    and retrieval, enhanced with real LLM integration.
    """
    
    def __init__(self, storage_dir: str = "knowledge_base", vector_db: Optional[EnhancedVectorDB] = None):
        """
        Initialize the knowledge base
        
        Args:
            storage_dir (str): Directory for persistent storage
            vector_db (EnhancedVectorDB, optional): Vector database to use
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize vector database
        self.vector_db = vector_db or EnhancedVectorDB()
        
        # Initialize Groq client
        self.groq_client = GroqClient()
        
        # Load or initialize metadata
        self.metadata_path = os.path.join(storage_dir, "metadata.json")
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'item_count': 0,
                'categories': {},
                'sources': {}
            }
            self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def add_item(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add an item to the knowledge base
        
        Args:
            text (str): The text to add
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: ID of the added item
        """
        # Prepare metadata
        item_metadata = metadata or {}
        
        # Add timestamp if not provided
        if 'timestamp' not in item_metadata:
            item_metadata['timestamp'] = datetime.now().isoformat()
        
        # Add source if not provided
        if 'source' not in item_metadata:
            item_metadata['source'] = 'user'
        
        # Add category if not provided
        if 'category' not in item_metadata:
            # Use LLM to categorize the text
            item_metadata['category'] = self._categorize_text(text)
        
        # Add the item to the vector database
        item_id = self.vector_db.add(text, item_metadata)
        
        # Update metadata
        self.metadata['item_count'] += 1
        
        # Update category statistics
        category = item_metadata.get('category', 'uncategorized')
        if category not in self.metadata['categories']:
            self.metadata['categories'][category] = 0
        self.metadata['categories'][category] += 1
        
        # Update source statistics
        source = item_metadata.get('source', 'unknown')
        if source not in self.metadata['sources']:
            self.metadata['sources'][source] = 0
        self.metadata['sources'][source] += 1
        
        # Save metadata
        self._save_metadata()
        
        # Save vector database if it has a save method
        if hasattr(self.vector_db, 'save'):
            vector_db_path = os.path.join(self.storage_dir, "vector_db.json")
            self.vector_db.save(vector_db_path)
        
        return item_id
    
    def _categorize_text(self, text: str) -> str:
        """
        Categorize text using LLM
        
        Args:
            text (str): The text to categorize
            
        Returns:
            str: The category
        """
        # Get existing categories
        existing_categories = list(self.metadata['categories'].keys())
        categories_text = ", ".join(existing_categories) if existing_categories else "none yet"
        
        prompt = f"""
        Categorize the following text into a single category.
        If it fits into one of the existing categories ({categories_text}), use that.
        Otherwise, create a new, concise category name (1-3 words).
        
        Text: "{text}"
        
        Return only the category name, without quotes or additional text.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=50)
            category = self.groq_client.extract_text_from_response(response).strip()
            return category or "uncategorized"
        except Exception as e:
            # Return a default category if categorization fails
            print(f"Categorization failed: {e}")
            return "uncategorized"
    
    def add_items_batch(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add multiple items to the knowledge base
        
        Args:
            texts (list): List of texts to add
            metadatas (list, optional): List of metadata dictionaries
            
        Returns:
            list: IDs of the added items
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Prepare metadatas
        for i, metadata in enumerate(metadatas):
            # Add timestamp if not provided
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.now().isoformat()
            
            # Add source if not provided
            if 'source' not in metadata:
                metadata['source'] = 'user'
            
            # Add category if not provided
            if 'category' not in metadata:
                # Use LLM to categorize the text
                metadata['category'] = self._categorize_text(texts[i])
        
        # Add the items to the vector database
        item_ids = self.vector_db.add_batch(texts, metadatas)
        
        # Update metadata
        self.metadata['item_count'] += len(texts)
        
        # Update category and source statistics
        for metadata in metadatas:
            category = metadata.get('category', 'uncategorized')
            if category not in self.metadata['categories']:
                self.metadata['categories'][category] = 0
            self.metadata['categories'][category] += 1
            
            source = metadata.get('source', 'unknown')
            if source not in self.metadata['sources']:
                self.metadata['sources'][source] = 0
            self.metadata['sources'][source] += 1
        
        # Save metadata
        self._save_metadata()
        
        # Save vector database if it has a save method
        if hasattr(self.vector_db, 'save'):
            vector_db_path = os.path.join(self.storage_dir, "vector_db.json")
            self.vector_db.save(vector_db_path)
        
        return item_ids
    
    def retrieve(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve items from the knowledge base
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            list: Relevant items
        """
        # Use query expansion for better recall
        results = self.vector_db.search_with_expansion(query, top_k=top_k*2)  # Get more results to filter
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result['similarity'] >= min_similarity
        ]
        
        # Sort by similarity and take top k
        filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
        return filtered_results[:top_k]
    
    def retrieve_with_explanation(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve items with an explanation of the retrieval process
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            
        Returns:
            dict: Retrieval results and explanation
        """
        # Get expanded queries
        expanded_queries = self.vector_db.expand_query(query)
        
        # Retrieve with each expanded query
        all_results = []
        for expanded_query in expanded_queries:
            results = self.vector_db.search(expanded_query, top_k=top_k)
            for result in results:
                result['expanded_query'] = expanded_query
            all_results.extend(results)
        
        # Remove duplicates and sort
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = unique_results[:top_k]
        
        # Generate explanation
        explanation = self._generate_retrieval_explanation(query, expanded_queries, top_results)
        
        return {
            'query': query,
            'expanded_queries': expanded_queries,
            'results': top_results,
            'explanation': explanation
        }
    
    def _generate_retrieval_explanation(self, query: str, expanded_queries: List[str], results: List[Dict[str, Any]]) -> str:
        """
        Generate an explanation of the retrieval process using LLM
        
        Args:
            query (str): The original query
            expanded_queries (list): The expanded queries
            results (list): The retrieved results
            
        Returns:
            str: Explanation of the retrieval process
        """
        # Format inputs for the LLM
        expanded_text = "\n".join([f"- {q}" for q in expanded_queries])
        
        results_text = "\n\n".join([
            f"Result {i+1} (Similarity: {result['similarity']:.4f}):\n{result['text'][:100]}..."
            for i, result in enumerate(results)
        ])
        
        prompt = f"""
        Explain the retrieval process for this query:
        
        Original query: "{query}"
        
        Step 1: The query was expanded to improve recall:
        {expanded_text}
        
        Step 2: The system searched for relevant information using these queries and found:
        {results_text}
        
        Provide a clear explanation of how the retrieval process worked, including:
        - How query expansion helped find relevant information
        - Why the top results were selected
        - Any challenges or limitations in the retrieval process
        
        Explanation:
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=300)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Return a simple explanation if generation fails
            print(f"Explanation generation failed: {e}")
            return f"The system expanded the query '{query}' to find the most relevant information in the knowledge base."
    
    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an item by ID
        
        Args:
            item_id (str): ID of the item to get
            
        Returns:
            dict or None: The item, or None if not found
        """
        return self.vector_db.get_item(item_id)
    
    def update_item(self, item_id: str, text: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update an item in the knowledge base
        
        Args:
            item_id (str): ID of the item to update
            text (str, optional): New text for the item
            metadata (dict, optional): New or updated metadata
            
        Returns:
            bool: True if the item was updated, False otherwise
        """
        # Get the current item
        item = self.get_item(item_id)
        if not item:
            return False
        
        # Update category statistics if category is changing
        if metadata and 'category' in metadata and metadata['category'] != item.get('metadata', {}).get('category'):
            old_category = item.get('metadata', {}).get('category', 'uncategorized')
            new_category = metadata['category']
            
            if old_category in self.metadata['categories']:
                self.metadata['categories'][old_category] -= 1
                if self.metadata['categories'][old_category] <= 0:
                    del self.metadata['categories'][old_category]
            
            if new_category not in self.metadata['categories']:
                self.metadata['categories'][new_category] = 0
            self.metadata['categories'][new_category] += 1
        
        # Update source statistics if source is changing
        if metadata and 'source' in metadata and metadata['source'] != item.get('metadata', {}).get('source'):
            old_source = item.get('metadata', {}).get('source', 'unknown')
            new_source = metadata['source']
            
            if old_source in self.metadata['sources']:
                self.metadata['sources'][old_source] -= 1
                if self.metadata['sources'][old_source] <= 0:
                    del self.metadata['sources'][old_source]
            
            if new_source not in self.metadata['sources']:
                self.metadata['sources'][new_source] = 0
            self.metadata['sources'][new_source] += 1
        
        # Update the item in the vector database
        updated = self.vector_db.update(item_id, text, metadata)
        
        if updated:
            # Save metadata
            self._save_metadata()
            
            # Save vector database if it has a save method
            if hasattr(self.vector_db, 'save'):
                vector_db_path = os.path.join(self.storage_dir, "vector_db.json")
                self.vector_db.save(vector_db_path)
        
        return updated
    
    def delete_item(self, item_id: str) -> bool:
        """
        Delete an item from the knowledge base
        
        Args:
            item_id (str): ID of the item to delete
            
        Returns:
            bool: True if the item was deleted, False otherwise
        """
        # Get the current item
        item = self.get_item(item_id)
        if not item:
            return False
        
        # Update category statistics
        category = item.get('metadata', {}).get('category', 'uncategorized')
        if category in self.metadata['categories']:
            self.metadata['categories'][category] -= 1
            if self.metadata['categories'][category] <= 0:
                del self.metadata['categories'][category]
        
        # Update source statistics
        source = item.get('metadata', {}).get('source', 'unknown')
        if source in self.metadata['sources']:
            self.metadata['sources'][source] -= 1
            if self.metadata['sources'][source] <= 0:
                del self.metadata['sources'][source]
        
        # Delete the item from the vector database
        deleted = self.vector_db.delete(item_id)
        
        if deleted:
            # Update metadata
            self.metadata['item_count'] -= 1
            
            # Save metadata
            self._save_metadata()
            
            # Save vector database if it has a save method
            if hasattr(self.vector_db, 'save'):
                vector_db_path = os.path.join(self.storage_dir, "vector_db.json")
                self.vector_db.save(vector_db_path)
        
        return deleted
    
    def get_all_items(self) -> List[Dict[str, Any]]:
        """
        Get all items in the knowledge base
        
        Returns:
            list: All items in the knowledge base
        """
        return self.vector_db.get_all()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            dict: Statistics about the knowledge base
        """
        return {
            'item_count': self.metadata['item_count'],
            'categories': self.metadata['categories'],
            'sources': self.metadata['sources'],
            'created_at': self.metadata['created_at'],
            'last_updated': self.metadata['last_updated']
        }
    
    def cluster_items(self, num_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster items in the knowledge base
        
        Args:
            num_clusters (int): Number of clusters to create
            
        Returns:
            dict: Clusters of items with labels
        """
        # Cluster items
        clusters = self.vector_db.cluster_items(num_clusters)
        
        # Generate labels for clusters
        labels = self.vector_db.get_cluster_labels(clusters)
        
        # Format the result
        result = {
            'num_clusters': len(clusters),
            'clusters': {}
        }
        
        for cluster_id, items in clusters.items():
            result['clusters'][cluster_id] = {
                'label': labels.get(cluster_id, f"Cluster {cluster_id}"),
                'size': len(items),
                'items': items
            }
        
        return result
    
    def generate_summary(self) -> str:
        """
        Generate a summary of the knowledge base using LLM
        
        Returns:
            str: Summary of the knowledge base
        """
        # Get statistics
        stats = self.get_statistics()
        
        # Get sample items from each category
        samples = {}
        all_items = self.get_all_items()
        
        for category in stats['categories']:
            category_items = [
                item for item in all_items
                if item.get('metadata', {}).get('category') == category
            ]
            samples[category] = category_items[:3]  # Take up to 3 samples per category
        
        # Format inputs for the LLM
        stats_text = json.dumps(stats, indent=2)
        
        samples_text = ""
        for category, items in samples.items():
            samples_text += f"\nCategory: {category}\n"
            for i, item in enumerate(items):
                samples_text += f"- {item['text'][:100]}...\n"
        
        prompt = f"""
        Generate a comprehensive summary of this knowledge base:
        
        Statistics:
        {stats_text}
        
        Sample items:
        {samples_text}
        
        Provide a summary that includes:
        - Overview of the knowledge base content and size
        - Description of the main categories and their relative importance
        - Insights about the knowledge distribution
        - Any notable patterns or gaps in the knowledge
        
        Summary:
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Return a simple summary if generation fails
            print(f"Summary generation failed: {e}")
            return f"This knowledge base contains {stats['item_count']} items across {len(stats['categories'])} categories."


class CitationManager:
    """
    Manager for adding citations to responses.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize the citation manager
        
        Args:
            knowledge_base (KnowledgeBase): The knowledge base to use for citations
        """
        self.knowledge_base = knowledge_base
        self.groq_client = GroqClient()
    
    def add_citations_to_response(
        self, 
        response: str, 
        sources: List[Dict[str, Any]], 
        citation_style: str = "standard"
    ) -> str:
        """
        Add citations to a response
        
        Args:
            response (str): The response to add citations to
            sources (list): The sources to cite
            citation_style (str): The citation style to use
            
        Returns:
            str: The response with citations
        """
        if not sources:
            return response
        
        # Format sources for the LLM
        sources_text = "\n\n".join([
            f"Source {i+1}:\nText: {source['text']}\nMetadata: {json.dumps(source.get('metadata', {}))}"
            for i, source in enumerate(sources)
        ])
        
        prompt = f"""
        Add citations to this response according to the {citation_style} citation style:
        
        Response:
        {response}
        
        Sources:
        {sources_text}
        
        Return the response with appropriate citations added. The citations should reference the specific sources used.
        If a part of the response doesn't come from any source, don't add a citation for that part.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=1000)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Add simple citations if LLM fails
            print(f"Citation generation failed: {e}")
            cited_response = response
            
            # Add a simple citation section at the end
            cited_response += "\n\nSources:\n"
            for i, source in enumerate(sources):
                source_info = source.get('metadata', {}).get('source', 'unknown')
                cited_response += f"[{i+1}] {source_info}\n"
            
            return cited_response
    
    def generate_bibliography(self, sources: List[Dict[str, Any]], style: str = "standard") -> str:
        """
        Generate a bibliography for a set of sources
        
        Args:
            sources (list): The sources to include in the bibliography
            style (str): The citation style to use
            
        Returns:
            str: The generated bibliography
        """
        if not sources:
            return "No sources to cite."
        
        # Format sources for the LLM
        sources_text = "\n\n".join([
            f"Source {i+1}:\nText: {source['text']}\nMetadata: {json.dumps(source.get('metadata', {}))}"
            for i, source in enumerate(sources)
        ])
        
        prompt = f"""
        Generate a bibliography in {style} style for these sources:
        
        {sources_text}
        
        Return only the formatted bibliography entries, one per line.
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Generate a simple bibliography if LLM fails
            print(f"Bibliography generation failed: {e}")
            bibliography = "Bibliography:\n"
            for i, source in enumerate(sources):
                metadata = source.get('metadata', {})
                source_info = metadata.get('source', 'unknown')
                date = metadata.get('date', 'n.d.')
                bibliography += f"{i+1}. {source_info} ({date})\n"
            
            return bibliography


class UncertaintyHandler:
    """
    Handler for managing uncertainty in knowledge base responses.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the uncertainty handler
        
        Args:
            confidence_threshold (float): Threshold for confidence
        """
        self.confidence_threshold = confidence_threshold
        self.groq_client = GroqClient()
    
    def assess_confidence(self, query: str, results: List[Dict[str, Any]]) -> float:
        """
        Assess confidence in the results for a query
        
        Args:
            query (str): The query
            results (list): The retrieved results
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        # Calculate confidence based on similarity scores
        similarities = [result['similarity'] for result in results]
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        
        # Weight max similarity more heavily
        confidence = 0.7 * max_similarity + 0.3 * avg_similarity
        
        return min(1.0, max(0.0, confidence))
    
    def generate_response_with_uncertainty(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        confidence: float
    ) -> str:
        """
        Generate a response that acknowledges uncertainty when appropriate
        
        Args:
            query (str): The query
            results (list): The retrieved results
            confidence (float): Confidence score
            
        Returns:
            str: Generated response
        """
        # Format results for the LLM
        results_text = "\n\n".join([
            f"Result {i+1} (Similarity: {result['similarity']:.4f}):\n{result['text']}"
            for i, result in enumerate(results)
        ])
        
        # Determine confidence level
        if confidence >= self.confidence_threshold:
            confidence_level = "high"
        elif confidence >= self.confidence_threshold / 2:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        prompt = f"""
        Generate a response to this query: "{query}"
        
        Based on these retrieved results:
        {results_text}
        
        Confidence level: {confidence_level} (score: {confidence:.2f})
        
        Guidelines:
        - If confidence is high, provide a direct and authoritative response
        - If confidence is medium, provide a response but acknowledge some uncertainty
        - If confidence is low, acknowledge significant uncertainty and provide caveats
        - If the results don't answer the query well, acknowledge the limitations
        - Synthesize information from multiple results when appropriate
        
        Response:
        """
        
        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Generate a simple response if LLM fails
            print(f"Response generation failed: {e}")
            
            if confidence >= self.confidence_threshold:
                return f"Based on my knowledge, {results[0]['text']}"
            elif confidence >= self.confidence_threshold / 2:
                return f"I'm not entirely certain, but based on available information, {results[0]['text']}"
            else:
                return f"I don't have enough reliable information to answer that question confidently. The most relevant information I have is: {results[0]['text']}"


# Example usage
if __name__ == "__main__":
    # Create a knowledge base
    kb = KnowledgeBase()
    
    # Add some items
    kb.add_item("Python is a high-level programming language known for its readability and simplicity.")
    kb.add_item("Machine learning is a subset of artificial intelligence that enables systems to learn from data.")
    kb.add_item("Natural language processing (NLP) is a field of AI focused on the interaction between computers and human language.")
    kb.add_item("Vector databases store data as high-dimensional vectors and enable semantic search.")
    kb.add_item("Embeddings are numerical representations of text that capture semantic meaning.")
    
    # Retrieve information
    query = "How do computers understand language?"
    results = kb.retrieve(query, top_k=2)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text']} (Similarity: {result['similarity']:.4f})")
    
    # Generate a response with uncertainty handling
    uncertainty = UncertaintyHandler()
    confidence = uncertainty.assess_confidence(query, results)
    response = uncertainty.generate_response_with_uncertainty(query, results, confidence)
    
    print("\nResponse with uncertainty handling:")
    print(response)
    
    # Add citations
    citation_manager = CitationManager(kb)
    cited_response = citation_manager.add_citations_to_response(response, results)
    
    print("\nResponse with citations:")
    print(cited_response)
