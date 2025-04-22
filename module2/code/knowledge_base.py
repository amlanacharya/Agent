"""
Knowledge Base Implementation
----------------------------
This file contains implementations of a knowledge base system that uses
vector embeddings for semantic search and retrieval, with additional
features for metadata, relationships, and confidence tracking.
"""

import os
import json
import time
import uuid
from datetime import datetime
import numpy as np

# Import our vector store implementation
from module2.code.vector_store import SimpleVectorDB, simple_embedding


class KnowledgeBase:
    """
    A knowledge base system that stores and retrieves information using
    vector embeddings for semantic search, with support for metadata,
    relationships, and confidence tracking.
    """
    
    def __init__(self, storage_dir="knowledge_base", embedding_function=None):
        """
        Initialize the knowledge base
        
        Args:
            storage_dir (str): Directory for persistent storage
            embedding_function: Function to convert text to vectors
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize vector database for semantic search
        self.vector_db_path = os.path.join(storage_dir, "vector_db.json")
        if os.path.exists(self.vector_db_path):
            self.vector_db = SimpleVectorDB.load(self.vector_db_path, embedding_function)
        else:
            self.vector_db = SimpleVectorDB(embedding_function or simple_embedding)
        
        # Initialize metadata storage
        self.metadata_path = os.path.join(storage_dir, "metadata.json")
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Initialize relationship graph
        self.relationships_path = os.path.join(storage_dir, "relationships.json")
        if os.path.exists(self.relationships_path):
            with open(self.relationships_path, 'r') as f:
                self.relationships = json.load(f)
        else:
            self.relationships = {}
    
    def add_knowledge(self, knowledge_text, metadata=None):
        """
        Add a new piece of knowledge to the knowledge base
        
        Args:
            knowledge_text (str): The text content to store
            metadata (dict, optional): Additional metadata for the knowledge
                Can include: source, confidence, tags, etc.
                
        Returns:
            str: The ID of the added knowledge
        """
        # Generate a unique ID for this knowledge
        knowledge_id = str(uuid.uuid4())
        
        # Store the text in the vector database for semantic search
        self.vector_db.add_item(knowledge_id, knowledge_text)
        
        # Store metadata
        self.metadata[knowledge_id] = {
            "text": knowledge_text,
            "timestamp": time.time(),
            "source": metadata.get("source", "unknown"),
            "confidence": metadata.get("confidence", 1.0),
            "last_accessed": time.time(),
            "tags": metadata.get("tags", []),
            "created_by": metadata.get("created_by", "system")
        }
        
        # Save changes
        self._save_metadata()
        self.vector_db.save(self.vector_db_path)
        
        return knowledge_id
    
    def retrieve_knowledge(self, query, top_k=5, min_confidence=0.0):
        """
        Retrieve knowledge relevant to the query
        
        Args:
            query (str): The search query
            top_k (int): Maximum number of results to return
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            list: Relevant knowledge entries with metadata and similarity scores
        """
        # Search the vector database
        results = self.vector_db.search(query, top_k=top_k*2)  # Get more results to filter by confidence
        
        # Enhance results with metadata and filter by confidence
        enhanced_results = []
        for knowledge_id, text, similarity, _ in results:
            if knowledge_id in self.metadata:
                # Update last accessed time
                self.metadata[knowledge_id]["last_accessed"] = time.time()
                
                # Check confidence threshold
                confidence = self.metadata[knowledge_id]["confidence"]
                if confidence >= min_confidence:
                    # Add metadata to result
                    enhanced_results.append({
                        "id": knowledge_id,
                        "text": self.metadata[knowledge_id]["text"],
                        "similarity": similarity,
                        "source": self.metadata[knowledge_id]["source"],
                        "confidence": confidence,
                        "timestamp": self.metadata[knowledge_id]["timestamp"],
                        "tags": self.metadata[knowledge_id].get("tags", [])
                    })
        
        # Save updated access times
        self._save_metadata()
        
        # Return top k results after filtering
        return enhanced_results[:top_k]
    
    def get_knowledge(self, knowledge_id):
        """
        Get a specific knowledge entry by ID
        
        Args:
            knowledge_id (str): The ID of the knowledge to retrieve
            
        Returns:
            dict: The knowledge entry with metadata or None if not found
        """
        if knowledge_id in self.metadata:
            # Update last accessed time
            self.metadata[knowledge_id]["last_accessed"] = time.time()
            self._save_metadata()
            
            return {
                "id": knowledge_id,
                "text": self.metadata[knowledge_id]["text"],
                "source": self.metadata[knowledge_id]["source"],
                "confidence": self.metadata[knowledge_id]["confidence"],
                "timestamp": self.metadata[knowledge_id]["timestamp"],
                "tags": self.metadata[knowledge_id].get("tags", [])
            }
        
        return None
    
    def update_knowledge(self, knowledge_id, text=None, metadata=None):
        """
        Update an existing knowledge entry
        
        Args:
            knowledge_id (str): The ID of the knowledge to update
            text (str, optional): New text content
            metadata (dict, optional): New or updated metadata
            
        Returns:
            bool: True if the knowledge was updated, False if not found
        """
        if knowledge_id not in self.metadata:
            return False
        
        # Update text in vector database if provided
        if text is not None:
            self.vector_db.update_item(knowledge_id, text)
            self.metadata[knowledge_id]["text"] = text
        
        # Update metadata if provided
        if metadata is not None:
            for key, value in metadata.items():
                self.metadata[knowledge_id][key] = value
        
        # Update last modified timestamp
        self.metadata[knowledge_id]["last_modified"] = time.time()
        
        # Save changes
        self._save_metadata()
        self.vector_db.save(self.vector_db_path)
        
        return True
    
    def delete_knowledge(self, knowledge_id):
        """
        Delete a knowledge entry
        
        Args:
            knowledge_id (str): The ID of the knowledge to delete
            
        Returns:
            bool: True if the knowledge was deleted, False if not found
        """
        if knowledge_id not in self.metadata:
            return False
        
        # Delete from vector database
        self.vector_db.delete_item(knowledge_id)
        
        # Delete metadata
        del self.metadata[knowledge_id]
        
        # Delete relationships
        if knowledge_id in self.relationships:
            del self.relationships[knowledge_id]
        
        # Remove from other relationships
        for source_id in list(self.relationships.keys()):
            self.relationships[source_id] = [
                rel for rel in self.relationships[source_id]
                if rel["target"] != knowledge_id
            ]
        
        # Save changes
        self._save_metadata()
        self._save_relationships()
        self.vector_db.save(self.vector_db_path)
        
        return True
    
    def add_relationship(self, source_id, target_id, relationship_type):
        """
        Add a relationship between knowledge entries
        
        Args:
            source_id (str): The ID of the source knowledge
            target_id (str): The ID of the target knowledge
            relationship_type (str): The type of relationship
            
        Returns:
            bool: True if the relationship was added, False if either ID is invalid
        """
        if source_id not in self.metadata or target_id not in self.metadata:
            return False
        
        if source_id not in self.relationships:
            self.relationships[source_id] = []
        
        # Check if relationship already exists
        for rel in self.relationships[source_id]:
            if rel["target"] == target_id and rel["type"] == relationship_type:
                return True  # Already exists
        
        # Add the relationship
        self.relationships[source_id].append({
            "target": target_id,
            "type": relationship_type,
            "timestamp": time.time()
        })
        
        # Save changes
        self._save_relationships()
        
        return True
    
    def get_related_knowledge(self, knowledge_id, relationship_type=None):
        """
        Get knowledge related to a specific entry
        
        Args:
            knowledge_id (str): The ID of the knowledge
            relationship_type (str, optional): Filter by relationship type
            
        Returns:
            list: Related knowledge entries with relationship information
        """
        if knowledge_id not in self.relationships:
            return []
        
        related = []
        for relation in self.relationships[knowledge_id]:
            target_id = relation["target"]
            rel_type = relation["type"]
            
            # Filter by relationship type if specified
            if relationship_type is not None and rel_type != relationship_type:
                continue
            
            if target_id in self.metadata:
                related.append({
                    "id": target_id,
                    "text": self.metadata[target_id]["text"],
                    "relationship": rel_type,
                    "source": self.metadata[target_id]["source"],
                    "confidence": self.metadata[target_id]["confidence"]
                })
        
        return related
    
    def get_knowledge_by_tag(self, tag):
        """
        Get knowledge entries with a specific tag
        
        Args:
            tag (str): The tag to search for
            
        Returns:
            list: Knowledge entries with the specified tag
        """
        results = []
        for knowledge_id, metadata in self.metadata.items():
            if "tags" in metadata and tag in metadata["tags"]:
                results.append({
                    "id": knowledge_id,
                    "text": metadata["text"],
                    "source": metadata["source"],
                    "confidence": metadata["confidence"],
                    "timestamp": metadata["timestamp"]
                })
        
        return results
    
    def get_knowledge_by_source(self, source):
        """
        Get knowledge entries from a specific source
        
        Args:
            source (str): The source to search for
            
        Returns:
            list: Knowledge entries from the specified source
        """
        results = []
        for knowledge_id, metadata in self.metadata.items():
            if metadata["source"] == source:
                results.append({
                    "id": knowledge_id,
                    "text": metadata["text"],
                    "confidence": metadata["confidence"],
                    "timestamp": metadata["timestamp"],
                    "tags": metadata.get("tags", [])
                })
        
        return results
    
    def get_all_knowledge(self):
        """
        Get all knowledge entries
        
        Returns:
            list: All knowledge entries with metadata
        """
        return [
            {
                "id": knowledge_id,
                "text": metadata["text"],
                "source": metadata["source"],
                "confidence": metadata["confidence"],
                "timestamp": metadata["timestamp"],
                "tags": metadata.get("tags", [])
            }
            for knowledge_id, metadata in self.metadata.items()
        ]
    
    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def _save_relationships(self):
        """Save relationships to disk"""
        with open(self.relationships_path, 'w') as f:
            json.dump(self.relationships, f)
    
    def save(self):
        """Save the entire knowledge base"""
        self._save_metadata()
        self._save_relationships()
        self.vector_db.save(self.vector_db_path)
    
    def clear(self):
        """Clear the entire knowledge base"""
        self.metadata = {}
        self.relationships = {}
        self.vector_db = SimpleVectorDB(self.vector_db.embedding_function)
        
        self._save_metadata()
        self._save_relationships()
        self.vector_db.save(self.vector_db_path)
    
    def __len__(self):
        """Get the number of knowledge entries"""
        return len(self.metadata)
    
    def __str__(self):
        """String representation of the knowledge base"""
        return f"KnowledgeBase(entries={len(self.metadata)}, path='{self.storage_dir}')"


class CitationManager:
    """
    A manager for formatting citations from knowledge base entries
    """
    
    def __init__(self, knowledge_base):
        """
        Initialize the citation manager
        
        Args:
            knowledge_base (KnowledgeBase): The knowledge base to use
        """
        self.knowledge_base = knowledge_base
    
    def format_citation(self, knowledge_entry, citation_style="standard"):
        """
        Format a citation for a knowledge entry
        
        Args:
            knowledge_entry (dict): The knowledge entry to cite
            citation_style (str): The citation style to use
                Options: "standard", "academic", "url"
                
        Returns:
            str: The formatted citation
        """
        if citation_style == "standard":
            source = knowledge_entry["source"]
            if source == "unknown":
                return "No source available"
            
            timestamp = datetime.fromtimestamp(knowledge_entry["timestamp"])
            date_str = timestamp.strftime("%Y-%m-%d")
            
            return f"{source} ({date_str})"
        
        elif citation_style == "academic":
            source = knowledge_entry["source"]
            if source == "unknown":
                return "Unknown source"
            
            timestamp = datetime.fromtimestamp(knowledge_entry["timestamp"])
            year = timestamp.year
            
            return f"{source}, {year}"
        
        elif citation_style == "url":
            source = knowledge_entry["source"]
            if source.startswith("http"):
                return f"[Source]({source})"
            else:
                return source
        
        return knowledge_entry["source"]
    
    def add_citations_to_response(self, response, knowledge_entries, citation_style="standard"):
        """
        Add citations to a response
        
        Args:
            response (str): The response text
            knowledge_entries (list): The knowledge entries used in the response
            citation_style (str): The citation style to use
                
        Returns:
            str: The response with added citations
        """
        if not knowledge_entries:
            return response
        
        # Add citations section
        response += "\n\nSources:"
        
        for i, entry in enumerate(knowledge_entries):
            citation = self.format_citation(entry, citation_style)
            response += f"\n[{i+1}] {citation}"
        
        return response


class UncertaintyHandler:
    """
    A handler for managing uncertainty in knowledge base responses
    """
    
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize the uncertainty handler
        
        Args:
            confidence_threshold (float): Threshold for confident answers
        """
        self.confidence_threshold = confidence_threshold
    
    def evaluate_confidence(self, knowledge_results):
        """
        Evaluate confidence in the knowledge results
        
        Args:
            knowledge_results (list): The knowledge results to evaluate
            
        Returns:
            float: The overall confidence score
        """
        if not knowledge_results:
            return 0.0
        
        # Calculate overall confidence based on similarity and stored confidence
        top_result = knowledge_results[0]
        return top_result["similarity"] * top_result["confidence"]
    
    def generate_response(self, question, knowledge_results):
        """
        Generate a response with appropriate uncertainty markers
        
        Args:
            question (str): The user's question
            knowledge_results (list): The knowledge results
            
        Returns:
            str: The generated response
        """
        confidence = self.evaluate_confidence(knowledge_results)
        
        if confidence >= self.confidence_threshold:
            # High confidence response
            prefix = "Based on my knowledge, "
            suffix = ""
        elif confidence >= self.confidence_threshold * 0.7:
            # Medium confidence
            prefix = "I believe that "
            suffix = ", though I'm not entirely certain."
        elif confidence >= self.confidence_threshold * 0.4:
            # Low confidence
            prefix = "I'm not entirely sure, but I think "
            suffix = ". Please verify this information."
        else:
            # Very low confidence
            return "I don't have enough reliable information to answer that question confidently."
        
        if knowledge_results:
            answer = prefix + knowledge_results[0]["text"] + suffix
            return answer
        else:
            return "I don't have information about that in my knowledge base."
    
    def should_ask_clarification(self, question, knowledge_results):
        """
        Determine if clarification is needed
        
        Args:
            question (str): The user's question
            knowledge_results (list): The knowledge results
            
        Returns:
            bool: True if clarification should be requested
        """
        confidence = self.evaluate_confidence(knowledge_results)
        
        # If confidence is low but not extremely low, ask for clarification
        return 0.3 <= confidence < self.confidence_threshold * 0.7
    
    def generate_clarification_request(self, question, knowledge_results):
        """
        Generate a request for clarification
        
        Args:
            question (str): The user's question
            knowledge_results (list): The knowledge results
            
        Returns:
            str: The clarification request
        """
        if not knowledge_results:
            return "Could you provide more details about what you're asking? I don't have information on this topic."
        
        # Extract key terms from the question
        # In a real implementation, this would use NLP techniques
        # For this example, we'll use a simple approach
        
        return f"I'm not sure I understand your question about '{question}'. Could you rephrase or provide more context?"


if __name__ == "__main__":
    # Example usage
    kb = KnowledgeBase("example_kb")
    
    # Add some knowledge
    kb.add_knowledge(
        "Python is a high-level, interpreted programming language known for its readability.",
        {"source": "Python Documentation", "confidence": 1.0, "tags": ["programming", "python"]}
    )
    
    kb.add_knowledge(
        "Python was created by Guido van Rossum and first released in 1991.",
        {"source": "Python History", "confidence": 0.95, "tags": ["programming", "python", "history"]}
    )
    
    kb.add_knowledge(
        "JavaScript is a programming language commonly used for web development.",
        {"source": "MDN Web Docs", "confidence": 0.9, "tags": ["programming", "javascript", "web"]}
    )
    
    # Retrieve knowledge
    results = kb.retrieve_knowledge("What is Python?")
    print("Query: What is Python?")
    for result in results:
        print(f"- {result['text']} (Similarity: {result['similarity']:.2f}, Confidence: {result['confidence']})")
    
    # Use the uncertainty handler
    uncertainty = UncertaintyHandler()
    response = uncertainty.generate_response("What is Python?", results)
    print("\nResponse:")
    print(response)
    
    # Add citations
    citation_manager = CitationManager(kb)
    cited_response = citation_manager.add_citations_to_response(response, results)
    print("\nResponse with citations:")
    print(cited_response)
