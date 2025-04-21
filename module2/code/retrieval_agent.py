"""
Retrieval Patterns Implementation
-------------------------------
This file contains implementations of various retrieval patterns for contextual memory,
including recency-based, conversation-aware, and multi-query retrieval.
"""

import time
import numpy as np
from collections import Counter

# Import from our other modules
# These imports assume the files are in the same directory
try:
    from memory_types import AgentMemorySystem
    from vector_store import SimpleVectorDB, RetrievalMemory
except ImportError:
    # If running as a standalone file, provide placeholders
    print("Warning: Running in standalone mode. Some functionality may be limited.")
    AgentMemorySystem = None
    SimpleVectorDB = None
    RetrievalMemory = None

# Basic retrieval patterns

def recency_based_retrieval(query, memory_system, max_age_hours=24, top_k=5):
    """
    Retrieve information based on recency and relevance

    Args:
        query (str): The search query
        memory_system: The memory system to search (must have a retrieve method)
        max_age_hours (int): Maximum age of memories to consider
        top_k (int): Number of results to return

    Returns:
        list: Top k relevant and recent items
    """
    # Calculate the cutoff timestamp
    cutoff_time = time.time() - (max_age_hours * 3600)

    # Get all memories from the retrieval system
    # We request more than top_k to have a larger pool for filtering
    all_results = memory_system.retrieve(query, top_k=top_k*2)

    # Filter and re-rank based on recency and relevance
    filtered_results = []
    for result in all_results:
        # Get timestamp from metadata (default to 0 if not found)
        timestamp = result.get('metadata', {}).get('timestamp', 0)

        if timestamp >= cutoff_time:
            # Calculate a combined score that considers both relevance and recency
            # Normalize recency to a 0-1 scale (1 being most recent)
            recency_score = 1.0 - ((time.time() - timestamp) / (max_age_hours * 3600))
            relevance_score = result.get('score', 0.5)  # Default to 0.5 if no score

            # Combine scores (you can adjust the weights)
            combined_score = (0.7 * relevance_score) + (0.3 * recency_score)

            # Add to filtered results with the new score
            result['combined_score'] = combined_score
            filtered_results.append(result)

    # Sort by combined score and return top k
    filtered_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
    return filtered_results[:top_k]


def conversation_aware_retrieval(query, conversation_history, memory_system, top_k=5):
    """
    Retrieve information based on conversation context

    Args:
        query (str): The current query
        conversation_history (list): Recent conversation turns
        memory_system: The memory system to search
        top_k (int): Number of results to return

    Returns:
        list: Top k contextually relevant items
    """
    # Extract recent user messages from conversation history
    # We assume each turn in history is a dict with 'user_input' and 'agent_response' keys
    recent_messages = []

    # Get up to 3 most recent turns
    for turn in conversation_history[-3:]:
        if isinstance(turn, dict) and 'user_input' in turn:
            recent_messages.append(turn['user_input'])

    # Combine the current query with recent context
    # We give more weight to the current query by repeating it
    enhanced_query = query + " " + query + " " + " ".join(recent_messages)

    # Retrieve based on the enhanced query
    results = memory_system.retrieve(enhanced_query, top_k=top_k)

    # Add information about the enhancement to each result
    for result in results:
        result['enhanced_with'] = {
            'original_query': query,
            'conversation_context': recent_messages
        }

    return results


def multi_query_retrieval(query, memory_system, top_k=5):
    """
    Retrieve information using multiple query variations

    Args:
        query (str): The original query
        memory_system: The memory system to search
        top_k (int): Number of results to return

    Returns:
        list: Top k results from multiple queries
    """
    # Generate query variations
    # In a real system, you might use an LLM to generate these variations
    query_variations = [
        query,  # Original query
        f"information about {query}",  # Expanded query
        f"explain {query}",  # Instruction-style query
    ]

    # If the query has multiple words, add a shortened version with key terms
    query_words = query.split()
    if len(query_words) > 3:
        # Use the first 3 words as a shortened query
        query_variations.append(" ".join(query_words[:3]))

    # Add a variation with synonyms if possible
    # This is a simplified approach - in a real system, use a thesaurus or word embedding model
    common_synonym_pairs = [
        ("create", "make"), ("build", "construct"), ("help", "assist"),
        ("information", "data"), ("problem", "issue"), ("example", "sample"),
        ("important", "significant"), ("different", "various"), ("show", "display")
    ]

    # Try to replace words with synonyms
    synonym_query = query
    for word1, word2 in common_synonym_pairs:
        if word1 in query.lower():
            synonym_query = synonym_query.lower().replace(word1, word2)
            query_variations.append(synonym_query)
            break  # Just do one replacement for simplicity

    # Collect results from all queries
    all_results = []
    seen_ids = set()  # To track unique results

    for variation in query_variations:
        # Get results for this variation
        results = memory_system.retrieve(variation, top_k=top_k//2)  # Use smaller top_k for each variation

        for result in results:
            # Check if we've already seen this result
            result_id = result.get('id', '')
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)

                # Add information about which query variation found this result
                result['found_by_query'] = variation
                all_results.append(result)

    # Re-rank combined results by score
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Return top k unique results
    return all_results[:top_k]

