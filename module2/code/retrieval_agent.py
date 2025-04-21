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


# Advanced retrieval systems

class ConversationMemory:
    """
    A conversation memory system that uses semantic search to find relevant past interactions.
    """

    def __init__(self, vector_store=None, max_history=100):
        """
        Initialize the conversation memory

        Args:
            vector_store: Vector database for storing conversation turns
            max_history (int): Maximum number of turns to store
        """
        self.vector_store = vector_store or SimpleVectorDB()
        self.max_history = max_history
        self.turns = []  # List to store conversation turns in order
        self.turn_counter = 0

    def add_turn(self, user_input, agent_response):
        """
        Add a conversation turn to memory

        Args:
            user_input (str): The user's input
            agent_response (str): The agent's response
        """
        # Create a turn object
        turn = {
            'turn_id': self.turn_counter,
            'timestamp': time.time(),
            'user_input': user_input,
            'agent_response': agent_response
        }

        # Add to the list of turns
        self.turns.append(turn)
        if len(self.turns) > self.max_history:
            self.turns = self.turns[-self.max_history:]

        # Store in vector database for semantic search
        # Store both the user input and agent response separately for more granular retrieval
        self.vector_store.add_item(
            f"turn_{self.turn_counter}_user",
            user_input,
            {'turn_id': self.turn_counter, 'type': 'user_input', 'timestamp': turn['timestamp']}
        )

        self.vector_store.add_item(
            f"turn_{self.turn_counter}_agent",
            agent_response,
            {'turn_id': self.turn_counter, 'type': 'agent_response', 'timestamp': turn['timestamp']}
        )

        # Increment turn counter
        self.turn_counter += 1

    def get_recent_turns(self, n=5):
        """
        Get the n most recent conversation turns

        Args:
            n (int): Number of turns to retrieve

        Returns:
            list: Recent conversation turns
        """
        return self.turns[-n:] if self.turns else []

    def search_conversation(self, query, top_k=3):
        """
        Search for relevant conversation turns

        Args:
            query (str): The search query
            top_k (int): Number of results to return

        Returns:
            list: Relevant conversation turns
        """
        # Search in the vector database
        results = self.vector_store.search(query, top_k=top_k)

        # Format the results
        formatted_results = []
        for item_id, text, score, metadata in results:
            turn_id = metadata.get('turn_id')
            turn_type = metadata.get('type')

            # Find the full turn
            full_turn = None
            for turn in self.turns:
                if turn['turn_id'] == turn_id:
                    full_turn = turn
                    break

            if full_turn:
                formatted_results.append({
                    'turn': full_turn,
                    'matched_text': text,
                    'matched_type': turn_type,
                    'score': score
                })

        return formatted_results

    def get_conversation_summary(self, query=None, max_turns=3):
        """
        Get a summary of the conversation, optionally focused on a specific query

        Args:
            query (str, optional): Focus the summary on this query
            max_turns (int): Maximum number of turns to include

        Returns:
            str: A summary of the conversation
        """
        if query and len(self.turns) > max_turns:
            # If a query is provided, find relevant turns
            relevant_turns = self.search_conversation(query, top_k=max_turns)
            turns_to_summarize = [result['turn'] for result in relevant_turns]
        else:
            # Otherwise, use the most recent turns
            turns_to_summarize = self.get_recent_turns(max_turns)

        # Create a simple summary (in a real system, you might use an LLM for this)
        summary = "Conversation summary:\n"
        for turn in turns_to_summarize:
            summary += f"User: {turn['user_input']}\n"
            summary += f"Agent: {turn['agent_response']}\n\n"

        return summary

    def clear(self):
        """
        Clear the conversation memory
        """
        self.turns = []
        self.turn_counter = 0
        # If the vector store has a clear method, use it
        if hasattr(self.vector_store, 'clear'):
            self.vector_store.clear()

    def __len__(self):
        """
        Get the number of turns in memory
        """
        return len(self.turns)

    def __str__(self):
        """
        String representation of the conversation memory
        """
        return f"ConversationMemory(turns={len(self.turns)}, max_history={self.max_history})"


class HybridRetrievalSystem:
    """
    A hybrid retrieval system that combines multiple memory types and retrieval methods.
    """

    def __init__(self, vector_memory, episodic_memory=None, long_term_memory=None):
        """
        Initialize the hybrid retrieval system

        Args:
            vector_memory: Vector-based retrieval memory
            episodic_memory: Memory for specific episodes/experiences
            long_term_memory: Key-value based long-term memory
        """
        self.vector_memory = vector_memory
        self.episodic_memory = episodic_memory
        self.long_term_memory = long_term_memory

    def retrieve(self, query, conversation_context=None, top_k=5):
        """
        Retrieve information using a hybrid approach

        Args:
            query (str): The search query
            conversation_context (list, optional): Recent conversation turns
            top_k (int): Number of results to return

        Returns:
            list: Top k results from multiple sources
        """
        results = []

        # 1. Check for exact matches in long-term memory
        if self.long_term_memory:
            exact_match = self.long_term_memory.retrieve(query)
            if exact_match:
                results.append({
                    'id': 'exact_match',
                    'text': exact_match,
                    'score': 1.0,
                    'source': 'long_term_memory'
                })

        # 2. Retrieve from vector memory with semantic search
        vector_results = self.vector_memory.retrieve(query, top_k=top_k)
        for result in vector_results:
            result['source'] = 'vector_memory'
            results.append(result)

        # 3. If conversation context is provided, enhance with conversation-aware retrieval
        if conversation_context:
            # Extract recent user messages
            recent_messages = []
            for turn in conversation_context[-3:]:
                if isinstance(turn, dict) and 'user_input' in turn:
                    recent_messages.append(turn['user_input'])

            # Combine the current query with recent context
            enhanced_query = query + " " + " ".join(recent_messages)

            # Get additional results with the enhanced query
            enhanced_results = self.vector_memory.retrieve(enhanced_query, top_k=top_k//2)
            for result in enhanced_results:
                # Check if this result is already included
                if not any(r.get('id') == result.get('id') for r in results):
                    result['source'] = 'conversation_context'
                    results.append(result)

        # 4. Check for relevant episodes if episodic memory is available
        if self.episodic_memory:
            episode_results = []
            for episode in self.episodic_memory.get_episodes_by_type('conversation'):
                # Simple keyword matching (in a real system, use more sophisticated matching)
                if query.lower() in str(episode.get('content', '')).lower():
                    episode_results.append({
                        'id': f"episode_{episode.get('timestamp', '')}",
                        'text': str(episode.get('content', '')),
                        'score': 0.7,  # Arbitrary score for demonstration
                        'source': 'episodic_memory'
                    })

            # Add top episode results
            results.extend(episode_results[:top_k//2])

        # Sort all results by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Return top k unique results
        unique_results = []
        seen_ids = set()

        for result in results:
            result_id = result.get('id', '')
            if result_id and result_id not in seen_ids and len(unique_results) < top_k:
                seen_ids.add(result_id)
                unique_results.append(result)

        return unique_results

    def retrieve_with_explanation(self, query, conversation_context=None, top_k=5):
        """
        Retrieve information and provide explanations for why each result was returned

        Args:
            query (str): The search query
            conversation_context (list, optional): Recent conversation turns
            top_k (int): Number of results to return

        Returns:
            list: Top k results with explanations
        """
        # Get results using the standard retrieve method
        results = self.retrieve(query, conversation_context, top_k)

        # Add explanations to each result
        for result in results:
            source = result.get('source', 'unknown')
            score = result.get('score', 0)

            # Generate explanation based on source and score
            if source == 'long_term_memory':
                explanation = "This is an exact match from long-term memory."
            elif source == 'vector_memory':
                if score > 0.9:
                    explanation = "This is a very close semantic match to your query."
                elif score > 0.7:
                    explanation = "This is semantically related to your query."
                else:
                    explanation = "This is somewhat related to your query."
            elif source == 'conversation_context':
                explanation = "This is relevant based on your recent conversation context."
            elif source == 'episodic_memory':
                explanation = "This is from a past conversation that seems relevant."
            else:
                explanation = "This was found through multiple retrieval methods."

            # Add the explanation to the result
            result['explanation'] = explanation

        return results

    def __str__(self):
        """
        String representation of the hybrid retrieval system
        """
        components = []
        if self.vector_memory:
            components.append("vector_memory")
        if self.episodic_memory:
            components.append("episodic_memory")
        if self.long_term_memory:
            components.append("long_term_memory")

        return f"HybridRetrievalSystem(components=[{', '.join(components)}])"


class ContextAwareRetrieval:
    """
    A retrieval system that considers the broader context of the interaction.
    """

    def __init__(self, memory_system, conversation_memory):
        """
        Initialize the context-aware retrieval system

        Args:
            memory_system: The main memory system
            conversation_memory: The conversation memory system
        """
        self.memory_system = memory_system
        self.conversation_memory = conversation_memory

    def retrieve(self, query, user_profile=None, top_k=5):
        """
        Retrieve information based on query and context

        Args:
            query (str): The search query
            user_profile (dict, optional): User profile information
            top_k (int): Number of results to return

        Returns:
            list: Contextually relevant results
        """
        # Get recent conversation context
        recent_turns = self.conversation_memory.get_recent_turns(3)

        # Extract key information from context
        context_keywords = self._extract_keywords(recent_turns)

        # Enhance query with context
        enhanced_query = self._enhance_query(query, context_keywords, user_profile)

        # Retrieve with enhanced query
        results = self.memory_system.retrieve(enhanced_query, top_k=top_k)

        # Add context information to results
        for result in results:
            result['context_info'] = {
                'original_query': query,
                'enhanced_query': enhanced_query,
                'context_keywords': context_keywords,
                'user_profile_used': bool(user_profile)
            }

        return results

    def _extract_keywords(self, conversation_turns):
        """
        Extract key information from conversation turns

        Args:
            conversation_turns (list): Recent conversation turns

        Returns:
            list: Extracted keywords
        """
        # In a real system, use NLP techniques to extract keywords
        # For simplicity, we'll just use a basic approach here
        all_text = " ".join([
            turn.get('user_input', '') + " " + turn.get('agent_response', '')
            for turn in conversation_turns
        ])

        # Simple keyword extraction (remove common words, split, etc.)
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
                        "in", "on", "at", "to", "for", "with", "by", "about", "like",
                        "from", "of", "that", "this", "these", "those"}

        words = all_text.lower().split()
        keywords = [word for word in words if word not in common_words and len(word) > 3]

        # Count occurrences and get top keywords
        word_counts = {}
        for word in keywords:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by count and return top 5
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:5]]

    def _enhance_query(self, query, context_keywords, user_profile=None):
        """
        Enhance the query with context and user profile

        Args:
            query (str): The original query
            context_keywords (list): Keywords from conversation context
            user_profile (dict, optional): User profile information

        Returns:
            str: Enhanced query
        """
        enhanced_query = query

        # Add context keywords with lower weight
        if context_keywords:
            enhanced_query += " " + " ".join(context_keywords)

        # Add user profile information if available
        if user_profile:
            # Add relevant user interests
            interests = user_profile.get('interests', [])
            if interests:
                enhanced_query += " " + " ".join(interests[:3])

            # Add user expertise level if available
            expertise = user_profile.get('expertise_level')
            if expertise:
                enhanced_query += f" {expertise} level"

        return enhanced_query

    def retrieve_with_context_explanation(self, query, user_profile=None, top_k=5):
        """
        Retrieve information and explain how context influenced the results

        Args:
            query (str): The search query
            user_profile (dict, optional): User profile information
            top_k (int): Number of results to return

        Returns:
            dict: Results and context explanation
        """
        # Get recent conversation context
        recent_turns = self.conversation_memory.get_recent_turns(3)

        # Extract key information from context
        context_keywords = self._extract_keywords(recent_turns)

        # Enhance query with context
        enhanced_query = self._enhance_query(query, context_keywords, user_profile)

        # Retrieve with enhanced query
        results = self.memory_system.retrieve(enhanced_query, top_k=top_k)

        # Create context explanation
        context_explanation = {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'context_keywords': context_keywords,
            'recent_conversation': [
                {'user': turn.get('user_input', ''), 'agent': turn.get('agent_response', '')}
                for turn in recent_turns
            ],
            'user_profile_used': user_profile is not None
        }

        if user_profile:
            context_explanation['user_interests'] = user_profile.get('interests', [])
            context_explanation['user_expertise'] = user_profile.get('expertise_level')

        return {
            'results': results,
            'context_explanation': context_explanation
        }

    def __str__(self):
        """
        String representation of the context-aware retrieval system
        """
        return f"ContextAwareRetrieval(memory_system={type(self.memory_system).__name__}, conversation_memory={type(self.conversation_memory).__name__})"


class RelevanceScorer:
    """
    A system for scoring the relevance of retrieved items based on multiple factors.
    """

    def __init__(self, weights=None):
        """
        Initialize the relevance scorer

        Args:
            weights (dict, optional): Custom weights for different factors
        """
        # Default weights for different factors
        self.weights = weights or {
            'base_similarity': 0.5,
            'recency': 0.2,
            'user_interest_match': 0.15,
            'conversation_relevance': 0.15
        }

        # Ensure weights sum to 1.0
        total = sum(self.weights.values())
        if total != 1.0:
            for key in self.weights:
                self.weights[key] /= total

    def score_item(self, item, query, user_profile=None, conversation_context=None):
        """
        Score an item's relevance based on multiple factors

        Args:
            item (dict): The item to score
            query (str): The original query
            user_profile (dict, optional): User profile information
            conversation_context (list, optional): Recent conversation turns

        Returns:
            float: The relevance score (0-1)
        """
        # Start with the base similarity score
        base_score = item.get('score', 0)

        # Calculate recency score
        recency_score = self._calculate_recency_score(item)

        # Calculate user interest match if profile is available
        user_interest_score = 0
        if user_profile:
            user_interest_score = self._calculate_user_interest_score(item, user_profile)

        # Calculate conversation relevance if context is available
        conversation_relevance = 0
        if conversation_context:
            conversation_relevance = self._calculate_conversation_relevance(
                item, conversation_context
            )

        # Calculate weighted score
        final_score = (
            self.weights['base_similarity'] * base_score +
            self.weights['recency'] * recency_score +
            self.weights['user_interest_match'] * user_interest_score +
            self.weights['conversation_relevance'] * conversation_relevance
        )

        # Ensure score is between 0 and 1
        return max(0, min(1, final_score))

    def _calculate_recency_score(self, item, max_age_hours=24):
        """
        Calculate a score based on item recency

        Args:
            item (dict): The item to score
            max_age_hours (int): Maximum age to consider

        Returns:
            float: Recency score (0-1)
        """
        timestamp = item.get('metadata', {}).get('timestamp', 0)
        if timestamp == 0:
            return 0.5  # Neutral score for items without timestamp

        age_seconds = time.time() - timestamp
        age_hours = age_seconds / 3600

        if age_hours > max_age_hours:
            return 0  # Older than max age

        # Linear decay from 1 (newest) to 0 (oldest)
        return 1.0 - (age_hours / max_age_hours)

    def _calculate_user_interest_score(self, item, user_profile):
        """
        Calculate a score based on match with user interests

        Args:
            item (dict): The item to score
            user_profile (dict): User profile information

        Returns:
            float: User interest score (0-1)
        """
        # Get user interests
        interests = user_profile.get('interests', [])
        if not interests:
            return 0.5  # Neutral score if no interests

        # Check for interest matches in item text
        item_text = item.get('text', '').lower()
        matches = sum(1 for interest in interests if interest.lower() in item_text)

        # Calculate score based on number of matches
        return min(1.0, matches / len(interests))

    def _calculate_conversation_relevance(self, item, conversation_context):
        """
        Calculate a score based on relevance to conversation context

        Args:
            item (dict): The item to score
            conversation_context (list): Recent conversation turns

        Returns:
            float: Conversation relevance score (0-1)
        """
        # Extract text from recent turns
        context_text = " ".join([
            turn.get('user_input', '') + " " + turn.get('agent_response', '')
            for turn in conversation_context[-3:]
        ]).lower()

        # Get item text
        item_text = item.get('text', '').lower()

        # Simple word overlap calculation
        context_words = set(context_text.split())
        item_words = set(item_text.split())

        if not context_words or not item_words:
            return 0.5  # Neutral score if no words

        # Calculate overlap
        overlap = len(context_words.intersection(item_words))

        # Calculate score based on overlap
        return min(1.0, overlap / min(len(context_words), len(item_words)))

    def score_and_rank_results(self, results, query, user_profile=None, conversation_context=None):
        """
        Score and rank a list of results

        Args:
            results (list): List of items to score and rank
            query (str): The original query
            user_profile (dict, optional): User profile information
            conversation_context (list, optional): Recent conversation turns

        Returns:
            list: Scored and ranked results
        """
        # Score each result
        for result in results:
            result['relevance_score'] = self.score_item(
                result, query, user_profile, conversation_context
            )

            # Add score breakdown for transparency
            result['score_breakdown'] = {
                'base_similarity': result.get('score', 0),
                'recency': self._calculate_recency_score(result),
                'user_interest': self._calculate_user_interest_score(result, user_profile) if user_profile else 0,
                'conversation_relevance': self._calculate_conversation_relevance(result, conversation_context) if conversation_context else 0
            }

        # Sort by relevance score
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return results

    def __str__(self):
        """
        String representation of the relevance scorer
        """
        return f"RelevanceScorer(weights={self.weights})"

