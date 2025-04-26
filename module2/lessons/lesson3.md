# 🚀 Module 2: Memory Systems - Lesson 3: Advanced Retrieval Patterns 🔍

## 🎯 Lesson Objectives

By the end of this lesson, you will:
- 🔄 Master **retrieval patterns** for contextual memory
- 🧩 Implement **hybrid retrieval systems** that combine multiple memory types
- 📚 Build a **conversation memory** system with semantic search
- 🔍 Create **context-aware retrieval** mechanisms
- 🧠 Design **relevance scoring** algorithms for better results

---

## 📚 Introduction to Retrieval Patterns

In previous lessons, we explored different memory types and vector databases. Now, we'll focus on **retrieval patterns** - strategies for effectively retrieving information from memory based on context.

Effective retrieval is about more than just finding similar content; it's about finding the **right information at the right time**. This requires:

1. **Understanding the context** of the current interaction
2. **Combining different memory types** for comprehensive retrieval
3. **Prioritizing relevance** based on multiple factors
4. **Adapting retrieval strategies** to different scenarios

---

## 🔄 Contextual Retrieval Patterns

### 1. Recency-Based Retrieval

![Recency](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

Recency-based retrieval prioritizes recent information, which is often more relevant to the current context:

```python
def recency_based_retrieval(query, memory_system, max_age_hours=24, top_k=5):
    """
    Retrieve information based on recency and relevance

    Args:
        query (str): The search query
        memory_system: The memory system to search
        max_age_hours (int): Maximum age of memories to consider
        top_k (int): Number of results to return

    Returns:
        list: Top k relevant and recent items
    """
    # Calculate the cutoff timestamp
    cutoff_time = time.time() - (max_age_hours * 3600)

    # Get all memories from the retrieval system
    all_results = memory_system.retrieve(query, top_k=top_k*2)

    # Filter and re-rank based on recency and relevance
    filtered_results = []
    for result in all_results:
        timestamp = result['metadata'].get('timestamp', 0)
        if timestamp >= cutoff_time:
            # Calculate a combined score that considers both relevance and recency
            recency_score = 1.0 - ((time.time() - timestamp) / (max_age_hours * 3600))
            relevance_score = result['score']

            # Combine scores (you can adjust the weights)
            combined_score = (0.7 * relevance_score) + (0.3 * recency_score)

            # Add to filtered results with the new score
            result['combined_score'] = combined_score
            filtered_results.append(result)

    # Sort by combined score and return top k
    filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
    return filtered_results[:top_k]
```

### 2. Conversation-Aware Retrieval

![Conversation](https://media.giphy.com/media/3o7btNDyBs5dKdhTqM/giphy.gif)

Conversation-aware retrieval considers the flow of the current conversation:

```python
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
    recent_messages = [
        turn['user_input'] for turn in conversation_history[-3:]
    ]

    # Combine the current query with recent context
    enhanced_query = query + " " + " ".join(recent_messages)

    # Retrieve based on the enhanced query
    results = memory_system.retrieve(enhanced_query, top_k=top_k)

    return results
```

### 3. Multi-Query Retrieval

![Multi-Query](https://media.giphy.com/media/3o7TKsQ8Xb3gcGEgZW/giphy.gif)

Multi-query retrieval generates multiple variations of a query to improve recall:

```python
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
    # Generate query variations (in a real system, you might use an LLM for this)
    query_variations = [
        query,  # Original query
        f"information about {query}",  # Expanded query
        f"explain {query}",  # Instruction-style query
        " ".join(query.split()[:3])  # Shortened query with key terms
    ]

    # Collect results from all queries
    all_results = []
    seen_ids = set()

    for variation in query_variations:
        results = memory_system.retrieve(variation, top_k=top_k)

        for result in results:
            # Avoid duplicates
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                all_results.append(result)

    # Re-rank combined results
    all_results.sort(key=lambda x: x['score'], reverse=True)

    return all_results[:top_k]
```

---

## 🧩 Hybrid Retrieval Systems

![Hybrid](https://media.giphy.com/media/3o7btNa0RUYa5E7iiQ/giphy.gif)

Hybrid retrieval systems combine multiple retrieval methods to get the best results:

```python
class HybridRetrievalSystem:
    """
    A hybrid retrieval system that combines multiple memory types and retrieval methods.
    """

    def __init__(self, vector_memory, episodic_memory, long_term_memory):
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
            recent_messages = [
                turn['user_input'] for turn in conversation_context[-3:]
            ]

            # Combine the current query with recent context
            enhanced_query = query + " " + " ".join(recent_messages)

            # Get additional results with the enhanced query
            enhanced_results = self.vector_memory.retrieve(enhanced_query, top_k=top_k//2)
            for result in enhanced_results:
                # Check if this result is already included
                if not any(r['id'] == result['id'] for r in results):
                    result['source'] = 'conversation_context'
                    results.append(result)

        # 4. Check for relevant episodes
        episode_results = []
        for episode in self.episodic_memory.get_episodes_by_type('conversation'):
            # Simple keyword matching (in a real system, use more sophisticated matching)
            if query.lower() in str(episode['content']).lower():
                episode_results.append({
                    'id': f"episode_{episode.get('timestamp', '')}",
                    'text': str(episode['content']),
                    'score': 0.7,  # Arbitrary score for demonstration
                    'source': 'episodic_memory'
                })

        # Add top episode results
        results.extend(episode_results[:top_k//2])

        # Sort all results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top k unique results
        unique_results = []
        seen_ids = set()

        for result in results:
            if result['id'] not in seen_ids and len(unique_results) < top_k:
                seen_ids.add(result['id'])
                unique_results.append(result)

        return unique_results
```

---

## 📚 Conversation Memory with Semantic Search

![Conversation Memory](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

Let's implement a conversation memory system that uses semantic search to find relevant past interactions:

```python
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
        self.turns = []
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
```

---

## 🔍 Context-Aware Retrieval

![Context-Aware](https://media.giphy.com/media/3o7btNDyBs5dKdhTqM/giphy.gif)

Context-aware retrieval considers the broader context of the interaction:

```python
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
            turn['user_input'] + " " + turn['agent_response']
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
```

---

## 🧠 Relevance Scoring Algorithms

![Relevance Scoring](https://media.giphy.com/media/3o7btZ1Gm7ZL25pLMs/giphy.gif)

Effective retrieval requires sophisticated relevance scoring that considers multiple factors:

```python
class RelevanceScorer:
    """
    A system for scoring the relevance of retrieved items based on multiple factors.
    """

    def __init__(self):
        """Initialize the relevance scorer"""
        pass

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

        # Initialize weights for different factors
        weights = {
            'base_similarity': 0.5,
            'recency': 0.2,
            'user_interest_match': 0.15,
            'conversation_relevance': 0.15
        }

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
            weights['base_similarity'] * base_score +
            weights['recency'] * recency_score +
            weights['user_interest_match'] * user_interest_score +
            weights['conversation_relevance'] * conversation_relevance
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
```

---

## 💪 Practice Exercises

1. **Implement a Query Expansion System**:
   - Create a system that expands queries with synonyms and related terms
   - Implement a method to generate multiple query variations
   - Test how query expansion affects retrieval results

2. **Build a Personalized Retrieval System**:
   - Design a retrieval system that adapts to user preferences
   - Implement user profile tracking and interest detection
   - Create a scoring system that prioritizes results based on user interests

3. **Create a Conversation Memory Analyzer**:
   - Implement a system that analyzes conversation patterns
   - Create methods to identify important topics in a conversation
   - Build a retrieval system that uses conversation analysis to improve results

---

## 🔍 Key Concepts to Remember

1. **Context Matters**: Effective retrieval considers the broader context of the interaction
2. **Hybrid Approaches**: Combining multiple retrieval methods often yields better results
3. **Relevance Scoring**: Sophisticated scoring algorithms improve retrieval quality
4. **Conversation Memory**: Tracking and analyzing conversations enhances contextual understanding
5. **Personalization**: Adapting retrieval to user preferences improves relevance

---

## 🚀 Next Steps

In the next lesson, we'll explore:
- Building a complete Knowledge Base Assistant
- Integrating all the memory systems we've learned about
- Implementing question answering based on stored knowledge
- Adding learning capabilities to acquire new information
- Creating a system that can identify when it doesn't know something

---

## 📚 Resources

- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [Pinecone Hybrid Search Guide](https://www.pinecone.io/learn/hybrid-search/)
- [Semantic Search Best Practices](https://www.sbert.net/examples/applications/semantic-search/README.html)
- [Conversation AI Patterns](https://www.microsoft.com/en-us/research/project/conversation-ai/)

---

## 🎯 Mini-Project Progress: Knowledge Base Assistant

![Knowledge Base](https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif)

In this lesson, we've learned about advanced retrieval patterns that will be essential for our Knowledge Base Assistant. We can now:
- Implement context-aware retrieval for more relevant answers
- Build a conversation memory system to maintain context
- Create hybrid retrieval systems that combine multiple approaches
- Design sophisticated relevance scoring for better results

In the next lesson, we'll integrate all these components to build our complete Knowledge Base Assistant.

---

> 💡 **Note on LLM Integration**: This lesson uses simulated retrieval functions for demonstration purposes. In a real implementation, you would integrate these retrieval patterns with LLMs to generate more sophisticated responses based on the retrieved information. For LLM integration, see the Module 2-LLM version.

---

Happy coding! 🚀
