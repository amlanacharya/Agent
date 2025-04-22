"""
Memory Types Implementation with LLM Integration
----------------------------------------------
This file contains implementations of different memory types for AI agents,
enhanced with real LLM integration using the Groq API.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from collections import deque

# Import our Groq client
try:
    # When running from the module2-llm/code directory
    from groq_client import GroqClient
except ImportError:
    # When running from the project root
    from module2_llm.code.groq_client import GroqClient


class WorkingMemory:
    """
    Working memory for immediate context and current task information.
    Enhanced with LLM for better context understanding.
    """

    def __init__(self, capacity: int = 5):
        """
        Initialize working memory

        Args:
            capacity (int): Maximum number of items to store
        """
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
        self.groq_client = GroqClient()

    def add(self, item: Any) -> None:
        """
        Add an item to working memory

        Args:
            item: The item to add
        """
        self.items.append({
            'content': item,
            'timestamp': time.time()
        })

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all items in working memory

        Returns:
            list: All items in working memory
        """
        return list(self.items)

    def clear(self) -> None:
        """Clear working memory"""
        self.items.clear()

    def summarize(self) -> str:
        """
        Generate a summary of the current working memory using LLM

        Returns:
            str: A summary of the current working memory
        """
        if not self.items:
            return "Working memory is empty."

        # Format the items for the LLM
        items_text = "\n".join([
            f"{i+1}. {item['content']}"
            for i, item in enumerate(self.items)
        ])

        prompt = f"""
        Summarize the following items currently in working memory:

        {items_text}

        Provide a concise summary that captures the key information.
        """

        try:
            response = self.groq_client.generate_text(prompt, max_tokens=150)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Fallback to simple summary if LLM fails
            return f"Working memory contains {len(self.items)} items."


class ShortTermMemory:
    """
    Short-term memory for recent interactions and context.
    Enhanced with LLM for better conversation understanding.
    """

    def __init__(self, capacity: int = 20):
        """
        Initialize short-term memory

        Args:
            capacity (int): Maximum number of items to store
        """
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
        self.groq_client = GroqClient()

    def add(self, item: Any) -> None:
        """
        Add an item to short-term memory

        Args:
            item: The item to add
        """
        self.items.append({
            'content': item,
            'timestamp': time.time()
        })

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the n most recent items

        Args:
            n (int): Number of items to retrieve

        Returns:
            list: The n most recent items
        """
        return list(self.items)[-n:]

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all items in short-term memory

        Returns:
            list: All items in short-term memory
        """
        return list(self.items)

    def clear(self) -> None:
        """Clear short-term memory"""
        self.items.clear()

    def extract_key_information(self, query: str) -> str:
        """
        Extract key information from short-term memory relevant to a query using LLM

        Args:
            query (str): The query to extract information for

        Returns:
            str: Extracted key information
        """
        if not self.items:
            return "No information available in short-term memory."

        # Format the items for the LLM
        items_text = "\n".join([
            f"{i+1}. {item['content']}"
            for i, item in enumerate(self.items)
        ])

        prompt = f"""
        Given the following recent conversation history in short-term memory:

        {items_text}

        Extract key information that is relevant to this query: "{query}"

        Provide only the most relevant details that would help answer the query.
        """

        try:
            response = self.groq_client.generate_text(prompt, max_tokens=200)
            return self.groq_client.extract_text_from_response(response)
        except Exception as e:
            # Fallback to returning recent items if LLM fails
            return f"Recent items in memory: {[item['content'] for item in self.get_recent(3)]}"


class LongTermMemory:
    """
    Long-term memory for persistent knowledge and learned information.
    Enhanced with LLM for better knowledge organization and retrieval.
    """

    def __init__(self, storage_path: str = "long_term_memory.json"):
        """
        Initialize long-term memory

        Args:
            storage_path (str): Path to the storage file
        """
        self.storage_path = storage_path
        self.memory = self._load_memory()
        self.groq_client = GroqClient()

    def _load_memory(self) -> Dict[str, Any]:
        """
        Load memory from storage

        Returns:
            dict: The loaded memory
        """
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {'facts': [], 'concepts': {}, 'last_updated': time.time()}
        else:
            return {'facts': [], 'concepts': {}, 'last_updated': time.time()}

    def _save_memory(self) -> None:
        """Save memory to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f)

    def add_fact(self, fact: str) -> None:
        """
        Add a fact to long-term memory

        Args:
            fact (str): The fact to add
        """
        self.memory['facts'].append({
            'content': fact,
            'timestamp': time.time()
        })
        self.memory['last_updated'] = time.time()
        self._save_memory()

    def add_concept(self, concept_name: str, concept_info: Dict[str, Any]) -> None:
        """
        Add or update a concept in long-term memory

        Args:
            concept_name (str): The name of the concept
            concept_info (dict): Information about the concept
        """
        self.memory['concepts'][concept_name] = {
            'info': concept_info,
            'timestamp': time.time()
        }
        self.memory['last_updated'] = time.time()
        self._save_memory()

    def get_facts(self) -> List[Dict[str, Any]]:
        """
        Get all facts from long-term memory

        Returns:
            list: All facts in long-term memory
        """
        return self.memory['facts']

    def get_concept(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a concept

        Args:
            concept_name (str): The name of the concept

        Returns:
            dict or None: Information about the concept, or None if not found
        """
        return self.memory['concepts'].get(concept_name)

    def get_all_concepts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all concepts from long-term memory

        Returns:
            dict: All concepts in long-term memory
        """
        return self.memory['concepts']

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search long-term memory for relevant information using LLM

        Args:
            query (str): The search query

        Returns:
            list: Relevant information from long-term memory
        """
        # Format the memory for the LLM
        facts_text = "\n".join([
            f"Fact {i+1}: {fact['content']}"
            for i, fact in enumerate(self.memory['facts'])
        ])

        concepts_text = "\n".join([
            f"Concept: {name}\nInfo: {concept['info']}"
            for name, concept in self.memory['concepts'].items()
        ])

        memory_text = f"""
        FACTS:
        {facts_text}

        CONCEPTS:
        {concepts_text}
        """

        prompt = f"""
        Given the following information in long-term memory:

        {memory_text}

        Find and extract information relevant to this query: "{query}"

        Return only the most relevant facts and concepts that directly address the query.
        If nothing is relevant, state that no relevant information was found.
        """

        try:
            response = self.groq_client.generate_text(prompt, max_tokens=300)
            result = self.groq_client.extract_text_from_response(response)

            # Return in a structured format
            return [{
                'content': result,
                'source': 'long_term_memory',
                'query': query
            }]
        except Exception as e:
            # Fallback to simple keyword matching if LLM fails
            results = []
            for fact in self.memory['facts']:
                if any(keyword in fact['content'].lower() for keyword in query.lower().split()):
                    results.append({
                        'content': fact['content'],
                        'source': 'long_term_memory',
                        'query': query
                    })

            for name, concept in self.memory['concepts'].items():
                if any(keyword in name.lower() or
                       (isinstance(concept['info'], str) and keyword in concept['info'].lower())
                       for keyword in query.lower().split()):
                    results.append({
                        'content': f"{name}: {concept['info']}",
                        'source': 'long_term_memory',
                        'query': query
                    })

            return results[:5]  # Return top 5 results


class EpisodicMemory:
    """
    Episodic memory for specific experiences and interactions.
    Enhanced with LLM for better episode understanding and retrieval.
    """

    def __init__(self):
        """Initialize episodic memory"""
        self.episodes = []
        self.groq_client = GroqClient()

    def add_episode(self, episode: Dict[str, Any]) -> None:
        """
        Add an episode to episodic memory

        Args:
            episode (dict): The episode to add
        """
        episode['timestamp'] = time.time()
        self.episodes.append(episode)

    def get_episodes(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the n most recent episodes

        Args:
            n (int): Number of episodes to retrieve

        Returns:
            list: The n most recent episodes
        """
        return self.episodes[-n:]

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """
        Get all episodes in episodic memory

        Returns:
            list: All episodes in episodic memory
        """
        return self.episodes

    def search_episodes(self, query: str) -> List[Dict[str, Any]]:
        """
        Search episodic memory for relevant episodes using LLM

        Args:
            query (str): The search query

        Returns:
            list: Relevant episodes from episodic memory
        """
        if not self.episodes:
            return []

        # Format the episodes for the LLM
        episodes_text = "\n\n".join([
            f"Episode {i+1}:\n" +
            "\n".join([f"{k}: {v}" for k, v in episode.items() if k != 'timestamp'])
            for i, episode in enumerate(self.episodes)
        ])

        prompt = f"""
        Given the following episodes in episodic memory:

        {episodes_text}

        Find and extract episodes relevant to this query: "{query}"

        Return only the most relevant episodes that directly address the query.
        If nothing is relevant, state that no relevant episodes were found.
        """

        try:
            response = self.groq_client.generate_text(prompt, max_tokens=300)
            result = self.groq_client.extract_text_from_response(response)

            # Parse the LLM response to identify which episodes were mentioned
            relevant_episodes = []
            for i, episode in enumerate(self.episodes):
                episode_marker = f"Episode {i+1}"
                if episode_marker in result:
                    relevant_episodes.append(episode)

            # If no episodes were clearly identified, return the LLM's analysis
            if not relevant_episodes:
                return [{
                    'content': result,
                    'source': 'episodic_memory_analysis',
                    'query': query
                }]

            return relevant_episodes
        except Exception as e:
            # Fallback to simple keyword matching if LLM fails
            results = []
            for episode in self.episodes:
                episode_text = str(episode)
                if any(keyword in episode_text.lower() for keyword in query.lower().split()):
                    results.append(episode)

            return results[:5]  # Return top 5 results


class AgentMemorySystem:
    """
    Integrated memory system combining different memory types.
    Enhanced with LLM for better memory integration and retrieval.
    """

    def __init__(self, storage_dir: str = "agent_memory"):
        """
        Initialize the agent memory system

        Args:
            storage_dir (str): Directory for persistent storage
        """
        os.makedirs(storage_dir, exist_ok=True)

        self.working_memory = WorkingMemory()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(os.path.join(storage_dir, "long_term_memory.json"))
        self.episodic = EpisodicMemory()
        self.groq_client = GroqClient()

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and update memory accordingly

        Args:
            user_input (str): The user's input

        Returns:
            dict: Processed input with metadata
        """
        # Add to short-term memory
        self.short_term.add({
            'role': 'user',
            'content': user_input
        })

        # Update working memory with current context
        self.working_memory.add(f"User said: {user_input}")

        # Use LLM to extract potential facts for long-term memory
        prompt = f"""
        Analyze this user input: "{user_input}"

        If it contains any factual information that would be useful to remember long-term,
        extract those facts in a concise format. If there are no facts to remember, respond with "No facts to store."
        """

        try:
            response = self.groq_client.generate_text(prompt, max_tokens=150)
            fact_extraction = self.groq_client.extract_text_from_response(response)

            if fact_extraction and "No facts to store" not in fact_extraction:
                self.long_term.add_fact(fact_extraction)
        except Exception:
            # Continue without fact extraction if LLM fails
            pass

        # Return processed input
        return {
            'text': user_input,
            'timestamp': time.time(),
            'processed': True
        }

    def store_response(self, response: str) -> None:
        """
        Store an agent response in memory

        Args:
            response (str): The agent's response
        """
        # Add to short-term memory
        self.short_term.add({
            'role': 'assistant',
            'content': response
        })

        # Update working memory
        self.working_memory.add(f"Assistant said: {response}")

    def create_episode(self, user_input: str, agent_response: str, metadata: Dict[str, Any] = None) -> None:
        """
        Create and store an episode in episodic memory

        Args:
            user_input (str): The user's input
            agent_response (str): The agent's response
            metadata (dict, optional): Additional metadata about the episode
        """
        episode = {
            'user_input': user_input,
            'agent_response': agent_response,
            'datetime': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.episodic.add_episode(episode)

    def retrieve_relevant_context(self, query: str, use_all_memory: bool = True) -> Dict[str, Any]:
        """
        Retrieve context relevant to a query from all memory systems

        Args:
            query (str): The query to retrieve context for
            use_all_memory (bool): Whether to use all memory systems

        Returns:
            dict: Relevant context from different memory systems
        """
        context = {}

        # Get context from working memory
        context['working_memory'] = self.working_memory.get_all()

        # Get context from short-term memory
        context['short_term'] = self.short_term.get_recent(5)

        if use_all_memory:
            # Get context from long-term memory
            context['long_term'] = self.long_term.search(query)

            # Get context from episodic memory
            context['episodic'] = self.episodic.search_episodes(query)

        # Use LLM to integrate and prioritize the context
        return self._integrate_context(query, context)

    def _integrate_context(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to integrate and prioritize context from different memory systems

        Args:
            query (str): The original query
            context (dict): Context from different memory systems

        Returns:
            dict: Integrated and prioritized context
        """
        # Format the context for the LLM
        context_text = ""

        if context.get('working_memory'):
            working_memory_text = "\n".join([
                f"- {item['content']}" for item in context['working_memory']
            ])
            context_text += f"WORKING MEMORY:\n{working_memory_text}\n\n"

        if context.get('short_term'):
            short_term_text = "\n".join([
                f"- {item['content']}" for item in context['short_term']
            ])
            context_text += f"SHORT-TERM MEMORY:\n{short_term_text}\n\n"

        if context.get('long_term'):
            long_term_text = "\n".join([
                f"- {item['content']}" for item in context['long_term']
            ])
            context_text += f"LONG-TERM MEMORY:\n{long_term_text}\n\n"

        if context.get('episodic'):
            episodic_text = "\n".join([
                f"- User: {episode.get('user_input', '')}\n  Assistant: {episode.get('agent_response', '')}"
                for episode in context['episodic']
            ])
            context_text += f"EPISODIC MEMORY:\n{episodic_text}\n\n"

        prompt = f"""
        Given this query: "{query}"

        And the following context from different memory systems:

        {context_text}

        Integrate and prioritize the most relevant information to answer the query.
        Focus on the most important and directly relevant details.
        Organize the information in a coherent way that would be most helpful for responding to the query.
        """

        try:
            response = self.groq_client.generate_text(prompt, max_tokens=500)
            integrated_context = self.groq_client.extract_text_from_response(response)

            # Return both the raw context and the integrated version
            return {
                'raw_context': context,
                'integrated_context': integrated_context,
                'query': query
            }
        except Exception as e:
            # Return the raw context if LLM integration fails
            return {
                'raw_context': context,
                'integrated_context': None,
                'query': query
            }


# Example usage
if __name__ == "__main__":
    # Create a memory system
    memory_system = AgentMemorySystem()

    # Process some inputs
    memory_system.process_input("My name is Alice and I like machine learning.")
    memory_system.store_response("Nice to meet you, Alice! What aspects of machine learning interest you the most?")

    memory_system.process_input("I'm particularly interested in natural language processing and transformers.")
    memory_system.store_response("That's a fascinating area! Transformer models have revolutionized NLP in recent years.")

    # Create an episode
    memory_system.create_episode(
        "Can you recommend some resources for learning about transformers?",
        "Certainly! I'd recommend the 'Attention is All You Need' paper, Hugging Face's transformers library, and the Illustrated Transformer blog post by Jay Alammar.",
        {"topic": "transformers", "intent": "resource_request"}
    )

    # Retrieve context for a query
    context = memory_system.retrieve_relevant_context("What is Alice interested in?")

    print("Integrated Context:")
    print(context.get('integrated_context', 'No integrated context available.'))
    print("\n" + "-" * 50 + "\n")

    # Test working memory summarization
    summary = memory_system.working_memory.summarize()
    print("Working Memory Summary:")
    print(summary)
