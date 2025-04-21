"""
Memory Types Implementation
--------------------------
This file contains implementations of different memory types for AI agents:
- Working Memory: For immediate context
- Short-Term Memory: For recent interactions
- Long-Term Memory: For persistent knowledge
- Episodic Memory: For specific experiences
"""

import json
import time
import os

class WorkingMemory:
    """
    Working memory for immediate context.
    
    This is like the "RAM" of an AI agent, holding the current context
    of the interaction. It has very limited capacity and persistence.
    """
    
    def __init__(self):
        """Initialize an empty working memory"""
        self.current_context = None
    
    def set_context(self, context):
        """
        Set the current working context
        
        Args:
            context: The context to store (can be any object)
        """
        self.current_context = context
    
    def get_context(self):
        """
        Retrieve the current working context
        
        Returns:
            The current context or None if not set
        """
        return self.current_context
    
    def clear(self):
        """Clear the working memory"""
        self.current_context = None
    
    def __str__(self):
        """String representation of working memory"""
        return f"WorkingMemory(context={self.current_context})"


class ShortTermMemory:
    """
    Short-term memory for recent interactions.
    
    This memory type stores a limited number of recent items,
    automatically removing the oldest items when capacity is reached.
    """
    
    def __init__(self, capacity=10):
        """
        Initialize short-term memory with a specific capacity
        
        Args:
            capacity (int): Maximum number of items to store
        """
        self.capacity = capacity
        self.memory = []
    
    def add(self, item):
        """
        Add an item to short-term memory
        
        Args:
            item: The item to add (can be any object)
        """
        self.memory.append(item)
        # Keep only the most recent items within capacity
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]
    
    def get_recent(self, n=None):
        """
        Get the n most recent items
        
        Args:
            n (int, optional): Number of recent items to retrieve.
                               If None, returns all items.
        
        Returns:
            list: The n most recent items
        """
        if n is None or n > len(self.memory):
            return self.memory
        return self.memory[-n:]
    
    def clear(self):
        """Clear the short-term memory"""
        self.memory = []
    
    def __len__(self):
        """Get the current number of items in memory"""
        return len(self.memory)
    
    def __str__(self):
        """String representation of short-term memory"""
        return f"ShortTermMemory(items={len(self.memory)}/{self.capacity})"


class LongTermMemory:
    """
    Long-term memory for persistent storage.
    
    This memory type provides persistent storage across sessions
    by saving to a JSON file. It uses a key-value structure for storage.
    """
    
    def __init__(self, file_path="long_term_memory.json"):
        """
        Initialize long-term memory with a specific file path
        
        Args:
            file_path (str): Path to the JSON file for persistent storage
        """
        self.file_path = file_path
        self.memory = self._load_memory()
    
    def _load_memory(self):
        """
        Load memory from persistent storage
        
        Returns:
            dict: The loaded memory or an empty dict if file doesn't exist
        """
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_memory(self):
        """Save memory to persistent storage"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path) or '.', exist_ok=True)
        with open(self.file_path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def store(self, key, value):
        """
        Store information in long-term memory
        
        Args:
            key (str): The key to store the value under
            value: The value to store (must be JSON serializable)
        """
        self.memory[key] = value
        self._save_memory()
    
    def retrieve(self, key):
        """
        Retrieve information from long-term memory
        
        Args:
            key (str): The key to retrieve
            
        Returns:
            The value associated with the key or None if not found
        """
        return self.memory.get(key)
    
    def forget(self, key):
        """
        Remove information from long-term memory
        
        Args:
            key (str): The key to remove
        """
        if key in self.memory:
            del self.memory[key]
            self._save_memory()
    
    def get_all(self):
        """
        Get all stored key-value pairs
        
        Returns:
            dict: All key-value pairs in memory
        """
        return self.memory.copy()
    
    def clear(self):
        """Clear all long-term memory"""
        self.memory = {}
        self._save_memory()
    
    def __len__(self):
        """Get the number of items in memory"""
        return len(self.memory)
    
    def __str__(self):
        """String representation of long-term memory"""
        return f"LongTermMemory(items={len(self.memory)}, path='{self.file_path}')"


class EpisodicMemory:
    """
    Episodic memory for storing specific experiences or interactions.
    
    This memory type records timestamped episodes with type, content,
    and optional metadata. It provides retrieval by ID, type, or timeframe.
    """
    
    def __init__(self):
        """Initialize an empty episodic memory"""
        self.episodes = []
    
    def record_episode(self, episode_type, content, metadata=None):
        """
        Record a new episode
        
        Args:
            episode_type (str): Type of episode (e.g., 'conversation', 'action')
            content: The content of the episode (can be any object)
            metadata (dict, optional): Additional metadata for the episode
            
        Returns:
            int: The ID of the recorded episode
        """
        episode = {
            "timestamp": time.time(),
            "type": episode_type,
            "content": content,
            "metadata": metadata or {}
        }
        self.episodes.append(episode)
        return len(self.episodes) - 1  # Return episode ID
    
    def get_episode(self, episode_id):
        """
        Retrieve a specific episode by ID
        
        Args:
            episode_id (int): The ID of the episode to retrieve
            
        Returns:
            dict: The episode or None if not found
        """
        if 0 <= episode_id < len(self.episodes):
            return self.episodes[episode_id]
        return None
    
    def get_episodes_by_type(self, episode_type):
        """
        Retrieve all episodes of a specific type
        
        Args:
            episode_type (str): The type of episodes to retrieve
            
        Returns:
            list: Episodes of the specified type
        """
        return [ep for ep in self.episodes if ep["type"] == episode_type]
    
    def get_episodes_in_timeframe(self, start_time, end_time=None):
        """
        Retrieve episodes within a specific timeframe
        
        Args:
            start_time (float): Start timestamp (in seconds since epoch)
            end_time (float, optional): End timestamp. If None, uses current time.
            
        Returns:
            list: Episodes within the specified timeframe
        """
        if end_time is None:
            end_time = time.time()
        
        return [
            ep for ep in self.episodes 
            if start_time <= ep["timestamp"] <= end_time
        ]
    
    def clear(self):
        """Clear all episodes"""
        self.episodes = []
    
    def __len__(self):
        """Get the number of episodes in memory"""
        return len(self.episodes)
    
    def __str__(self):
        """String representation of episodic memory"""
        return f"EpisodicMemory(episodes={len(self.episodes)})"


class AgentMemorySystem:
    """
    Integrated memory system for AI agents.
    
    This class combines working, short-term, long-term, and episodic
    memory into a unified system for AI agents.
    """
    
    def __init__(self, storage_dir="agent_memory"):
        """
        Initialize the memory system
        
        Args:
            storage_dir (str): Directory for persistent storage
        """
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize memory components
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory(capacity=20)
        self.long_term = LongTermMemory(os.path.join(storage_dir, "long_term_memory.json"))
        self.episodic = EpisodicMemory()
    
    def process_interaction(self, user_input, agent_response):
        """
        Process a new interaction
        
        Args:
            user_input (str): The user's input
            agent_response (str): The agent's response
        """
        # Update working memory with current context
        self.working.set_context({
            "user_input": user_input,
            "agent_response": agent_response
        })
        
        # Add to short-term memory
        self.short_term.add({
            "timestamp": time.time(),
            "user_input": user_input,
            "agent_response": agent_response
        })
        
        # Record as an episode
        self.episodic.record_episode(
            episode_type="conversation",
            content={
                "user_input": user_input,
                "agent_response": agent_response
            }
        )
    
    def get_conversation_context(self, n_turns=5):
        """
        Get recent conversation context
        
        Args:
            n_turns (int): Number of conversation turns to retrieve
            
        Returns:
            list: Recent conversation turns
        """
        return self.short_term.get_recent(n_turns)
    
    def store_fact(self, key, value):
        """
        Store a fact in long-term memory
        
        Args:
            key (str): The key to store the value under
            value: The value to store
        """
        self.long_term.store(key, value)
    
    def retrieve_fact(self, key):
        """
        Retrieve a fact from long-term memory
        
        Args:
            key (str): The key to retrieve
            
        Returns:
            The value associated with the key or None if not found
        """
        return self.long_term.retrieve(key)
    
    def save_user_preference(self, preference_name, preference_value):
        """
        Save a user preference
        
        Args:
            preference_name (str): Name of the preference
            preference_value: Value of the preference
        """
        preferences = self.long_term.retrieve("user_preferences") or {}
        preferences[preference_name] = preference_value
        self.long_term.store("user_preferences", preferences)
    
    def get_user_preferences(self):
        """
        Get all user preferences
        
        Returns:
            dict: User preferences
        """
        return self.long_term.retrieve("user_preferences") or {}
    
    def __str__(self):
        """String representation of the memory system"""
        return (
            f"AgentMemorySystem(\n"
            f"  working={self.working},\n"
            f"  short_term={self.short_term},\n"
            f"  long_term={self.long_term},\n"
            f"  episodic={self.episodic}\n"
            f")"
        )


# Simple demonstration if run directly
if __name__ == "__main__":
    print("Memory Systems Demo")
    print("------------------")
    
    # Create a memory system
    memory = AgentMemorySystem(storage_dir="demo_memory")
    
    # Process some interactions
    memory.process_interaction(
        user_input="Hello, my name is Alice.",
        agent_response="Hi Alice! How can I help you today?"
    )
    
    memory.process_interaction(
        user_input="I'm interested in learning about AI.",
        agent_response="That's great! AI is a fascinating field. What specific aspects are you interested in?"
    )
    
    memory.process_interaction(
        user_input="I want to learn about neural networks.",
        agent_response="Neural networks are a fundamental part of modern AI. They're inspired by the human brain and consist of layers of interconnected nodes."
    )
    
    # Store some facts
    memory.store_fact("user_name", "Alice")
    memory.store_fact("user_interests", ["AI", "neural networks"])
    
    # Save a user preference
    memory.save_user_preference("notification_frequency", "daily")
    
    # Display the memory state
    print("\nMemory System State:")
    print(memory)
    
    # Display conversation context
    print("\nRecent Conversation:")
    for turn in memory.get_conversation_context():
        print(f"User: {turn['user_input']}")
        print(f"Agent: {turn['agent_response']}")
        print()
    
    # Display stored facts
    print("Stored Facts:")
    print(f"User name: {memory.retrieve_fact('user_name')}")
    print(f"User interests: {memory.retrieve_fact('user_interests')}")
    
    # Display user preferences
    print("\nUser Preferences:")
    print(memory.get_user_preferences())
