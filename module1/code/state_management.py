"""
State Management Implementation
----------------------------
This file contains implementations of various state management patterns for AI agents.
Use this as a starting point for the exercises in Module 1, Lesson 3.
"""

import json
import time
import copy
import uuid
import os

class ShortTermMemory:
    """
    A simple short-term memory system with limited capacity.
    Stores recent items and automatically removes oldest items when capacity is reached.
    """
    
    def __init__(self, capacity=10):
        """
        Initialize short-term memory with a specified capacity
        
        Args:
            capacity (int): Maximum number of items to store
        """
        self.capacity = capacity
        self.memory = []
    
    def add(self, item):
        """
        Add an item to short-term memory, removing oldest if at capacity
        
        Args:
            item: The item to add to memory
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)  # Remove oldest item
        self.memory.append(item)
    
    def get_recent(self, n=None):
        """
        Get the n most recent items (or all if n is None)
        
        Args:
            n (int, optional): Number of recent items to retrieve
            
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
        """Return the current number of items in memory"""
        return len(self.memory)


class LongTermMemory:
    """
    A persistent storage system for agent memory that persists across sessions.
    """
    
    def __init__(self, storage_path="agent_memory.json"):
        """
        Initialize long-term memory with a storage path
        
        Args:
            storage_path (str): Path to the storage file
        """
        self.storage_path = storage_path
        self.memory = self._load_or_initialize()
    
    def _load_or_initialize(self):
        """
        Load existing memory or initialize a new one
        
        Returns:
            dict: The loaded or initialized memory
        """
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "user_profiles": {},
                "past_sessions": [],
                "learned_preferences": {},
                "frequent_tasks": {}
            }
    
    def store(self, category, key, value):
        """
        Store information in a specific category
        
        Args:
            category (str): The category to store in
            key (str): The key for the stored value
            value: The value to store
        """
        if category not in self.memory:
            self.memory[category] = {}
        self.memory[category][key] = value
        self._save()
    
    def retrieve(self, category, key=None, default=None):
        """
        Retrieve information from memory
        
        Args:
            category (str): The category to retrieve from
            key (str, optional): The specific key to retrieve
            default: Value to return if key is not found
            
        Returns:
            The retrieved value or default
        """
        if category not in self.memory:
            return default
        if key is None:
            return self.memory[category]
        return self.memory[category].get(key, default)
    
    def _save(self):
        """Save memory to persistent storage"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def clear_category(self, category):
        """
        Clear a specific category of memory
        
        Args:
            category (str): The category to clear
        """
        if category in self.memory:
            self.memory[category] = {}
            self._save()
    
    def clear_all(self):
        """Clear all memory"""
        self.memory = {
            "user_profiles": {},
            "past_sessions": [],
            "learned_preferences": {},
            "frequent_tasks": {}
        }
        self._save()


class EpisodicMemory:
    """
    Records complete interaction sessions with the agent.
    """
    
    def __init__(self):
        """Initialize episodic memory with an empty session list and a new current session"""
        self.sessions = []
        self.current_session = self._create_new_session()
    
    def _create_new_session(self):
        """
        Create a new session object
        
        Returns:
            dict: A new session object
        """
        return {
            "id": str(uuid.uuid4()),
            "start_time": time.time(),
            "interactions": [],
            "summary": None
        }
    
    def add_interaction(self, user_input, agent_response=None):
        """
        Add an interaction to the current session
        
        Args:
            user_input (str): The user's input
            agent_response (str, optional): The agent's response
        """
        self.current_session["interactions"].append({
            "timestamp": time.time(),
            "user_input": user_input,
            "agent_response": agent_response
        })
    
    def update_last_response(self, agent_response):
        """
        Update the agent response for the last interaction
        
        Args:
            agent_response: The agent's response to add
        """
        if self.current_session["interactions"]:
            self.current_session["interactions"][-1]["agent_response"] = agent_response
    
    def end_session(self, summary=None):
        """
        End the current session and store it
        
        Args:
            summary (str, optional): A summary of the session
        """
        self.current_session["end_time"] = time.time()
        self.current_session["duration"] = self.current_session["end_time"] - self.current_session["start_time"]
        self.current_session["summary"] = summary
        self.sessions.append(self.current_session)
        
        # Start a new session
        self.current_session = self._create_new_session()
    
    def get_session(self, session_id):
        """
        Retrieve a specific session by ID
        
        Args:
            session_id (str): The ID of the session to retrieve
            
        Returns:
            dict: The session object or None if not found
        """
        for session in self.sessions:
            if session["id"] == session_id:
                return session
        return None
    
    def get_recent_sessions(self, n=5):
        """
        Get the n most recent sessions
        
        Args:
            n (int): Number of recent sessions to retrieve
            
        Returns:
            list: The n most recent sessions
        """
        return sorted(self.sessions, key=lambda s: s["start_time"], reverse=True)[:n]
    
    def get_current_session(self):
        """
        Get the current active session
        
        Returns:
            dict: The current session object
        """
        return self.current_session


class AgentStateManager:
    """
    A comprehensive state manager for agents that combines different types of state.
    """
    
    def __init__(self, storage_dir="agent_data"):
        """
        Initialize the state manager
        
        Args:
            storage_dir (str): Directory for persistent storage
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize memory systems
        self.short_term = ShortTermMemory(capacity=20)
        self.long_term = LongTermMemory(os.path.join(storage_dir, "long_term_memory.json"))
        self.episodic = EpisodicMemory()
        
        # Initialize application state
        self.app_state = {
            "tasks": [],
            "categories": ["work", "personal", "health", "finance"],
            "views": ["all", "today", "upcoming", "completed"]
        }
        
        # Initialize user profile
        self.user_profile = {
            "name": None,
            "preferences": {
                "notification_frequency": "daily",
                "task_sort_order": "deadline",
                "theme": "light"
            }
        }
        
        # Load saved state if available
        self._load_state()
    
    def update_conversation(self, role, content):
        """
        Add a new message to conversation memory
        
        Args:
            role (str): The role of the speaker (user or agent)
            content (str): The message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        # Add to short-term memory
        self.short_term.add(message)
        
        # Add to episodic memory
        if role == "user":
            self.episodic.add_interaction(content)
        else:
            self.episodic.update_last_response(content)
    
    def get_conversation_history(self, max_messages=None):
        """
        Get recent conversation history
        
        Args:
            max_messages (int, optional): Maximum number of messages to retrieve
            
        Returns:
            list: Recent conversation messages
        """
        return self.short_term.get_recent(max_messages)
    
    def update_user_profile(self, **kwargs):
        """
        Update user profile with provided key-value pairs
        
        Args:
            **kwargs: Key-value pairs to update in the profile
        """
        for key, value in kwargs.items():
            if key == "preferences" and isinstance(value, dict):
                self.user_profile["preferences"].update(value)
            else:
                self.user_profile[key] = value
        
        # Store in long-term memory
        self.long_term.store("user_profiles", "current", self.user_profile)
    
    def add_task(self, task):
        """
        Add a new task
        
        Args:
            task (dict): The task to add
            
        Returns:
            str: The ID of the added task
        """
        # Generate a unique ID if not provided
        if "id" not in task:
            task["id"] = f"task-{len(self.app_state['tasks']) + 1:03d}"
        
        # Add creation timestamp
        task["created_at"] = time.time()
        
        # Add to application state
        self.app_state["tasks"].append(task)
        
        return task["id"]
    
    def update_task(self, task_id, **updates):
        """
        Update an existing task
        
        Args:
            task_id (str): The ID of the task to update
            **updates: Key-value pairs to update in the task
            
        Returns:
            bool: True if the task was updated, False otherwise
        """
        for i, task in enumerate(self.app_state["tasks"]):
            if task["id"] == task_id:
                self.app_state["tasks"][i].update(updates)
                return True
        return False
    
    def get_tasks(self, filters=None):
        """
        Retrieve tasks, optionally filtered by criteria
        
        Args:
            filters (dict, optional): Criteria to filter tasks by
            
        Returns:
            list: Filtered tasks
        """
        if not filters:
            return self.app_state["tasks"]
        
        filtered_tasks = self.app_state["tasks"]
        
        for key, value in filters.items():
            filtered_tasks = [task for task in filtered_tasks if task.get(key) == value]
        
        return filtered_tasks
    
    def delete_task(self, task_id):
        """
        Delete a task by ID
        
        Args:
            task_id (str): The ID of the task to delete
            
        Returns:
            bool: True if the task was deleted, False otherwise
        """
        for i, task in enumerate(self.app_state["tasks"]):
            if task["id"] == task_id:
                del self.app_state["tasks"][i]
                return True
        return False
    
    def save_state(self):
        """Save the current state to persistent storage"""
        # Save application state
        app_state_path = os.path.join(self.storage_dir, "app_state.json")
        with open(app_state_path, 'w') as f:
            json.dump(self.app_state, f, indent=2)
        
        # User profile is already saved in long-term memory
        
        # Episodic memory is saved separately
        episodic_path = os.path.join(self.storage_dir, "episodic_memory.json")
        with open(episodic_path, 'w') as f:
            json.dump({
                "sessions": self.episodic.sessions,
                "current_session": self.episodic.current_session
            }, f, indent=2)
    
    def _load_state(self):
        """Load state from persistent storage"""
        # Load application state
        app_state_path = os.path.join(self.storage_dir, "app_state.json")
        try:
            with open(app_state_path, 'r') as f:
                self.app_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Use default if file doesn't exist or is invalid
        
        # Load user profile from long-term memory
        stored_profile = self.long_term.retrieve("user_profiles", "current")
        if stored_profile:
            self.user_profile = stored_profile
        
        # Load episodic memory
        episodic_path = os.path.join(self.storage_dir, "episodic_memory.json")
        try:
            with open(episodic_path, 'r') as f:
                episodic_data = json.load(f)
                self.episodic.sessions = episodic_data["sessions"]
                self.episodic.current_session = episodic_data["current_session"]
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Use default if file doesn't exist or is invalid


# Example usage
if __name__ == "__main__":
    # Create a state manager
    state_manager = AgentStateManager(storage_dir="agent_data_test")
    
    # Add some conversation messages
    state_manager.update_conversation("user", "Hello, can you help me organize my tasks?")
    state_manager.update_conversation("agent", "Of course! I'd be happy to help you organize your tasks.")
    state_manager.update_conversation("user", "Great, I need to track my project deadlines.")
    
    # Update user profile
    state_manager.update_user_profile(
        name="Alice",
        preferences={
            "notification_frequency": "weekly",
            "task_sort_order": "priority"
        }
    )
    
    # Add some tasks
    state_manager.add_task({
        "description": "Complete project proposal",
        "deadline": "2023-06-15",
        "priority": "high",
        "status": "in_progress"
    })
    
    state_manager.add_task({
        "description": "Schedule team meeting",
        "deadline": "2023-06-10",
        "priority": "medium",
        "status": "not_started"
    })
    
    # Save state
    state_manager.save_state()
    
    # Print some information
    print("Conversation History:")
    for message in state_manager.get_conversation_history():
        print(f"[{message['role']}]: {message['content']}")
    
    print("\nUser Profile:")
    print(f"Name: {state_manager.user_profile['name']}")
    print(f"Preferences: {state_manager.user_profile['preferences']}")
    
    print("\nTasks:")
    for task in state_manager.get_tasks():
        print(f"- {task['description']} (Priority: {task['priority']}, Status: {task['status']})")
