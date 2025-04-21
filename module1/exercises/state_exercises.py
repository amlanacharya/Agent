"""
State Management Exercises
-----------------------
This file contains solutions for the state management exercises in Module 1, Lesson 3.
"""

import json
import time
import os
from datetime import datetime

class ConversationMemorySystem:
    """Solution for Exercise 1: Implement a Conversation Memory System"""
    
    def __init__(self, max_history=100):
        """
        Initialize the conversation memory system
        
        Args:
            max_history (int): Maximum number of messages to store
        """
        self.max_history = max_history
        self.conversations = []
        self.topics = {}  # Map of topic -> list of message indices
    
    def add_message(self, role, content, topics=None):
        """
        Add a message to the conversation history
        
        Args:
            role (str): The role of the speaker (user or agent)
            content (str): The message content
            topics (list, optional): List of topics this message relates to
        
        Returns:
            int: The index of the added message
        """
        # Create message object
        message = {
            "id": len(self.conversations),
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "topics": topics or []
        }
        
        # Add to conversations
        self.conversations.append(message)
        
        # Enforce max history
        if len(self.conversations) > self.max_history:
            removed = self.conversations.pop(0)
            # Clean up topic references to the removed message
            for topic in removed["topics"]:
                if topic in self.topics and 0 in self.topics[topic]:
                    self.topics[topic].remove(0)
                    
            # Adjust indices in topics
            for topic, indices in self.topics.items():
                self.topics[topic] = [idx - 1 for idx in indices if idx > 0]
        
        # Add to topics
        for topic in message["topics"]:
            if topic not in self.topics:
                self.topics[topic] = []
            self.topics[topic].append(message["id"])
        
        return message["id"]
    
    def get_conversation_history(self, limit=None, start_index=0):
        """
        Get conversation history
        
        Args:
            limit (int, optional): Maximum number of messages to return
            start_index (int, optional): Index to start from
            
        Returns:
            list: Conversation messages
        """
        if limit is None:
            return self.conversations[start_index:]
        return self.conversations[start_index:start_index + limit]
    
    def get_messages_by_topic(self, topic):
        """
        Get messages related to a specific topic
        
        Args:
            topic (str): The topic to filter by
            
        Returns:
            list: Messages related to the topic
        """
        if topic not in self.topics:
            return []
        
        return [self.conversations[idx] for idx in self.topics[topic] if idx < len(self.conversations)]
    
    def get_messages_by_date(self, start_date, end_date=None):
        """
        Get messages within a date range
        
        Args:
            start_date (float): Start timestamp
            end_date (float, optional): End timestamp (defaults to now)
            
        Returns:
            list: Messages within the date range
        """
        if end_date is None:
            end_date = time.time()
        
        return [msg for msg in self.conversations 
                if start_date <= msg["timestamp"] <= end_date]
    
    def summarize_conversation(self, num_messages=None):
        """
        Generate a simple summary of the conversation
        
        Args:
            num_messages (int, optional): Number of messages to summarize
            
        Returns:
            dict: Summary statistics
        """
        messages = self.get_conversation_history(limit=num_messages)
        
        # Count messages by role
        role_counts = {}
        for msg in messages:
            role = msg["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # Count topics
        topic_counts = {}
        for msg in messages:
            for topic in msg["topics"]:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Calculate time span
        if messages:
            start_time = messages[0]["timestamp"]
            end_time = messages[-1]["timestamp"]
            duration = end_time - start_time
        else:
            duration = 0
        
        return {
            "message_count": len(messages),
            "role_distribution": role_counts,
            "top_topics": sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "duration_seconds": duration
        }
    
    def clear_history(self):
        """Clear all conversation history"""
        self.conversations = []
        self.topics = {}


class UserProfileManager:
    """Solution for Exercise 2: Build a User Profile Manager"""
    
    def __init__(self, storage_path="user_profiles.json"):
        """
        Initialize the user profile manager
        
        Args:
            storage_path (str): Path to the storage file
        """
        self.storage_path = storage_path
        self.profiles = self._load_profiles()
        self.current_user_id = None
        self.behavior_log = []  # Track user behaviors for preference suggestions
    
    def _load_profiles(self):
        """
        Load profiles from storage
        
        Returns:
            dict: User profiles
        """
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_profiles(self):
        """Save profiles to storage"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        
        with open(self.storage_path, 'w') as f:
            json.dump(self.profiles, f, indent=2)
    
    def create_user(self, user_id, name, initial_preferences=None):
        """
        Create a new user profile
        
        Args:
            user_id (str): Unique identifier for the user
            name (str): User's name
            initial_preferences (dict, optional): Initial preference settings
            
        Returns:
            bool: True if created, False if user already exists
        """
        if user_id in self.profiles:
            return False
        
        self.profiles[user_id] = {
            "name": name,
            "created_at": time.time(),
            "last_active": time.time(),
            "preferences": initial_preferences or {
                "theme": "light",
                "notification_frequency": "daily",
                "task_sort_order": "deadline"
            },
            "behavior_patterns": {},
            "suggested_preferences": {}
        }
        
        self._save_profiles()
        return True
    
    def set_current_user(self, user_id):
        """
        Set the current active user
        
        Args:
            user_id (str): The user ID to set as current
            
        Returns:
            bool: True if successful, False if user doesn't exist
        """
        if user_id not in self.profiles:
            return False
        
        self.current_user_id = user_id
        self.profiles[user_id]["last_active"] = time.time()
        self._save_profiles()
        return True
    
    def get_current_user(self):
        """
        Get the current user's profile
        
        Returns:
            dict: The current user's profile or None if no user is set
        """
        if not self.current_user_id:
            return None
        return self.profiles.get(self.current_user_id)
    
    def update_preferences(self, preferences, user_id=None):
        """
        Update user preferences
        
        Args:
            preferences (dict): Preferences to update
            user_id (str, optional): User ID to update (defaults to current user)
            
        Returns:
            bool: True if successful, False otherwise
        """
        target_id = user_id or self.current_user_id
        if not target_id or target_id not in self.profiles:
            return False
        
        self.profiles[target_id]["preferences"].update(preferences)
        self.profiles[target_id]["last_active"] = time.time()
        self._save_profiles()
        return True
    
    def get_preference(self, preference_key, context=None, user_id=None):
        """
        Get a specific user preference, optionally considering context
        
        Args:
            preference_key (str): The preference to retrieve
            context (dict, optional): Context that might affect the preference
            user_id (str, optional): User ID to query (defaults to current user)
            
        Returns:
            The preference value or None if not found
        """
        target_id = user_id or self.current_user_id
        if not target_id or target_id not in self.profiles:
            return None
        
        # Get the base preference
        base_preference = self.profiles[target_id]["preferences"].get(preference_key)
        
        # If no context, return the base preference
        if not context:
            return base_preference
        
        # Check for context-specific overrides
        # This is a simplified implementation - a real system would have more sophisticated logic
        if "time_of_day" in context and preference_key == "theme":
            # Example: Switch to dark theme in the evening
            hour = datetime.fromtimestamp(time.time()).hour
            if hour >= 20 or hour < 6:  # Evening/night hours
                return "dark"
        
        return base_preference
    
    def log_behavior(self, behavior_type, details):
        """
        Log user behavior for preference suggestions
        
        Args:
            behavior_type (str): Type of behavior (e.g., "task_creation", "theme_switch")
            details (dict): Details about the behavior
        """
        if not self.current_user_id:
            return
        
        # Log the behavior
        log_entry = {
            "type": behavior_type,
            "timestamp": time.time(),
            "details": details
        }
        
        self.behavior_log.append(log_entry)
        
        # Update behavior patterns in user profile
        patterns = self.profiles[self.current_user_id]["behavior_patterns"]
        if behavior_type not in patterns:
            patterns[behavior_type] = {"count": 0, "details": []}
        
        patterns[behavior_type]["count"] += 1
        patterns[behavior_type]["details"].append({
            "timestamp": time.time(),
            "data": details
        })
        
        # Keep only the 10 most recent details
        patterns[behavior_type]["details"] = patterns[behavior_type]["details"][-10:]
        
        # Generate preference suggestions based on behavior
        self._generate_suggestions()
        
        self._save_profiles()
    
    def _generate_suggestions(self):
        """Generate preference suggestions based on user behavior"""
        if not self.current_user_id:
            return
        
        user = self.profiles[self.current_user_id]
        patterns = user["behavior_patterns"]
        suggestions = {}
        
        # Example: Suggest dark theme if user frequently uses the app at night
        night_usage = [entry for entry in self.behavior_log 
                      if datetime.fromtimestamp(entry["timestamp"]).hour >= 20
                      or datetime.fromtimestamp(entry["timestamp"]).hour < 6]
        
        if len(night_usage) > 5 and user["preferences"].get("theme") != "dark":
            suggestions["theme"] = {
                "value": "dark",
                "reason": "You often use the app during evening hours",
                "confidence": 0.7
            }
        
        # Example: Suggest weekly notifications if user ignores daily ones
        if "notification_interaction" in patterns:
            ignored_count = sum(1 for entry in patterns["notification_interaction"]["details"]
                              if entry["data"].get("action") == "ignored")
            
            if ignored_count > 3 and user["preferences"].get("notification_frequency") == "daily":
                suggestions["notification_frequency"] = {
                    "value": "weekly",
                    "reason": "You often ignore daily notifications",
                    "confidence": 0.8
                }
        
        user["suggested_preferences"] = suggestions
    
    def get_suggestions(self, user_id=None):
        """
        Get preference suggestions for a user
        
        Args:
            user_id (str, optional): User ID to query (defaults to current user)
            
        Returns:
            dict: Suggested preferences
        """
        target_id = user_id or self.current_user_id
        if not target_id or target_id not in self.profiles:
            return {}
        
        return self.profiles[target_id]["suggested_preferences"]


class TaskStateManager:
    """Solution for Exercise 3: Develop a Task State Manager"""
    
    def __init__(self, storage_path="tasks.json"):
        """
        Initialize the task state manager
        
        Args:
            storage_path (str): Path to the storage file
        """
        self.storage_path = storage_path
        self.tasks = self._load_tasks()
        self.next_id = max([int(task_id.split('-')[1]) for task_id in self.tasks.keys()], default=0) + 1
        self.change_log = []
    
    def _load_tasks(self):
        """
        Load tasks from storage
        
        Returns:
            dict: Tasks indexed by ID
        """
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_tasks(self):
        """Save tasks to storage"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        
        with open(self.storage_path, 'w') as f:
            json.dump(self.tasks, f, indent=2)
    
    def create_task(self, description, **kwargs):
        """
        Create a new task
        
        Args:
            description (str): Task description
            **kwargs: Additional task properties
            
        Returns:
            str: ID of the created task
        """
        # Validate required fields
        if not description:
            raise ValueError("Task description is required")
        
        # Generate task ID
        task_id = f"task-{self.next_id:03d}"
        self.next_id += 1
        
        # Create task object with defaults
        task = {
            "id": task_id,
            "description": description,
            "created_at": time.time(),
            "updated_at": time.time(),
            "status": "pending",
            "priority": "medium",
            "tags": []
        }
        
        # Update with any provided properties
        task.update(kwargs)
        
        # Validate task object
        self._validate_task(task)
        
        # Add to tasks dictionary
        self.tasks[task_id] = task
        
        # Log the change
        self._log_change("create", task_id, None, task)
        
        # Save to storage
        self._save_tasks()
        
        return task_id
    
    def get_task(self, task_id):
        """
        Get a task by ID
        
        Args:
            task_id (str): The task ID
            
        Returns:
            dict: The task or None if not found
        """
        return self.tasks.get(task_id)
    
    def update_task(self, task_id, **updates):
        """
        Update a task
        
        Args:
            task_id (str): The task ID
            **updates: Properties to update
            
        Returns:
            bool: True if successful, False if task not found
        """
        if task_id not in self.tasks:
            return False
        
        # Get the current task state
        old_task = self.tasks[task_id].copy()
        
        # Update the task
        self.tasks[task_id].update(updates)
        
        # Always update the updated_at timestamp
        self.tasks[task_id]["updated_at"] = time.time()
        
        # Validate the updated task
        try:
            self._validate_task(self.tasks[task_id])
        except ValueError as e:
            # Restore the old state if validation fails
            self.tasks[task_id] = old_task
            raise e
        
        # Log the change
        self._log_change("update", task_id, old_task, self.tasks[task_id])
        
        # Save to storage
        self._save_tasks()
        
        return True
    
    def delete_task(self, task_id):
        """
        Delete a task
        
        Args:
            task_id (str): The task ID
            
        Returns:
            bool: True if successful, False if task not found
        """
        if task_id not in self.tasks:
            return False
        
        # Store the task before deletion for the log
        old_task = self.tasks[task_id].copy()
        
        # Delete the task
        del self.tasks[task_id]
        
        # Log the change
        self._log_change("delete", task_id, old_task, None)
        
        # Save to storage
        self._save_tasks()
        
        return True
    
    def get_all_tasks(self):
        """
        Get all tasks
        
        Returns:
            list: All tasks
        """
        return list(self.tasks.values())
    
    def filter_tasks(self, **filters):
        """
        Filter tasks by criteria
        
        Args:
            **filters: Criteria to filter by
            
        Returns:
            list: Filtered tasks
        """
        result = self.get_all_tasks()
        
        for key, value in filters.items():
            if key == "tags" and isinstance(value, list):
                # Special handling for tags - match any tag in the list
                result = [task for task in result if any(tag in task.get("tags", []) for tag in value)]
            elif key == "created_after" and isinstance(value, (int, float)):
                # Filter by creation time
                result = [task for task in result if task.get("created_at", 0) > value]
            elif key == "created_before" and isinstance(value, (int, float)):
                # Filter by creation time
                result = [task for task in result if task.get("created_at", 0) < value]
            elif key == "updated_after" and isinstance(value, (int, float)):
                # Filter by update time
                result = [task for task in result if task.get("updated_at", 0) > value]
            elif key == "text_search" and isinstance(value, str):
                # Search in description
                result = [task for task in result if value.lower() in task.get("description", "").lower()]
            else:
                # Standard equality filter
                result = [task for task in result if task.get(key) == value]
        
        return result
    
    def sort_tasks(self, key="created_at", reverse=False):
        """
        Sort tasks by a specific key
        
        Args:
            key (str): The key to sort by
            reverse (bool): Whether to sort in reverse order
            
        Returns:
            list: Sorted tasks
        """
        tasks = self.get_all_tasks()
        
        # Handle special case for due_date which might be None
        if key == "due_date":
            # Put tasks with no due date at the end
            tasks_with_due_date = [task for task in tasks if task.get("due_date")]
            tasks_without_due_date = [task for task in tasks if not task.get("due_date")]
            
            sorted_tasks = sorted(tasks_with_due_date, key=lambda x: x.get(key), reverse=reverse)
            
            if reverse:
                # For reverse sort, put tasks without due date first
                return tasks_without_due_date + sorted_tasks
            else:
                # For normal sort, put tasks without due date last
                return sorted_tasks + tasks_without_due_date
        
        # Normal sorting
        return sorted(tasks, key=lambda x: x.get(key, 0), reverse=reverse)
    
    def get_change_log(self, limit=None):
        """
        Get the task change log
        
        Args:
            limit (int, optional): Maximum number of entries to return
            
        Returns:
            list: Change log entries
        """
        if limit is None:
            return self.change_log
        return self.change_log[-limit:]
    
    def _validate_task(self, task):
        """
        Validate a task object
        
        Args:
            task (dict): The task to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Required fields
        required_fields = ["id", "description", "status"]
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Task is missing required field: {field}")
        
        # Status validation
        valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
        if task["status"] not in valid_statuses:
            raise ValueError(f"Invalid status: {task['status']}. Must be one of {valid_statuses}")
        
        # Priority validation
        if "priority" in task:
            valid_priorities = ["low", "medium", "high"]
            if task["priority"] not in valid_priorities:
                raise ValueError(f"Invalid priority: {task['priority']}. Must be one of {valid_priorities}")
        
        # Tags validation
        if "tags" in task and not isinstance(task["tags"], list):
            raise ValueError("Tags must be a list")
    
    def _log_change(self, action, task_id, old_state, new_state):
        """
        Log a task change
        
        Args:
            action (str): The action performed (create, update, delete)
            task_id (str): The task ID
            old_state (dict): The previous state of the task
            new_state (dict): The new state of the task
        """
        log_entry = {
            "action": action,
            "task_id": task_id,
            "timestamp": time.time(),
            "old_state": old_state,
            "new_state": new_state
        }
        
        self.change_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.change_log) > 1000:
            self.change_log = self.change_log[-1000:]


# Example usage
if __name__ == "__main__":
    print("=== Conversation Memory System Example ===")
    conversation = ConversationMemorySystem()
    
    # Add some messages with topics
    conversation.add_message("user", "Hello, I need help with my tasks", ["greeting", "tasks"])
    conversation.add_message("agent", "I can help you manage your tasks. What would you like to do?", ["tasks"])
    conversation.add_message("user", "I need to create a report for tomorrow", ["tasks", "reports", "deadlines"])
    
    # Get messages by topic
    task_messages = conversation.get_messages_by_topic("tasks")
    print(f"Messages about tasks: {len(task_messages)}")
    
    # Get conversation summary
    summary = conversation.summarize_conversation()
    print(f"Conversation summary: {summary}")
    
    print("\n=== User Profile Manager Example ===")
    profile_manager = UserProfileManager()
    
    # Create a user
    profile_manager.create_user("user123", "John Doe")
    profile_manager.set_current_user("user123")
    
    # Update preferences
    profile_manager.update_preferences({"theme": "dark"})
    
    # Log some behaviors
    profile_manager.log_behavior("app_usage", {"time_of_day": "evening"})
    profile_manager.log_behavior("notification_interaction", {"action": "ignored"})
    profile_manager.log_behavior("notification_interaction", {"action": "ignored"})
    profile_manager.log_behavior("notification_interaction", {"action": "ignored"})
    
    # Get suggestions
    suggestions = profile_manager.get_suggestions()
    print(f"Preference suggestions: {suggestions}")
    
    print("\n=== Task State Manager Example ===")
    task_manager = TaskStateManager()
    
    # Create tasks
    task1 = task_manager.create_task(
        "Complete quarterly report",
        priority="high",
        tags=["work", "reports"],
        due_date="2023-06-30"
    )
    
    task2 = task_manager.create_task(
        "Buy groceries",
        priority="medium",
        tags=["personal", "shopping"]
    )
    
    # Update a task
    task_manager.update_task(task1, status="in_progress")
    
    # Filter tasks
    high_priority = task_manager.filter_tasks(priority="high")
    print(f"High priority tasks: {len(high_priority)}")
    
    # Sort tasks
    sorted_tasks = task_manager.sort_tasks(key="priority", reverse=True)
    print(f"Tasks sorted by priority (highest first): {[task['description'] for task in sorted_tasks]}")
    
    # Get change log
    changes = task_manager.get_change_log()
    print(f"Change log entries: {len(changes)}")
